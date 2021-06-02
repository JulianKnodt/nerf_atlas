# Global runner for all NeRF methods.
# For convenience, we want all methods using NeRF to use this one file.

import numpy as np
import argparse
import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import random
from tqdm import trange
import json
import math

import src.loaders
import src.nerf as nerf
import src.utils as utils
import src.sdf as sdf
from src.utils import ( save_image, save_plot, CylinderGaussian, ConicGaussian )
from src.neural_blocks import ( Upsampler, SpatialEncoder )

def arguments():
  a = argparse.ArgumentParser()
  a.add_argument("-d", "--data", help="path to data", required=True)
  a.add_argument(
    "--data-kind", help="Kind of data to load",
    choices=["original", "single_video", "dnerf", "pixel-single"],
    default="original",
  )
  a.add_argument("--load", help="model to load from", type=str)
  a.add_argument(
    "--derive_kind", help="Attempt to derive the kind if a single file is given",
    action="store_false",
  )
  # various size arguments
  a.add_argument("--size", help="size to train at w/ upsampling", type=int, default=32)
  a.add_argument(
    "--render-size", help="size to render images at w/o upsampling", type=int, default=16
  )

  a.add_argument("--epochs", help="number of epochs to train for", type=int, default=30000)
  a.add_argument("--batch-size", help="size of each training batch", type=int, default=8)
  a.add_argument("--neural-upsample", help="add neural upsampling", action="store_true")
  a.add_argument("--crop", help="train with cropping", action="store_true")
  a.add_argument("--crop-size",help="what size to use while cropping",type=int, default=16)
  a.add_argument("--steps", help="Number of depth steps", type=int, default=64)
  # TODO type of crop strategy? Maybe can do random resizing?
  a.add_argument(
    "--mip", help="Use MipNeRF with different sampling", type=str,
    choices=["cone", "cylinder"],
  )
  a.add_argument("--sdf", help="Use a backing SDF", action="store_true")

  a. add_argument(
    "--feature-space",
    help="when using neural upsampling, what is the feature space size",
    type=int, default=32,
  )
  a.add_argument(
    "--model", help="which model do we want to use", type=str,
    choices=["tiny", "plain", "ae"], default="plain",
  )
  # this default for LR seems to work pretty well?
  a.add_argument(
    "-lr", "--learning-rate", help="learning rate", type=float, default=5e-4,
  )
  a.add_argument(
    "--valid-freq", help="how often validation images are generated", type=int, default=500,
  )
  a.add_argument("--seed", help="random seed to use", type=int, default=1337)
  a.add_argument("--decay", help="weight_decay value", type=float, default=0)
  a.add_argument("--notest", help="do not run test set", action="store_true")
  a.add_argument("--data-parallel", help="Use data parallel for the model", action="store_true")
  a.add_argument("--omit-bg", help="Omit bg with some probability", action="store_true")

  a.add_argument("--save", help="Where to save the model", type=str, default="models/model.pt")
  a.add_argument("--log", help="Where to save log of arguments", type=str, default="log.json")
  a.add_argument("--save-freq", help="# of epochs between saves", type=int, default=10_000)

  cam = a.add_argument_group("camera parameters")
  cam.add_argument("--near", help="near plane for camera", type=float, default=2)
  cam.add_argument("--far", help="far plane for camera", type=float, default=6)

  reporting = a.add_argument_group("reporting parameters")
  reporting.add_argument("-q", "--quiet", help="Silence tqdm (UNIMPL)", action="store_true")
  # TODO add more arguments here
  return a.parse_args()

# TODO better automatic device discovery here

device = "cpu"
if torch.cuda.is_available():
  device = torch.device("cuda:0")
  torch.cuda.set_device(device)

#torch.autograd.set_detect_anomaly(True)

def render(
  model,
  cam,
  crop,
  # how big should the image be
  size,
  args,

  device="cuda",
  with_noise=5e-3,

  times=None,
):
  batch_dims = len(cam)

  ii, jj = torch.meshgrid(
    torch.arange(size, device=device, dtype=torch.float),
    torch.arange(size, device=device, dtype=torch.float),
  )

  positions = torch.stack([ii.transpose(-1, -2), jj.transpose(-1, -2)], dim=-1)
  t,l,h,w = crop
  positions = positions[t:t+h,l:l+w,:]


  rays = cam.sample_positions(
    positions, size=size, with_noise=with_noise,
  )
  if times is not None: return model((rays, times))
  elif args.data_kind == "pixel-single": return model((rays, positions))
  return model(rays)

# loads the dataset
def load(args, training=True):
  assert(args.data is not None)
  kind = args.data_kind
  if args.derive_kind:
    if args.data.endswith(".mp4"): kind = "single_video"
    elif args.data.endswith(".jpg"): kind = "pixel-single"
    # TODO if single image use pixel

  if not args.neural_upsample: args.size = args.render_size
  size = args.size
  if kind == "original":
    return src.loaders.original(
      args.data, training=training, normalize=False, size=size, device=device
    )
  elif kind == "dnerf":
    return src.loaders.dnerf(
      args.data, training=training, normalize=False, size=size, device=device
    )
  elif kind == "single_video":
    return src.loaders.single_video(args.data)
  elif kind == "pixel-single":
    img, cam = src.loaders.single_image(args.data)
    setattr(args, "img", img)
    args.batch_size = 1
    return img, cam
  else:
    raise NotImplementedError(kind)

def sqr(x): return x * x

# train the model with a given camera and some labels (imgs or imgs+times)
def train(model, cam, labels, opt, args, sched=None):
  if args.epochs == 0: return
  t = trange(args.epochs)
  batch_size = args.batch_size
  times=None
  if args.data_kind == "dnerf":
    times = labels[-1]
    labels = labels[0]

  get_crop = lambda: (0,0, args.render_size, args.render_size)
  if args.crop:
    get_crop = lambda: (
      random.randint(0, args.render_size-args.crop_size),
      random.randint(0, args.render_size-args.crop_size),
      args.crop_size, args.crop_size,
    )

  for i in t:
    opt.zero_grad()

    idxs = random.sample(range(len(cam)), batch_size)
    #idxs = [0] * len(idxs) # DEBUG

    ts = None if times is None else times[idxs]
    c0,c1,c2,c3 = crop = get_crop()
    ref = labels[idxs][:, c0:c0+c2,c1:c1+c3, :]

    # omit items which are all darker with some likelihood.
    if args.omit_bg and (i % args.save_freq) != 0 and \
      ref.mean() + 0.3 < sqr(random.random()): continue

    out = render(model, cam[idxs], crop, size=args.render_size, times=ts, args=args)
    # TODO add config for this sqrt? It's optional.
    loss = F.mse_loss(out, ref)#.sqrt()
    l2_loss = loss.item()
    if args.sdf:
      eik = sdf.eikonal_loss(model.sdf.normals)
      loss = loss + eik
      t.set_postfix(
        l2=f"{l2_loss:.04f}", eik=f"{eik:.02f}",
        lr=f"{sched.get_last_lr()[0]:.1e}",
        refresh=False,
      )
    else:
      t.set_postfix(
        l2=f"{l2_loss:.04f}", lr=f"{sched.get_last_lr()[0]:.1e}", refresh=False,
      )
    loss.backward()
    opt.step()
    if sched is not None: sched.step()
    if i % args.valid_freq == 0:
      save_plot(f"outputs/valid_{i:05}.png", ref[0], out[0])
    if i % args.save_freq == 0 and i != 0: save(model, args)

def test(model, cam, labels, args):
  times = None
  model = model.eval()
  if args.data_kind == "dnerf":
    times = labels[-1]
    labels = labels[0]

  with torch.no_grad():
    for i in range(labels.shape[0]):
      ts = None if times is None else times[i:i+1, ...]
      exp = labels[i]
      got = torch.zeros_like(exp)
      acc = torch.zeros_like(got)
      if args.sdf: got_sdf = torch.zeros_like(got)
      N = math.ceil(args.render_size/args.crop_size)
      for x in range(N):
        for y in range(N):
          c0 = x * args.crop_size
          c1 = y * args.crop_size
          out = render(
            model, cam[i:i+1, ...], (c0,c1,args.crop_size,args.crop_size), size=args.render_size,
            with_noise=False, times=ts, args=args,
          ).squeeze(0)
          got[c0:c0+args.crop_size, c1:c1+args.crop_size, :] = out

          #weights = nerf.volumetric_integrate(
          #  model.nerf.density, model.nerf.density[..., None]
          #)
          weights = model.density * nerf.cumuprod_exclusive((1-model.density).clamp(min=1e-10))
          acc[c0:c0+args.crop_size, c1:c1+args.crop_size, :] = \
            weights.max(dim=0)[0][..., None]

      loss = F.mse_loss(got, exp)
      print(f"[{i:03}]: L2 {loss.item():.03f} PSNR {utils.mse2psnr(loss).item():.03f}")
      save_plot(f"outputs/out_{i:03}.png", exp, got)
      save_plot(f"outputs/acc_{i:03}.png", exp, acc)

def load_mip(args):
  if args.mip is None: return None
  elif args.mip == "cone": return ConicGaussian()
  elif args.mip == "cylinder": return CylinderGaussian()

  raise NotImplementedError(f"Unknown mip kind {args.mip}")

def load_model(args):
  if not args.neural_upsample: args.feature_space = 3
  mip = load_mip(args)
  kwargs = {
    "mip": mip,
    "out_features": args.feature_space,
    "device": device,
    "steps": args.steps,
    "t_near": args.near,
    "t_far": args.far,
    "per_pixel_latent_size": 64 if args.data_kind == "pixel-single" else 0,
    "per_point_latent_size": (1 + 3 + 128) if args.sdf else 0,
  }
  if args.model == "tiny":
    model = nerf.TinyNeRF(**kwargs).to(device)
  elif args.model == "plain":
    model = nerf.PlainNeRF(**kwargs).to(device)
  elif args.model == "ae":
    model = nerf.NeRFAE(**kwargs).to(device)
  else:
    raise NotImplementedError(args.model)

  # Add in a dynamic model if using dnerf with the underlying model.
  if args.data_kind == "dnerf":
    model = nerf.DynamicNeRF(model, device=device).to(device)
  if args.data_kind == "pixel-single":
    encoder = SpatialEncoder().to(device)
    # args.img is populated in load (single_image)
    model = nerf.SinglePixelNeRF(model, encoder=encoder, img=args.img, device=device).to(device)

  if args.sdf: model = sdf.SDFNeRF(model, sdf.SDF()).to(device)
  # tack on neural upsampling if specified
  if args.neural_upsample:
    upsampler =  Upsampler(
      in_size=args.render_size,
      out=args.size,

      in_features=args.feature_space,
      out_features=3,
    ).to(device)
    # stick a neural upsampling block afterwards
    model = nn.Sequential(model, upsampler, nn.Sigmoid())

  if args.data_parallel: model = nn.DataParallel(model)
  return model

def save(model, args):
  print(f"Saved to {args.save}")
  torch.save(model, args.save)
  if args.log is not None:
    with open(args.log, 'w') as f:
      json.dump(args.__dict__, f, indent=2)

def seed(s):
  torch.manual_seed(s)
  random.seed(s)
  np.random.seed(s)

def main():
  args = arguments()
  seed(args.seed)

  labels, cam = load(args)
  model = load_model(args) if args.load is None else torch.load(args.load)

  # for some reason AdamW doesn't seem to work here
  opt = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.decay)


  sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=1e-5)
  train(model, cam, labels, opt, args, sched=sched)

  if args.epochs != 0:
    save(model, args)

  if args.notest: return
  model.eval()

  test_labels, test_cam = load(args, training=False)
  test(model, test_cam, test_labels, args)

if __name__ == "__main__": main()

