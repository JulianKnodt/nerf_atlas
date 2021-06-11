# Global runner for all NeRF methods.
# For convenience, we want all methods using NeRF to use this one file.

import numpy as np
import argparse
import torch
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms.functional as TVF
import torch.nn as nn
import random
import json
import math
import matplotlib.pyplot as plt

from datetime import datetime
from tqdm import trange
from itertools import chain

import src.loaders
import src.nerf as nerf
import src.utils as utils
import src.sdf as sdf
from src.utils import ( save_image, save_plot, CylinderGaussian, ConicGaussian )
from src.neural_blocks import ( Upsampler, SpatialEncoder )

def arguments():
  a = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  a.add_argument("-d", "--data", help="path to data", required=True)
  a.add_argument(
    "--data-kind", help="Kind of data to load", default="original",
    choices=["original", "single_video", "dnerf", "pixel-single"],
  )
  a.add_argument(
    "--derive_kind", help="Attempt to derive the kind if a single file is given",
    action="store_false",
  )
  # various size arguments
  a.add_argument("--size", help="size to train at w/ upsampling", type=int, default=32)
  a.add_argument(
    "--render-size", help="size to render images at w/o upsampling", type=int, default=16
  )
  a.add_argument("--dnerfae", help="Use DNeRFAE on top of DNeRF", action="store_true")

  a.add_argument("--epochs", help="number of epochs to train for", type=int, default=30000)
  a.add_argument("--batch-size", help="size of each training batch", type=int, default=8)
  a.add_argument("--neural-upsample", help="add neural upsampling", action="store_true")
  a.add_argument("--crop", help="train with cropping", action="store_true")
  a.add_argument("--crop-size",help="what size to use while cropping",type=int, default=16)
  a.add_argument("--steps", help="Number of depth steps", type=int, default=64)
  a.add_argument(
    "--mip", help="Use MipNeRF with different sampling", type=str, choices=["cone", "cylinder"],
  )
  a.add_argument("--nerf-eikonal", help="Add eikonal loss for NeRF", action="store_true")
  a.add_argument("--fat-sigmoid", help="Use wider sigmoid activation for features", action="store_false")
  a.add_argument("--n-sparsify-alpha", help="Epochs for sparsifying alpha", type=int, default=0)
  a.add_argument("--sdf", help="Use a backing SDF", action="store_true")
  a.add_argument("--train-camera", help="Train camera parameters", action="store_true")
  a.add_argument("--blur", help="Blur before loss comparison", action="store_true")
  a.add_argument("--sharpen", help="Sharpen before loss comparison", action="store_true")

  a. add_argument(
    "--feature-space", help="when using neural upsampling, what is the feature space size",
    type=int, default=32,
  )
  a.add_argument(
    "--model", help="which model do we want to use", type=str,
    choices=["tiny", "plain", "ae"], default="plain",
  )
  # this default for LR seems to work pretty well?
  a.add_argument("-lr", "--learning-rate", help="learning rate", type=float, default=5e-4)
  a.add_argument("--seed", help="random seed to use", type=int, default=1337)
  a.add_argument("--decay", help="weight_decay value", type=float, default=0)
  a.add_argument("--notest", help="do not run test set", action="store_true")
  a.add_argument("--data-parallel", help="Use data parallel for the model", action="store_true")
  a.add_argument("--omit-bg", help="Omit black bg with some probability", action="store_true")
  a.add_argument("--l1-loss", help="Add l1 loss with output", action="store_true")
  a.add_argument("--no-l2-loss", help="Remove l2 loss", action="store_true")
  a.add_argument("--no-sched", help="Do not use a scheduler", action="store_true")
  a.add_argument("--serial-idxs", help="Train on images in serial", action="store_true")


  cam = a.add_argument_group("camera parameters")
  cam.add_argument("--near", help="near plane for camera", type=float, default=2)
  cam.add_argument("--far", help="far plane for camera", type=float, default=6)

  rprt = a.add_argument_group("reporting parameters")
  rprt.add_argument("-q", "--quiet", help="Silence tqdm", action="store_true")
  rprt.add_argument("--save", help="Where to save the model", type=str, default="models/model.pt")
  rprt.add_argument("--log", help="Where to save log of arguments", type=str, default="log.json")
  rprt.add_argument("--save-freq", help="# of epochs between saves", type=int, default=5000)
  rprt.add_argument(
    "--valid-freq", help="how often validation images are generated", type=int, default=500,
  )
  rprt.add_argument("--load", help="model to load from", type=str)

  return a.parse_args()

# TODO better automatic device discovery here
device = "cpu"
if torch.cuda.is_available():
  device = torch.device("cuda:0")
  torch.cuda.set_device(device)

# XXX DEBUG
#torch.autograd.set_detect_anomaly(True)

def render(
  model, cam, crop,
  # how big should the image be
  size,

  args,
  times=None, with_noise=0.1,
):
  ii, jj = torch.meshgrid(
    torch.arange(size, device=device, dtype=torch.float),
    torch.arange(size, device=device, dtype=torch.float),
  )

  positions = torch.stack([ii.transpose(-1, -2), jj.transpose(-1, -2)], dim=-1)
  t,l,h,w = crop
  positions = positions[t:t+h,l:l+w,:]

  rays = cam.sample_positions(positions, size=size, with_noise=with_noise)

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
  else: raise NotImplementedError(kind)

def sqr(x): return x * x

# train the model with a given camera and some labels (imgs or imgs+times)
def train(model, cam, labels, opt, args, sched=None):
  if args.epochs == 0: return

  assert(args.l1_loss or not args.no_l2_loss), "Must have either l1 or l2 loss"
  if args.l1_loss and not args.no_l2_loss:
    loss_fn = lambda x, ref: (F.mse_loss(x, ref) + F.l1_loss(x, ref))/2
  elif args.l1_loss and args.no_l2_loss: loss_fn = F.l1_loss
  else: loss_fn = F.mse_loss

  iters = range(args.epochs) if args.quiet else trange(args.epochs)
  update = lambda kwargs: iters.set_postfix(**kwargs)
  if args.quiet: update = lambda _: None
  batch_size = min(args.batch_size, len(cam))
  times=None
  if args.data_kind == "dnerf":
    times = labels[-1]
    labels = labels[0]

  get_crop = lambda: (0,0, args.render_size, args.render_size)
  cs = args.crop_size
  if args.crop:
    get_crop = lambda: (
      random.randint(0, args.render_size-cs), random.randint(0, args.render_size-cs), cs, cs,
    )

  next_idxs = lambda _: random.sample(range(len(cam)), batch_size)
  if args.serial_idxs: next_idxs = lambda i: [i%len(cam)] * batch_size
  losses = []
  for i in iters:
    opt.zero_grad()

    idxs = next_idxs(i)

    ts = None if times is None else times[idxs]
    c0,c1,c2,c3 = crop = get_crop()
    ref = labels[idxs][:, c0:c0+c2,c1:c1+c3, :]

    # omit items which are all darker with some likelihood.
    if args.omit_bg and (i % args.save_freq) != 0 and (i % args.valid_freq) != 0 and \
      ref.mean() + 0.3 < sqr(random.random()): continue

    out = render(model, cam[idxs], crop, size=args.render_size, times=ts, args=args)
    if args.blur:
      r = 1 + 2 * random.randint(0,2)
      out = TVF.gaussian_blur(out.permute(0,3,1,2), r).permute(0,2,3,1)
      ref = TVF.gaussian_blur(ref.permute(0,3,1,2), r).permute(0,2,3,1) # TODO cache ref blur?
    if args.sharpen:
      out = TVF.sharpen(out.permute(0,3,1,2), 1.5).permute(0,2,3,1)
      ref = TVF.sharpen(ref.permute(0,3,1,2), 1.5).permute(0,2,3,1) # TODO cache ref sharpen?
    loss = loss_fn(out, ref)
    l2_loss = loss.item()
    display = {
      "l2": f"{l2_loss:.04f}",
      "refresh": False,
    }
    if sched is not None: display["lr"] = f"{sched.get_last_lr()[0]:.1e}"
    if args.sdf:
      eik = sdf.eikonal_loss(model.sdf.normals)
      s = sdf.sigmoid_loss(model.min_along_rays, model.nerf.acc())
      loss = loss + eik + s
      #display["eik"] = f"{eik:.02f}"
      #display["s"] = f"{s:.02f}"
    if args.nerf_eikonal:
      loss = loss + 1e-3 * model.nerf.eikonal_loss
      display["n-eik"] = f"{model.nerf.eikonal_loss:.02f}"
    # experiment with emptying the model at the beginning
    if i < args.n_sparsify_alpha: loss = loss + (model.nerf.alpha - 0.5).square().mean()

    update(display)
    losses.append(l2_loss)

    assert(loss.isfinite().item()), "Got NaN loss"
    loss.backward()
    opt.step()
    if sched is not None: sched.step()
    if i % args.valid_freq == 0:
      with torch.no_grad():
        ref0 = ref[0]
        acc = model.nerf.acc()[0,...,None].expand_as(ref0)
        save_plot(f"outputs/valid_{i:05}.png", ref0, out[0].clamp(min=0, max=1), acc)
    if i % args.save_freq == 0 and i != 0: save(model, args)
  window = min(500, len(losses))
  losses = np.convolve(losses, np.ones(window)/window, mode='valid')
  plt.plot(range(len(losses)), losses)
  plt.savefig("outputs/training_loss.png", bbox_inches='tight')
  plt.close()

def test(model, cam, labels, args, training: bool = True):
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
          acc[c0:c0+args.crop_size, c1:c1+args.crop_size, :] = model.nerf.acc()[..., None]

      loss = F.mse_loss(got, exp)
      print(f"[{i:03}]: L2 {loss.item():.03f} PSNR {utils.mse2psnr(loss).item():.03f}")
      name = f"outputs/train_{i:03}.png" if training else f"outputs/test_{i:03}.png"
      save_plot(name, exp, got.clamp(min=0, max=1), acc)

def load_mip(args):
  if args.mip is None: return None
  elif args.mip == "cone": return ConicGaussian()
  elif args.mip == "cylinder": return CylinderGaussian()

  raise NotImplementedError(f"Unknown mip kind {args.mip}")

def load_model(args):
  if not args.neural_upsample: args.feature_space = 3
  if args.data_kind == "dnerf" and args.dnerfae: args.model = "ae"
  kwargs = {
    "mip": load_mip(args),
    "out_features": args.feature_space,
    "device": device,
    "steps": args.steps,
    "t_near": args.near,
    "t_far": args.far,
    "per_pixel_latent_size": 64 if args.data_kind == "pixel-single" else 0,
    "per_point_latent_size": (1 + 3 + 64) if args.sdf else 0,
    "use_fat_sigmoid": args.fat_sigmoid,
    "eikonal_loss": args.nerf_eikonal,
  }
  if args.model == "tiny": constructor = nerf.TinyNeRF
  elif args.model == "plain": constructor = nerf.PlainNeRF
  elif args.model == "ae": constructor = nerf.NeRFAE
  else: raise NotImplementedError(args.model)
  model = constructor(**kwargs).to(device)

  # Add in a dynamic model if using dnerf with the underlying model.
  if args.data_kind == "dnerf":
    constructor = nerf.DynamicNeRFAE if arg.dnerfae else nerf.DynamicNeRF
    model = constructor(canonical=model, device=device).to(device)

  if args.data_kind == "pixel-single":
    encoder = SpatialEncoder().to(device)
    # args.img is populated in load (single_image)
    model = nerf.SinglePixelNeRF(model, encoder=encoder, img=args.img, device=device).to(device)

  og_model = model
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
    setattr(model, "nerf", og_model)

  if args.data_parallel:
    model = nn.DataParallel(model)
    setattr(model, "nerf", og_model)
  return model

def save(model, args):
  print(f"Saved to {args.save}")
  torch.save(model, args.save)
  if args.log is not None:
    setattr(args, "curr_time", datetime.today().strftime('%Y-%m-%d-%H:%M:%S'))
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

  parameters = model.parameters()
  if args.train_camera: parameters = chain(parameters, cam.parameters())

  # for some reason AdamW doesn't seem to work here
  # eps = 1e-7 was in the original paper.
  opt = optim.Adam(parameters, lr=args.learning_rate, weight_decay=args.decay, eps=1e-7)

  # TODO should T_max = -1 or args.epochs
  sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=5e-5)
  if args.no_sched: sched = None
  train(model, cam, labels, opt, args, sched=sched)

  if args.epochs != 0: save(model, args)
  if args.notest: return

  test(model, cam, labels, args, training=True)

  test_labels, test_cam = load(args, training=False)
  test(model, test_cam, test_labels, args, training=False)

if __name__ == "__main__": main()

