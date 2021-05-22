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

import src.loaders
import src.nerf as nerf
from src.utils import ( save_image, save_plot, CylinderGaussian )
from src.neural_blocks import Upsampler

def arguments():
  a = argparse.ArgumentParser()
  a.add_argument("-d", "--data", help="path to data", required=True)
  a.add_argument(
    "--data-kind", help="kind of data to load",
    choices=["original", "single_video", "dnerf"],
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
  # TODO type of crop strategy? Maybe can do random resizing?
  a.add_argument(
    "--mip", help="Use MipNeRF with different sampling", type=str,
    choices=["cone", "cylinder"],
  )

  a. add_argument(
    "--feature-space",
    help="when using neural upsampling, what is the feature space size",
    type=int, default=32,
  )
  a.add_argument(
    "--model", help="which model do we want to use", type=str,
    choices=["tiny", "plain", "ae"], default="plain",
  )
  a.add_argument(
    "-lr", "--learning-rate", help="learning rate", type=float, default=5e-3,
  )
  a.add_argument(
    "--valid-freq", help="how often validation images are generated", type=int, default=500,
  )
  a.add_argument("--seed", help="random seed to use", type=int, default=1337)
  a.add_argument("--decay", help="weight_decay value", type=float, default=1e-5)
  a.add_argument("--notest", help="run test set", action="store_true")
  a.add_argument("--data-parallel", help="Use data parallel for the model", action="store_true")
  a.add_argument("--omit-bg", help="Omit bg with some probability", action="store_true")

  a.add_argument("--save", help="Where to save the model", type=str, default="models/model.pt")
  a.add_argument("--log", help="Where to save log of arguments", type=str, default="log.json")

  cam = a.add_argument_group("camera parameters")
  cam.add_argument("--near", help="near plane for camera", default=2)
  cam.add_argument("--far", help="far plane for camera", default=6)
  # TODO add more arguments here
  return a.parse_args()

# TODO better automatic device discovery here

device = "cpu"
if torch.cuda.is_available():
  device = torch.device("cuda:0")
  torch.cuda.set_device(device)

def render(
  model,
  cam,
  crop,
  # how big should the image be
  size,

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
  return model(rays)

# loads the dataset
def load(args, training=True):
  assert(args.data is not None)
  kind = args.data_kind
  if args.derive_kind:
    if args.data.endswith(".mp4"): kind = "single_video"

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
  else:
    raise NotImplementedError(kind)

def sqr(x): return x * x

# train the model with a given camera and some labels (imgs or imgs+times)
def train(model, cam, labels, opt, args, sched=None):
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

    # omit items which are all black with some likelihood
    if args.omit_bg and ref.mean() + 5e-3 < sqr(random.random()): continue

    out = render(
      model, cam[idxs], crop, size=args.render_size, times=ts,
    )
    loss = F.mse_loss(out, ref)
    loss.backward()
    loss = loss.item()
    opt.step()
    t.set_postfix(loss=f"{loss:03f}", lr=f"{sched.get_last_lr()[0]:.1e}", refresh=False)
    if sched is not None: sched.step()
    if (i % args.valid_freq == 0):
      # TODO render whole thing if crop otherwise use output
      save_plot(f"outputs/valid_{i:05}.png", ref[0], out[0])

def test(model, cam, labels, args):
  times = None
  if args.data_kind == "dnerf":
    times = labels[-1]
    labels = labels[0]

  crop = (0,0, args.render_size, args.render_size)
  with torch.no_grad():
    for i in range(labels.shape[0]):
      ts = None if times is None else times[i:i+1]
      out = render(
        model, cam[i:i+1], crop, size=args.render_size, with_noise=False, times=ts
      ).squeeze(0)
      loss = F.mse_loss(out, labels[i])
      print(loss.item())
      save_plot(f"outputs/out_{i:03}.png", labels[i:i+1], out)

def load_mip(args):
  if args.mip is None:
    return None
  elif args.mip == "cone":
    raise NotImplementedError()
  elif args.mip == "cylinder":
    return CylinderGaussian()
  else:
    raise NotImplementedError(f"Unknown mip kind {args.mip}")

def load_model(args):
  if not args.neural_upsample: args.feature_space = 3
  mip = load_mip(args)
  kwargs = {
    "mip": mip,
    "out_features": args.feature_space,
    "device": device,
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
  else:
    model = nn.Sequential(model, nn.Sigmoid())

  if args.data_parallel: model = nn.DataParallel(model)
  return model

# in theory this is an interesting loss function but it runs into numerical stability issues
# probably need to converd the prod().log() into a sum of logs.
#def all_mse_loss(ref, img, eps=1e-2):
#  return ((ref - img).square() + eps).reciprocal().prod().log().neg()

def save(model, args):
  torch.save(model, args.save)
  if args.log is not None:
    with open(args.log, 'w') as f:
      json.dump(args.__dict__, f, indent=2)
  ...

def seed(s):
  torch.manual_seed(s)
  random.seed(s)
  np.random.seed(s)

def main():
  args = arguments()
  seed(args.seed)

  model = load_model(args) if args.load is None else torch.load(args.load)

  # for some reason AdamW doesn't seem to work here
  opt = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.decay)

  labels, cam = load(args)

  sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=1e-8)
  train(model, cam, labels, opt, args, sched=sched)

  if args.epochs != 0: save(model, args.save)

  if args.notest: return
  model.eval()

  test_labels, test_cam = load(args, training=False)
  test(model, test_cam, test_labels, args)

if __name__ == "__main__": main()

