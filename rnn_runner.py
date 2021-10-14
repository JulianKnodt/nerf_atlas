# Global runner for all NeRF methods.
# For convenience, we want all methods using NeRF to use this one file.
import argparse
import random
import json
import math
import time

import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms.functional as TVF
import torch.nn as nn
import matplotlib.pyplot as plt

from datetime import datetime
from tqdm import trange
from itertools import chain

import src.loaders as loaders
import src.nerf as nerf
import src.utils as utils
import src.sdf as sdf
import src.refl as refl
import src.cameras as cameras
import src.hyper_config as hyper_config
import src.renderers as renderers
from src.lights import light_kinds
from src.utils import ( save_image, save_plot, load_image, dir_to_elev_azim )

import os

def arguments():
  a = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  a.add_argument("-d", "--data", help="path to data", required=True)
  a.add_argument(
    "--data-kind", help="Kind of data to load", default="original",
    choices=[
      "original", "single_video", "dnerf", "dtu", "pixel-single", "nerv_point", "shiny",
    ],
  )

  a.add_argument("--outdir", help="path to output directory", type=str, default="outputs/")
  a.add_argument("--size", help="pre-upsampling size", type=int, default=16)

  a.add_argument("--epochs", help="number of epochs to train for", type=int, default=30000)
  a.add_argument("--batch-size", help="# views for each training batch", type=int, default=8)
  a.add_argument("--crop", help="train with cropping", action="store_true")
  a.add_argument("--crop-size",help="what size to use while cropping",type=int, default=16)
  a.add_argument("--steps", help="Number of depth steps", type=int, default=64)
  a.add_argument(
    "--sigmoid-kind", help="What sigmoid to use, curr keeps old", default="thin",
    choices=list(utils.sigmoid_kinds.keys()),
  )
  a.add_argument(
    "--bg", help="What kind of background to use for NeRF", type=str,
    choices=["black", "white", "mlp", "noise"], default="black",
  )
  # this default for LR seems to work pretty well?
  a.add_argument("-lr", "--learning-rate", help="learning rate", type=float, default=5e-4)
  a.add_argument("--seed", help="Random seed to use, -1 is no seed", type=int, default=1337)
  a.add_argument("--decay", help="Weight decay value", type=float, default=0)
  a.add_argument("--notest", help="Do not run test set", action="store_true")
  a.add_argument(
    "--loss-fns", help="Loss functions to use", nargs="+", type=str,
    # TODO add SSIM here? Or LPIPS?
    choices=["l1", "l2", "rmse"], default=["l2"],
  )
  a.add_argument(
    "--color-spaces", help="Color spaces to compare on", nargs="+", type=str,
    choices=["rgb", "hsv", "luminance", "xyz"], default=["rgb"],
  )
  a.add_argument("--no-sched", help="Do not use a scheduler", action="store_true")

  refla = a.add_argument_group("reflectance")
  refla.add_argument(
    "--refl-kind", help="What kind of reflectance model to use",
    choices=refl.refl_kinds,
  )
  refla.add_argument(
    "--space-kind", choices=["identity", "surface", "none"], default="identity",
    help="Space to encode texture: surface builds a map from 3D (identity) to 2D",
  )

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
  rprt.add_argument("--nosave", help="do not save", action="store_true")
  rprt.add_argument("--load", help="model to load from", type=str)
  rprt.add_argument("--loss-window", help="# epochs to smooth loss over", type=int, default=250)
  rprt.add_argument("--notraintest", help="Do not test on training set", action="store_true")
  rprt.add_argument(
    "--duration-sec", help="Max number of seconds to run this for, s <= 0 implies None",
    type=float, default=0,
  )
  rprt.add_argument(
    "--skip-loss", type=int, default=0, help="Number of epochs to skip reporting loss for",
  )
  rprt.add_argument(
    "--msssim-loss", action="store_true", help="Report ms-ssim loss during testing",
  )
  rprt.add_argument(
    "--not-magma", action="store_true", help="Do not use magma for depth maps (instead use default)",
  )

  args = a.parse_args()

  # runtime checks
  if not os.path.exists(args.outdir): os.mkdir(args.outdir)

  if not args.not_magma: plt.magma()
  setattr(args, "derive_kind", False)
  setattr(args, "model", "rnn_nerf")
  setattr(args, "volsdf_alternate", False)

  return args

loss_map = {
  "l2": F.mse_loss,
  "l1": F.l1_loss,
  "rmse": lambda x, ref: F.mse_loss(x, ref).clamp(min=1e-10).sqrt(),
}

color_fns = {
  "hsv": utils.rgb2hsv,
  "luminance": utils.rgb2luminance,
  "xyz": utils.rgb2xyz,
}

device = "cpu"
if torch.cuda.is_available():
  device = torch.device("cuda:0")
  torch.cuda.set_device(device)

# DEBUG
#torch.autograd.set_detect_anomaly(True); print("HAS DEBUG")

def render(
  model, cam, crop,
  # how big should the image be
  size,
  args, with_noise=0.1,
):
  ii, jj = torch.meshgrid(
    torch.arange(size, device=device, dtype=torch.float),
    torch.arange(size, device=device, dtype=torch.float),
  )
  positions = torch.stack([ii.transpose(-1, -2), jj.transpose(-1, -2)], dim=-1)
  t,l,h,w = crop
  positions = positions[t:t+h,l:l+w,:]
  rays = cam.sample_positions(positions, size=size, with_noise=with_noise)
  return model(rays)

def sqr(x): return x * x
def save_losses(args, losses):
  outdir = args.outdir
  window = args.loss_window

  window = min(window, len(losses))
  losses = np.convolve(losses, np.ones(window)/window, mode='valid')
  losses = losses[args.skip_loss:]
  plt.plot(range(len(losses)), losses)
  plt.savefig(os.path.join(outdir, "training_loss.png"), bbox_inches='tight')
  plt.close()

def load_loss_fn(args, model):
  # different losses like l1 or l2
  loss_fns = [loss_map[lfn] for lfn in args.loss_fns]
  assert(len(loss_fns) > 0), "must provide at least 1 loss function"
  if len(loss_fns) == 1: loss_fn = loss_fns[0]
  else:
    def loss_fn(x, ref):
      loss = 0
      for fn in loss_fns: loss = loss + fn(x, ref)
      return loss

  assert(len(args.color_spaces) > 0), "must provide at least 1 color space"
  # different colors like rgb, hsv
  if len(args.color_spaces) == 1 and args.color_spaces[0] == "rgb":
    # do nothing since this is the default return value
    ...
  elif len(args.color_spaces) == 1:
    cfn = color_fns[args.color_spaces[0]]
    prev_loss_fn = loss_fn
    loss_fn = lambda x, ref: prev_loss_fn(cfn(x), cfn(ref))
  elif "rgb" in args.color_spaces:
    prev_loss_fn = loss_fn
    cfns = [color_fns[cs] for cs in args.color_spaces if cs != "rgb"]
    def loss_fn(x, ref):
      loss = prev_loss_fn(x, ref)
      for cfn in cfns: loss = loss + prev_loss_fn(cfn(x), cfn(ref))
      return loss
  else:
    prev_loss_fn = loss_fn
    cfns = [color_fns[cs] for cs in args.color_spaces]
    def loss_fn(x, ref):
      loss = 0
      for cfn in cfns: loss = loss + prev_loss_fn(cfn(x), cfn(ref))
      return loss

  return loss_fn

def train(model, cam, labels, opt, args, light=None, sched=None):
  if args.epochs == 0: return

  loss_fn = load_loss_fn(args, model)

  iters = range(args.epochs) if args.quiet else trange(args.epochs)
  update = lambda kwargs: iters.set_postfix(**kwargs)
  if args.quiet: update = lambda _: None
  batch_size = min(args.batch_size, len(cam))

  get_crop = lambda: (0,0, args.size, args.size)
  cs = args.crop_size
  if args.crop:
    get_crop = lambda: (
      random.randint(0, args.size-cs), random.randint(0, args.size-cs), cs, cs,
    )

  next_idxs = lambda _: random.sample(range(len(cam)), batch_size)

  losses = []
  start = time.time()
  should_end = lambda: False
  if args.duration_sec > 0: should_end = lambda: time.time() - start > args.duration_sec

  for i in iters:
    if should_end():
      print("Training timed out")
      break

    opt.zero_grad()

    idxs = next_idxs(i)

    c0,c1,c2,c3 = crop = get_crop()
    ref = labels[idxs][:, c0:c0+c2,c1:c1+c3, :]

    out = render(model, cam[idxs], crop, size=args.size, args=args)
    loss = 0
    for n, img in enumerate(out[3:]):
      loss = loss + (n+1) * loss_fn(img, ref)
    out = out[-1]
    assert(loss.isfinite()), f"Got {loss.item()} loss"
    l2_loss = loss.item()
    display = {
      "l2": f"{l2_loss:.04f}",
      "refresh": False,
    }
    if sched is not None: display["lr"] = f"{sched.get_last_lr()[0]:.1e}"

    update(display)
    losses.append(l2_loss)

    assert(loss.isfinite().item()), "Got NaN loss"
    loss.backward()
    opt.step()
    if sched is not None: sched.step()

    # Save outputs within the cropped region.
    if i % args.valid_freq == 0:
      with torch.no_grad():
        ref0 = ref[0,...,:3]
        items = [ref0, out[0,...,:3].clamp(min=0, max=1)]
        save_plot(os.path.join(args.outdir, f"valid_{i:05}.png"), *items)

    if i % args.save_freq == 0 and i != 0:
      save(model, args)
      save_losses(args, losses)
  save(model, args)
  save_losses(args, losses)

def test(model, cam, labels, args, training: bool = True, light=None):
  model = model.eval()

  ls = []
  gots = []

  with torch.no_grad():
    for i in range(labels.shape[0]):
      exp = labels[i,...,:3]
      got = torch.zeros_like(exp)
      gotn1 = torch.zeros_like(exp)
      gotn2 = torch.zeros_like(exp)
      depth = torch.zeros(*got.shape[:-1], 1, device=got.device, dtype=torch.float)

      N = math.ceil(args.size/args.crop_size)
      for x in range(N):
        for y in range(N):
          c0 = x * args.crop_size
          c1 = y * args.crop_size
          out = render(
            model, cam[i:i+1, ...], (c0,c1,args.crop_size,args.crop_size), size=args.size,
            with_noise=False, args=args,
          )
          got[c0:c0+args.crop_size, c1:c1+args.crop_size, :] = out[-1].squeeze(0)
          gotn1[c0:c0+args.crop_size, c1:c1+args.crop_size, :] = out[-2].squeeze(0)
          gotn2[c0:c0+args.crop_size, c1:c1+args.crop_size, :] = out[-3].squeeze(0)

      gots.append(got)
      loss = F.mse_loss(got, exp)
      psnr = utils.mse2psnr(loss).item()
      print(f"[{i:03}]: L2 {loss.item():.03f} PSNR {psnr:.03f}")
      name = f"train_{i:03}.png" if training else f"test_{i:03}.png"
      items = [
        exp,
        got.clamp(min=0, max=1),
        gotn1.clamp(min=0, max=1),
        gotn2.clamp(min=0, max=1),
      ]
      save_plot(os.path.join(args.outdir, name), *items)
      #save_image(os.path.join(args.outdir, f"got_{i:03}.png"), got)
      ls.append(psnr)

  print(f"""[Summary ({"training" if training else "test"})]:
\tmean {np.mean(ls):.03f}
\tmin {min(ls):.03f}
\tmax {max(ls):.03f}
\tvar {np.var(ls):.03f}""")
  if args.msssim_loss:
    with torch.no_grad():
      msssim = utils.msssim_loss(gots, labels)
      print(f"\tms-ssim {msssim:.03f}")

def load_model(args):
  return nerf.RecurrentNeRF().to(device)

def save(model, args):
  if args.nosave: return
  print(f"Saved to {args.save}")
  torch.save(model, args.save)
  if args.log is not None:
    setattr(args, "curr_time", datetime.today().strftime('%Y-%m-%d-%H:%M:%S'))
    with open(args.log, 'w') as f:
      json.dump(args.__dict__, f, indent=2)

def seed(s):
  if s == -1: return
  torch.manual_seed(s)
  random.seed(s)
  np.random.seed(s)

def main():
  args = arguments()
  seed(args.seed)

  labels, cam, light = loaders.load(args, training=True, device=device)

  model = load_model(args) if args.load is None else torch.load(args.load, map_location=device)

  parameters = model.parameters()

  opt = optim.Adam(parameters, lr=args.learning_rate, weight_decay=args.decay, eps=1e-7)

  # TODO should T_max = -1 or args.epochs
  sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=5e-5)
  if args.no_sched: sched = None
  train(model, cam, labels, opt, args, light=light, sched=sched)


  if not args.notraintest: test(model, cam, labels, args, training=True, light=light)

  if args.notest: return
  test_labels, test_cam, test_light = loaders.load(args, training=False, device=device)
  test(model, test_cam, test_labels, args, training=False, light=test_light)

if __name__ == "__main__": main()

