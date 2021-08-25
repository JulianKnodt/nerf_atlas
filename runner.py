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
from src.utils import ( save_image, save_plot, load_image )
from src.neural_blocks import ( Upsampler, SpatialEncoder, StyleTransfer )

import os

def arguments():
  a = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  a.add_argument("-d", "--data", help="path to data", required=True)
  a.add_argument(
    "--data-kind", help="Kind of data to load", default="original",
    choices=[
      "original", "single_video", "dnerf", "dtu", "pixel-single", "nerv_point",
      "shiny"
    ],
  )
  a.add_argument(
    "--derive-kind", help="Attempt to derive the kind if a single file is given",
    action="store_false",
  )

  a.add_argument("--outdir", help="path to output directory", type=str, default="outputs/")
  a.add_argument(
    "--timed-outdir", help="Create new output directory with date and time of run", action="store_true"
  )

  # various size arguments
  a.add_argument("--size", help="post-upsampling size", type=int, default=32)
  a.add_argument("--render-size", help="pre-upsampling size", type=int, default=16)

  a.add_argument("--epochs", help="number of epochs to train for", type=int, default=30000)
  a.add_argument("--batch-size", help="# views for each training batch", type=int, default=8)
  a.add_argument("--neural-upsample", help="add neural upsampling", action="store_true")
  a.add_argument("--crop", help="train with cropping", action="store_true")
  a.add_argument("--crop-size",help="what size to use while cropping",type=int, default=16)
  a.add_argument("--steps", help="Number of depth steps", type=int, default=64)
  a.add_argument(
    "--mip", help="Use MipNeRF with different sampling", type=str, choices=["cone", "cylinder"],
  )
  a.add_argument(
    "--sigmoid-kind", help="What sigmoid to use, curr keeps old", default="thin",
    choices=["normal", "fat", "thin", "cyclic", "softmax", "curr"],
  )
  a.add_argument("--sparsify-alpha", help="Weight for sparsifying alpha",type=float,default=0)
  a.add_argument("--backing-sdf", help="Use a backing SDF", action="store_true")

  a. add_argument(
    "--feature-space", help="when using neural upsampling, what is the feature space size",
    type=int, default=32,
  )
  a.add_argument(
    "--model", help="which model do we want to use", type=str,
    choices=["tiny", "plain", "ae", "volsdf", "sdf"], default="plain",
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
  a.add_argument("--data-parallel", help="Use data parallel for the model", action="store_true")
  a.add_argument("--omit-bg", help="Omit black bg with some probability", action="store_true")
  a.add_argument(
    "--train-parts", help="Which parts of the model should be trained",
    choices=["all", "refl", "[TODO]Camera"], default="all",
  )
  a.add_argument(
    "--loss-fns", help="Loss functions to use", nargs="+", type=str,
    # TODO add SSIM here? Or LPIPS?
    choices=["l1", "l2", "rmse"], default=["l2"],
  )
  a.add_argument(
    "--color-spaces", help="Color spaces to compare on", nargs="+", type=str,
    choices=["rgb", "hsv", "luminance", "xyz"], default=["rgb"],
  )
  a.add_argument(
    "--tone-map", help="Add tone mapping (1/(1+x)) before loss function", action="store_true",
  )
  a.add_argument(
    "--nerv-multi-point", help="Use NeRV multi point light dataset for testing",
    action="store_true",
  )
  a.add_argument("--style-img", help="Image to use for style transfer", default=None)
  a.add_argument("--no-sched", help="Do not use a scheduler", action="store_true")
  a.add_argument("--serial-idxs", help="Train on images in serial", action="store_true")
  # TODO really fix MPIs
  a.add_argument("--mpi", help="Use multi-plain imaging", action="store_true")
  a.add_argument(
    "--replace", nargs="*", choices=["refl", "occ", "bg"], default=[], type=str,
    help="Modules to replace on this run, if any. Take caution for overwriting existing parts.",
  )

  a.add_argument(
    "--volsdf-alternate", help="Use alternating volume rendering/SDF training volsdf",
    action="store_true",
  )
  a.add_argument(
    "--latent-size",type=int, default=32,
    help="Latent-size to use in shape models. If not supported by the shape model, it will be ignored.",
  )
  a.add_argument(
    "--spherical-harmonic-order", default=2, type=int,
    help="Learn spherical harmonic coefficients up to given order. Used w/ --refl-kind=sph-har",
  )

  refla = a.add_argument_group("reflectance")
  refla.add_argument(
    "--refl-kind", help="What kind of reflectance model to use",
    choices=["curr", "view_only", "basic", "diffuse", "rusin", "multi_rusin", "sph-har"], default="curr",
  )
  refla.add_argument(
    "--normal-kind", choices=[None, "elaz", "raw"], default=None,
    help="How to include normals in reflectance model. Not all surface models support normals",
  )
  refla.add_argument(
    "--space-kind", choices=["identity", "surface"], default="identity",
    help="Space to encode texture: surface builds a map from 3D (identity) to 2D",
  )

  rdra = a.add_argument_group("integrator")
  rdra.add_argument(
    "--integrator-kind", choices=[None, "direct", "path"], default=None,
    help="Integrator to use for surface rendering",
  )
  rdra.add_argument(
    "--occ-kind", choices=[None, "hard", "learned", "all-learned"], default=None,
    help="Occlusion method for shadows to use in integration",
  )

  lighta = a.add_argument_group("light")
  lighta.add_argument(
    "--light-kind",
    help="Kind of light to use while rendering. Dataset indicates light is in dataset",
    choices=[None, "point", "field", "dataset"], default=None,
  )

  sdfa = a.add_argument_group("sdf")
  sdfa.add_argument(
    "--sdf-eikonal", help="Weight of SDF eikonal loss", type=float, default=0,
  )
  sdfa.add_argument(
    "--smooth-normals", help="Amount to attempt to smooth normals", type=float, default=0,
  )
  sdfa.add_argument(
    "--smooth-eps", help="size of random uniform perturbation for smooth normals regularization", 
    type=float, default=0,
  )
  sdfa.add_argument(
    "--sdf-kind", help="Which SDF model to use", type=str,
    choices=["spheres", "siren", "local", "mlp", "triangles"], default="mlp",
  )
  sdfa.add_argument("--sphere-init", help="Initialize SDF to a sphere", action="store_true")
  sdfa.add_argument(
    "--bound-sphere-rad", type=float, default=-1,
    help="Intersect the learned SDF with a bounding sphere at the origin, < 0 is no sphere",
  )
  sdfa.add_argument(
    "--sdf-isect-kind", choices=["sphere", "secant", "bisect"], default="sphere",
    help="Marching kind to use when computing SDF intersection.",
  )

  dnerfa = a.add_argument_group("dnerf")
  dnerfa.add_argument("--dnerfae", help="Use DNeRFAE on top of DNeRF", action="store_true")
  dnerfa.add_argument(
    "--dnerf-tf-smooth-weight", help="L2 smooth dnerf tf", type=float, default=0,
  )
  dnerfa.add_argument("--time-gamma", help="Apply a gamma based on time", action="store_true")
  dnerfa.add_argument("--gru-flow", help="Use GRU for Δx", action="store_true")
  dnerfa.add_argument("--nicepath", help="Render a nice path for DNeRF", action="store_true")
  dnerfa.add_argument("--with-canon", help="Preload a canonical NeRF", type=str, default=None)
  dnerfa.add_argument("--fix-canon", help="Do not train canonical NeRF", action="store_true")

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
  rprt.add_argument("--sphere_visualize", help="Radius to use for spherical visualization", default=None, type=int)
  rprt.add_argument(
    "--duration-sec", help="Max number of seconds to run this for, s <= 0 implies None",
    type=float, default=0,
  )
  rprt.add_argument(
    "--param-file", type=str, default=None, help="Path to JSON file to use for hyper-parameters",
  )
  rprt.add_argument(
    "--skip-loss", type=int, default=0, help="Number of epochs to skip reporting loss for",
  )
  rprt.add_argument(
    "--msssim-loss", action="store_true", help="Report ms-ssim loss during testing",
  )

  meta = a.add_argument_group("meta runner parameters")
  meta.add_argument("--torchjit", help="Use torch jit for model", action="store_true")
  meta.add_argument("--train-imgs", help="# training examples", type=int, default=-1)

  ae = a.add_argument_group("auto encoder parameters")
  ae.add_argument("--latent-l2-weight", help="L2 regularize latent codes", type=float, default=0)
  ae.add_argument("--normalize-latent", help="L2 normalize latent space", action="store_true")
  ae.add_argument("--encoding-size",help="Intermediate encoding size for AE",type=int,default=32)

  args = a.parse_args()

  # runtime checks
  hyper_config.load(args)
  if args.timed_outdir:
    args.outdir = os.path.join(args.outdir, datetime.today().strftime('%Y-%m-%d-%H:%M:%S'))
  if not os.path.exists(args.outdir): os.mkdir(args.outdir)

  if not args.neural_upsample:
    args.render_size = args.size
    args.feature_space = 3

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

# TODO better automatic device discovery here
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
  if args.style_img != None:
    return StyleTransfer(load_image(args.style_img, resize=(args.size, args.size)))

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


  if args.tone_map: loss_fn = utils.tone_map(loss_fn)
  if args.volsdf_alternate:
    return nerf.alternating_volsdf_loss(model, loss_fn, sdf.masked_loss(loss_fn))
  if args.model == "sdf": loss_fn = sdf.masked_loss(loss_fn)
  return loss_fn

# train the model with a given camera and some labels (imgs or imgs+times)
# light is a per instance light.
def train(model, cam, labels, opt, args, light=None, sched=None):
  if args.epochs == 0: return

  loss_fn = load_loss_fn(args, model)

  iters = range(args.epochs) if args.quiet else trange(args.epochs)
  update = lambda kwargs: iters.set_postfix(**kwargs)
  if args.quiet: update = lambda _: None
  batch_size = min(args.batch_size, len(cam))
  times=None
  if args.data_kind == "dnerf":
    times = labels[-1]
    labels = labels[0]

  get_crop = lambda: (0,0, args.size, args.size)
  cs = args.crop_size
  if args.crop:
    get_crop = lambda: (
      random.randint(0, args.render_size-cs), random.randint(0, args.render_size-cs), cs, cs,
    )

  next_idxs = lambda _: random.sample(range(len(cam)), batch_size)
  if args.serial_idxs: next_idxs = lambda i: [i%len(cam)] * batch_size
  #next_idxs = lambda i: [i%10] * batch_size # DEBUG

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

    ts = None if times is None else times[idxs]
    c0,c1,c2,c3 = crop = get_crop()
    ref = labels[idxs][:, c0:c0+c2,c1:c1+c3, :]

    if light is not None: model.refl.light = light[idxs]

    # omit items which are all darker with some likelihood. This is mainly used when
    # attempting to focus on learning the refl and not the shape.
    if args.omit_bg and (i % args.save_freq) != 0 and (i % args.valid_freq) != 0 and \
      ref.mean() + 0.3 < sqr(random.random()): continue

    out = render(model, cam[idxs], crop, size=args.render_size, times=ts, args=args)
    loss = loss_fn(out, ref)
    assert(loss.isfinite()), f"Got {loss.item()} loss"
    l2_loss = loss.item()
    display = {
      "l2": f"{l2_loss:.04f}",
      "refresh": False,
    }
    if sched is not None: display["lr"] = f"{sched.get_last_lr()[0]:.1e}"

    if args.backing_sdf:
      eik = sdf.eikonal_loss(model.sdf.normals)
      s = sdf.sigmoid_loss(model.min_along_rays, model.nerf.acc())
      loss = loss + eik + s

    if args.latent_l2_weight > 0:
      loss = loss + model.nerf.latent_l2_loss * latent_l2_weight

    # experiment with emptying the model at the beginning
    if args.sparsify_alpha > 0: loss = loss + args.sparsify_alpha * (model.nerf.alpha).square().mean()
    if args.dnerf_tf_smooth_weight > 0:
      loss = loss + args.dnerf_tf_smooth_weight * model.delta_smoothness

    # prepare one set of points for either smoothing normals or eikonal.
    if args.sdf_eikonal > 0 or args.smooth_normals > 0:
      pts = 8*(torch.rand(1<<13, 3, device=device, requires_grad=True)-0.5)
      n = model.sdf.normals(pts)

    # E[d sdf(x)/dx] = 1, enforces that the SDF is valid.
    if args.sdf_eikonal > 0: loss = loss + args.sdf_eikonal * utils.eikonal_loss(n)

    # dn/dx -> 0, hopefully smoothes out the local normals of the surface.
    # TODO does it matter to normalize the normal or not?
    if args.smooth_normals > 0:
      if args.smooth_eps > 0:
        # epsilon-perturbation implementation from unisurf
        perturbation = F.normalize(2*torch.rand(3, device=device)-1, dim=-1) * args.smooth_eps
        perturbation = perturbation.expand(pts.shape)
        delta_n = n - model.sdf.normals(pts + perturbation)
      else:
        # TODO maybe lower dimensionality of n?
        delta_n = torch.autograd.grad(
          inputs=pts, outputs=F.normalize(n,dim=-1), create_graph=True,
          grad_outputs=torch.ones_like(n),
        )[0]
      # TODO maybe convert this to abs? seems to work for nerfactor, altho unisurf uses square?
      loss = loss + args.smooth_normals * torch.linalg.norm(delta_n, dim=-1).square().mean()

    update(display)
    losses.append(l2_loss)

    assert(loss.isfinite().item()), "Got NaN loss"
    loss.backward()
    opt.step()
    if sched is not None: sched.step()

    if i % args.valid_freq == 0:
      with torch.no_grad():
        ref0 = ref[0,...,:3]
        items = [ref0, out[0,...,:3].clamp(min=0, max=1)]
        if hasattr(model, "nerf") and args.model != "volsdf":
          items.append(model.nerf.acc()[0,...,None].expand_as(ref0).clamp(min=0, max=1))
          items.append(model.nerf.acc_smooth()[0,...].expand_as(ref0).clamp(min=0, max=1))
        elif out.shape[-1] == 4:
          items.append(ref[0,...,-1,None].expand_as(ref0))
          items.append(out[0,...,-1,None].expand_as(ref0).sigmoid())
        save_plot(os.path.join(args.outdir, f"valid_{i:05}.png"), *items)
    if i % args.save_freq == 0 and i != 0:
      save(model, args)
      save_losses(args, losses)
  save(model, args)
  save_losses(args, losses)

def test(model, cam, labels, args, training: bool = True, light=None):
  times = None
  model = model.eval()
  if args.data_kind == "dnerf":
    times = labels[-1]
    labels = labels[0]

  ls = []
  gots = []
  with torch.no_grad():
    for i in range(labels.shape[0]):
      ts = None if times is None else times[i:i+1, ...]
      exp = labels[i,...,:3]
      got = torch.zeros_like(exp)
      acc = torch.zeros_like(got)
      normals = torch.zeros_like(got)
      if args.backing_sdf: got_sdf = torch.zeros_like(got)
      if light is not None: model.refl.light = light[i:i+1]

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
          if hasattr(model, "nerf"):
            acc[c0:c0+args.crop_size, c1:c1+args.crop_size, :] = \
              model.nerf.acc_smooth()[0,...].clamp(min=0,max=1)
          if hasattr(model, "n") and hasattr(model, "nerf"):
            normals[c0:c0+args.crop_size, c1:c1+args.crop_size, :] = \
              (nerf.volumetric_integrate(model.nerf.weights, F.normalize(model.n, dim=-1))[0,...]+1)/2
          elif hasattr(model, "sdf"):
            ...
            #normals[c0:c0+args.crop_size, c1:c1+args.crop_size, :] = \
            #  model.sdf.debug_normals(

      gots.append(got)
      loss = F.mse_loss(got, exp)
      psnr = utils.mse2psnr(loss).item()
      ts = "" if ts is None else f",t={ts.item():.02f}"
      print(f"[{i:03}{ts}]: L2 {loss.item():.03f} PSNR {psnr:.03f}")
      name = f"train_{i:03}.png" if training else f"test_{i:03}.png"
      name = os.path.join(args.outdir, name)
      items = [exp, got.clamp(min=0, max=1)]
      #if hasattr(model, "nerf"): items.append(acc)
      if hasattr(model, "n") and hasattr(model, "nerf"): items.append(normals.clamp(min=0,max=1))
      save_plot(name, *items)
      ls.append(psnr)

  print(f"""[Summary ({"training" if training else "test"})]:
          mean {np.mean(ls):.03f}
          min {min(ls):.03f}
          max {max(ls):.03f}
          var {np.var(ls):.03f}""")
  if args.msssim_loss:
    msssim = utils.msssim_loss(gots, refs)
    print(f"\tms-ssim {mssim:.03f}")

# Sets these parameters on the model on each run, regardless if loaded from previous state.
def set_per_run(model, args):
  if len(args.replace) == 0: return
  ls = model.total_latent_size()
  if "occ" in args.replace:
    if args.occ_kind != None and hasattr(model, "occ"):
      model.occ = renderers.load_occlusion_kind(args.occ_kind, ls).to(device)

  if "refl" in args.replace:
    if args.refl_kind != "curr" and hasattr(model, "refl"):
      refl_inst = refl.load(args, ls).to(device)
      model.set_refl(refl_inst)
  if "bg" in args.replace:
    if isinstance(model, nerf.CommonNeRF): model.set_bg(args.bg)
    elif hasattr(model, "nerf"): model.nerf.set_bg(args.bg)

  if args.model == "sdf": return
  if not hasattr(model, "nerf"): return
  #if args.sigmoid_kind != "curr": model.nerf.set_sigmoid(args.sigmoid_kind)


def load_model(args):
  if args.model == "sdf": return sdf.load(args, with_integrator=True).to(device)
  if args.data_kind == "dnerf" and args.dnerfae: args.model = "ae"
  if args.model != "ae": args.latent_l2_weight = 0
  mip = utils.load_mip(args)
  per_pixel_latent_size = 64 if args.data_kind == "pixel-single" else 0
  per_pt_latent_size = (1 + 3 + 64) if args.backing_sdf else 0
  instance_latent_size = 0
  kwargs = {
    "mip": mip,
    "out_features": args.feature_space,
    "device": device,
    "steps": args.steps,
    "t_near": args.near,
    "t_far": args.far,
    "per_pixel_latent_size": per_pixel_latent_size,
    "per_point_latent_size": per_pt_latent_size,
    "instance_latent_size": instance_latent_size,
    "sigmoid_kind": args.sigmoid_kind,
    "sigmoid_kind": args.sigmoid_kind if args.sigmoid_kind != "curr" else "thin",
    "bg": args.bg,
  }
  if args.model == "tiny": constructor = nerf.TinyNeRF
  elif args.model == "plain": constructor = nerf.PlainNeRF
  elif args.model == "ae":
    constructor = nerf.NeRFAE
    kwargs["normalize_latent"] = args.normalize_latent
    kwargs["encoding_size"] = args.encoding_size
  elif args.model == "volsdf":
    constructor = nerf.VolSDF
    kwargs["sdf"] = sdf.load(args, with_integrator=False)
    kwargs["occ_kind"] = args.occ_kind
    kwargs["integrator_kind"] = args.integrator_kind or "direct"
  else: raise NotImplementedError(args.model)
  model = constructor(**kwargs).to(device)

  # set reflectance kind for new models (but volsdf handles it differently)
  if args.refl_kind != "curr":
    ls = model.refl.latent_size
    refl_inst = refl.load(args, ls).to(device)
    model.set_refl(refl_inst)

  if args.model == "ae" and args.latent_l2_weight > 0:
    model.set_regularize_latent()

  if args.mpi: model = MPI(canonical=model, device=device)

  # Add in a dynamic model if using dnerf with the underlying model.
  if args.data_kind == "dnerf":
    if args.with_canon is not None:
      model = torch.load(args.with_canon, map_location=device)
      assert(isinstance(model, nerf.CommonNeRF)), f"Can only use NeRF subtype, got {type(model)}"
      assert((not args.dnerfae) or isinstance(model, nerf.NeRFAE)), \
        f"Can only use NeRFAE canonical with DNeRFAE, got {type(model)}"
    constructor = nerf.DynamicNeRFAE if args.dnerfae else nerf.DynamicNeRF
    model = constructor(
      canonical=model,
      gru_flow=args.gru_flow,
      device=device
    ).to(device)
    if args.dnerf_tf_smooth_weight > 0: model.set_smooth_delta()

  if args.data_kind == "pixel-single":
    encoder = SpatialEncoder().to(device)
    # args.img is populated in load (single_image)
    model = nerf.SinglePixelNeRF(model, encoder=encoder, img=args.img, device=device).to(device)

  og_model = model
  if args.backing_sdf: model = sdf.SDFNeRF(model, sdf.SDF()).to(device)
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
    #setattr(model, "nerf", og_model) # TODO how to specify this?

  if args.data_parallel:
    model = nn.DataParallel(model)
    setattr(model, "nerf", og_model)

  if args.torchjit: model = torch.jit.script(model)
  if args.volsdf_alternate: model = nerf.AlternatingVolSDF(model)
  return model

def save(model, args):
  if args.nosave: return
  print(f"Saved to {args.save}")
  if args.torchjit: raise NotImplementedError()
  else: torch.save(model, args.save)
  if args.log is not None:
    setattr(args, "curr_time", datetime.today().strftime('%Y-%m-%d-%H:%M:%S'))
    with open(args.log, 'w') as f:
      json.dump(args.__dict__, f, indent=2)

def sphere_visualize(args, model):
  # TODO check this
  for i in range(args.sphere_visualize):
    for j in range(args.sphere_visualize):
      p = utils.spherical_pose(math.pi/(2 * j), 2 * math.pi/i, args.far + 1)
      cam = cameras.OrthogonalCamera(
        pos=p,
        at=torch.tensor([0,0,0], requires_grad=False),
        up=torch.tensor([0,1,0], requires_grad=False),
        view_width=args.far - args.near,
      )

      got = torch.zeros(args.render_size, args.render_size, 3, device=device)
      N = math.ceil(args.render_size/args.crop_size)
      for x in range(N):
        for y in range(N):
          c0 = x * args.crop_size
          c1 = y * args.crop_size
          out = render(
            model, cam[i:i+1, ...], (c0,c1,args.crop_size,args.crop_size), size=args.render_size,
            with_noise=False, args=args,
          ).squeeze(0)
          got[c0:c0+args.crop_size, c1:c1+args.crop_size, :] = out
      save_image(os.path.join(args.outdir, f"visualize_{i:03}_{j:03}.png"), got)


def seed(s):
  if s == -1: return
  torch.manual_seed(s)
  random.seed(s)
  np.random.seed(s)

def main():
  args = arguments()
  seed(args.seed)

  labels, cam, light = loaders.load(args, training=True, device=device)
  if args.train_imgs > 0:
    if type(labels) == tuple: labels = tuple(l[:args.train_imgs, ...] for l in labels)
    else: labels = labels[:args.train_imgs, ...]
    cam = cam[:args.train_imgs, ...]

  model = load_model(args) if args.load is None else torch.load(args.load, map_location=device)
  set_per_run(model, args)

  if args.train_parts == "all": parameters = model.parameters()
  elif args.train_parts == "refl": parameters = model.refl.parameters()
  elif args.train_parts == "camera": raise NotImplementedError("TODO")
  else: raise NotImplementedError()

  # for some reason AdamW doesn't seem to work here
  # eps = 1e-7 was in the original paper.
  opt = optim.Adam(parameters, lr=args.learning_rate, weight_decay=args.decay, eps=1e-7)

  # TODO should T_max = -1 or args.epochs
  sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=5e-5)
  if args.no_sched: sched = None
  train(model, cam, labels, opt, args, light=light, sched=sched)

  if args.notest: return

  if not args.notraintest: test(model, cam, labels, args, training=True, light=light)

  if not args.notest:
    test_labels, test_cam, test_light = loaders.load(args, training=False, device=device)
    test(model, test_cam, test_labels, args, training=False, light=test_light)

  if args.sphere_visualize is not None:
    sphere_visualize(args, model)

if __name__ == "__main__": main()

