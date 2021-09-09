import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import argparse
from src.utils import ( autograd, eikonal_loss, save_image, save_plot )
from src.neural_blocks import ( SkipConnMLP, FourierEncoder, PointNet )
from src.cameras import ( OrthogonalCamera )
from src.march import ( bisect )
from src.nerf import VolSDF
import src.refl as refl
from tqdm import trange
import matplotlib.pyplot as plt

import os
import random
import math
import numpy as np

def arguments():
  a = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  a.add_argument(
    # TODO add more here to learn between
    "--target", choices=["sphere", "volsdf"],
    default="sphere", help="What kind of SDF to learn",
  )
  a.add_argument(
    "--volsdf-model", type=str, help="Location of volsdf model to learn from",
  )
  a.add_argument(
    "--epochs", type=int, default=5000, help="Number of epochs to train for",
  )
  a.add_argument(
    "--bounds", type=float, default=1.5, help="Bounded region to train SDF in",
  )
  a.add_argument(
    "--batch-size", type=int, default=16, help="Number of batches to train at the same time",
  )
  a.add_argument(
    "--sample-size", type=int, default=1<<12, help="Number of points to train per batch",
  )
  a.add_argument(
    "--G-step", type=int, default=1, help="Number of steps to take before optimizing G",
  )
  a.add_argument(
    "--G-rep", type=int, default=1, help="Number of repeated G steps",
  )
  a.add_argument(
    "--eikonal-weight", type=float, default=1e-2, help="Weight of eikonal loss",
  )
  a.add_argument("--noglobal", action="store_true",help="Don't perform global discrimination")
  a.add_argument("--nolocal", action="store_true",help="Don't perform local discrimination")
  a.add_argument("--save-freq", type=int, default=5000, help="How often to save the model")
  a.add_argument("--num-test-samples", type=int, default=256, help="How many tests to run",)
  a.add_argument("--load", action="store_true", help="Load old generator and discriminator")
  a.add_argument("--render-size", type=int, default=180, help="Size to render result images")
  a.add_argument(
    "--refl-kind", type=str, default=None,
    choices = [None] + [r for r in refl.refl_kinds if r != "weighted"],
    help="The reflectance model in GAN to learn textures, if any",
  )
  a.add_argument("--nosave", action="store_true", help="Do not save.")
  a.add_argument("--G-lr", type=float, default=1e-3, help="Generator Learning rate")
  a.add_argument("--D-lr", type=float, default=5e-4, help="Discriminator Learning rate")
  a.add_argument("--D-decay", type=float, default=1e-5, help="D decay weight")
  # TODO maybe have something special for spherical harmonics since that can be used for
  # regression without a specific viewing direction?
  return a.parse_args()


device="cpu"
if torch.cuda.is_available():
  device = torch.device("cuda:0")
  torch.cuda.set_device(device)

# Computes samples within some bounds, returning [..., N, 3] samples.
def random_samples_within(bounds: [..., 6], samples:int = 1024):
  # lower-left, upper right
  ll, ur = bounds.split([3,3], dim=-1)
  rand = torch.rand(*ll.shape, device=bounds.device, requires_grad=True)
  samples = ll + rand * (ur - ll)
  return samples

# Picks a random cube bounding box inside of an existing bound.
# Please pass half the size of the desired bounding box, i.e. for a bounding box of size 1 pass
# 0.5.
def subbound(bounds: [..., 6], half_size: float):
  assert(half_size > 0), "Must pass positive size"
  ll, ur = bounds.split([3,3], dim=-1)
  center = torch.rand_like(ll)
  ll = ll + half_size
  ur = ur - half_size
  center = (center + ll) * (ur - ll)
  return torch.cat([center-half_size, center+half_size], dim=-1)

# rescales the points inside of the bound to a canonical [-1, 1] space.
# The half_size of the bounding box is necessary to compute the scaling factor.
def rescale_pts_in_bound(ll, pts, sdf_values, half_size: float, end_size:float, dst_ll):
  sz_ratio = end_size/half_size
  # TODO add different final location parameter
  rescaled_pts = ((pts - ll)/half_size - 1) * end_size
  # https://iquilezles.org/www/articles/distfunctions/distfunctions.htm
  rescaled_sdf_values = sdf_values * sz_ratio

  return rescaled_pts, rescaled_sdf_values

def pts_from_bd_to(ll1, ll2, pts, sdf_values, size_ratio: float):
  new_pts = (pts - ll1) * size_ratio + ll2
  new_sdf_values = sdf_values * size_ratio
  return new_pts, new_sdf_values


# [WIP] returns semi-equispaced samples within some bounds, sampling a fixed amt of times per
# dimesnion.
def stratified_rand_samples(bounds: [..., 6], samples_per_dim:int=8):
  ll, ur = bounds.split([3,3], dim=-1)
  dim = torch.linspace(0,1,samples_per_dim, device=ll.device, dtype=torch.float)
  samples = torch.stack(torch.meshgrid(dim, dim, dim), dim=-1).reshape(-1, 3)
  samples = samples + torch.randn_like(samples)*0.01
  print(samples.shape, ll.shape)
  samples = ll.unsqueeze(-2) + samples * (ur-ll).unsqueeze(-2)
  exit()
  return samples.requires_grad_()

# computes boundes for a bunch of samples.
def bounds_for(samples: [..., "Batch", 3]):
  ll = samples.min(dim=-2)[0]
  ur = samples.max(dim=-2)[0]
  return torch.cat([ll, ur], dim=-1)

def scaled_training_step(
  i, latent_noise, view,

  target, bounds, max_bound,
  G, opt_G, prev_G_loss,
  D, opt_D,

  args,
):
  D.zero_grad()
  exp_sample_size = 0.5 + random.random()/2
  subbd_exp = subbound(bounds, exp_sample_size)
  exp_pts = random_samples_within(subbd_exp, args.sample_size)
  exp_sub_vals = target(exp_pts, view)
  exp_pts, exp_sub_vals = rescale_pts_in_bound(
    subbd_exp[..., :3], exp_pts, exp_sub_vals, exp_sample_size,

    1, subbound(bounds, 1)[..., :3],
  )
  s2_real = D(exp_pts.detach(), exp_sub_vals.detach())

  got_sample_size = 0.5 + random.random()/2
  subbd_got = subbound(bounds, got_sample_size)
  #got_tgt_bd_sz = 0.25 + random.random()/2
  #subbd_got_tgt = subbound(bounds, got_tgt_bd_sz)
  got_pts = random_samples_within(subbd_got, args.sample_size)
  got_sub_vals = G(got_pts, view, latent=latent_noise)
  new_got_pts, got_sub_vals = rescale_pts_in_bound(
    subbd_got[..., :3], got_pts, got_sub_vals, got_sample_size,

    1, subbound(bounds, 1)[..., :3],
  )
  s2_fake = D(new_got_pts.detach(), got_sub_vals.detach())

  real_loss = F.binary_cross_entropy_with_logits(s2_real, torch.ones_like(s2_real))
  fake_loss = F.binary_cross_entropy_with_logits(s2_fake, torch.zeros_like(s2_fake))

  D_loss = real_loss + fake_loss
  D_loss.backward()
  opt_D.step()

  G_loss = prev_G_loss
  if i % args.G_step == 0:
    for i in range(args.G_rep):
      # partial SDF training
      G.zero_grad()

      got_pts = random_samples_within(subbd_got, args.sample_size)
      got_sub_vals, got_n, _sdf_latent = G.vals_normal(got_pts, view, latent=latent_noise)
      new_got_pts, got_sub_vals = rescale_pts_in_bound(
        subbd_got[..., :3], got_pts, got_sub_vals, got_sample_size,

        1, subbound(bounds, 1)[..., :3],
      )

      s2_fool = D(got_pts, got_sub_vals)
      fooling_loss = F.binary_cross_entropy_with_logits(s2_fool, torch.ones_like(s2_fool))

      G_loss = fooling_loss
      (G_loss + args.eikonal_weight*eikonal_loss(got_n)).backward()
      opt_G.step()
      G_loss = G_loss.item()
  return D_loss.item(), G_loss

def whole_training_step(
  i, latent_noise, view,

  target, bounds, max_bound,
  G, opt_G, G_loss,
  D, opt_D, D_loss,

  args,
):
  pt_samples1 = random_samples_within(bounds, args.sample_size)

  D.zero_grad()
  exp = target(pt_samples1, view)

  got = G(pt_samples1, view, latent_noise)

  s1_real = D(pt_samples1.detach(), exp.detach())
  s1_fake = D(pt_samples1.detach(), got.detach())

  real_loss = F.binary_cross_entropy_with_logits(s1_real, torch.ones_like(s1_real))
  fake_loss = F.binary_cross_entropy_with_logits(s1_fake, torch.zeros_like(s1_fake))

  D_loss = real_loss + fake_loss
  D_loss.backward()
  opt_D.step()
  D_loss = D_loss.item()

  if i % args.G_step == 0:
    G.zero_grad()
    #pt_samples1 = random_samples_within(bounds, args.sample_size)
    got, got_n, _sdf_latent = G.vals_normal(pt_samples1, view=view, latent=latent_noise)

    s1_fool = D(pt_samples1.detach(), got)
    fooling_loss = F.binary_cross_entropy_with_logits(s1_fool, torch.ones_like(s1_fool))

    G_loss = fooling_loss
    (G_loss + args.eikonal_weight*eikonal_loss(got_n)).backward()
    opt_G.step()
    G_loss = G_loss.item()
  return D_loss, G_loss

# trains a GAN with a target SDF, as well as a discriminator point net.
def train(
  targets,
  model, opt_G,
  discriminator, opt_D,

  args,
):
  if args.epochs == 0: return

  D = discriminator
  max_b = abs(args.bounds)
  min_b = -max_b
  bounds = torch.tensor([
    min_b, min_b, min_b, max_b, max_b, max_b,
  ], device=device, dtype=torch.float)
  sample_size = args.sample_size
  batch_size = args.batch_size
  bounds = bounds[None,None,:].expand(batch_size, sample_size, 6)
  # if we do not have a reflectance model we do not need to sample view directions.
  get_view = lambda: None
  if isinstance(model, SDFAndRefl):
    get_view = lambda: \
      F.normalize(torch.randn(batch_size, sample_size, 3, device=device), eps=1e-6, dim=-1)

  G_losses, D_losses, sG_losses, sD_losses = [], [], [], []
  t = trange(args.epochs)
  G_loss, D_loss, sG_loss, sD_loss = None, None, None, None
  for i in t:
    target = random.choice(targets)
    # whole SDF discriminator step
    view = get_view()
    latent_noise = torch.randn(batch_size, 1, model.latent_size, device=device)\
      .mul(5)\
      .expand(batch_size, sample_size, model.latent_size)

    if not args.noglobal:
      D_loss, G_loss = whole_training_step(
        i, latent_noise, view,

        target, bounds, max_b,
        model, opt_G, G_loss,
        D, opt_D, D_loss,

        args,
      )

      t.set_postfix(D=f"{D_loss:.03f}", G=f"{G_loss:.03f}")
      G_losses.append(G_loss)
      D_losses.append(D_loss)

    if not args.nolocal:
      sD_loss, sG_loss = scaled_training_step(
        i, latent_noise, view,

        target, bounds, max_b,
        model, opt_G, sG_loss,
        D, opt_D,
        args,
      )
      sG_losses.append(sG_loss)
      sD_losses.append(sD_loss)
      if args.noglobal: t.set_postfix(sD=f"{sD_loss:.03f}", sG=f"{sG_loss:.03f}")

    # not sure why I removed this but I don't see a super great need to re-enable it
    if i != 0 and i % args.save_freq == 0: save(model, D, args)
    ...
  save(model, D, args)
  if len(G_losses) != 0: save_losses(args, G_losses, D_losses)
  if len(sG_losses) != 0: save_losses(args, sG_losses, sD_losses, scaled=True)

def save_losses(args, G_losses, D_losses, scaled=False):
  window = 100
  assert(len(G_losses) == len(D_losses))
  loss_len = len(G_losses)
  window = min(window, loss_len)
  G_losses = np.convolve(G_losses, np.ones(window)/window, mode='valid')
  D_losses = np.convolve(D_losses, np.ones(window)/window, mode='valid')
  loss_len = len(G_losses)

  plt.plot(range(loss_len), G_losses, label='G')
  plt.plot(range(loss_len), D_losses, label='D')
  plt.legend()
  title = "scaled_gan_loss" if scaled else "gan_loss.png"
  plt.savefig(os.path.join("outputs/", title), bbox_inches="tight")
  plt.close()

# unit sphere
sphere = lambda x, v: torch.linalg.norm(x, dim=-1, keepdim=True) - 1
# random size sphere
rand_sphere = lambda x, v: torch.linalg.norm(x, dim=-1, keepdim=True) - (random.random() + 0.1)
# random torus for fun
def rand_torus(p, v):
  x,y,z = p.split([1,1,1], dim=-1)
  q = torch.stack([torch.cat([x,z], dim=-1).norm(dim=-1) - random.random(), y.squeeze(-1)], dim=-1)
  out = q.norm(dim=-1,keepdim=True) - random.random()
  return out

# a box with center on the origin of a given size
def origin_aabb(size:float):
  def aux(x, v):
    v = x.abs() - size
    return v.clamp(min=1e-6).norm(keepdim=True, dim=-1) + \
      v.max(dim=-1, keepdim=True)[0].clamp(max=-1e-6)
  return aux

# the intersection of two sdfs (bounded correct)
def intersection(a, b): return lambda x, v: a(x, v).maximum(b(x, v))

# given an sdf and some pts, computes the SDF at those points & the normal
def values_normals(sdf, pts):
  assert(pts.requires_grad)
  values = sdf(pts)
  if values.shape[-1] != 1: values = values.unsqueeze(-1)
  assert(values.shape[-1] == 1), f"unsqueeze a dimension from {values.shape}"
  normals = autograd(pts, values)
  return values, normals

class SDFAndRefl(nn.Module):
  def __init__(self, sdf, refl):
    super().__init__()
    self.model = sdf
    self.refl = refl
    self.dropout = nn.Dropout(inplace=True)
  def forward(self, x, view, latent=None):
    v = self.model(x, view, latent)
    v, sdf_latent = v[..., 0, None], v[..., 1:]
    rgb = self.refl(x, view, latent=sdf_latent)
    # we do not care about the color off the surface of the SDF, but really close is probably
    # similar so regress with some epsilon.
    #rgb = torch.where(v.abs() < 1e-1, rgb, torch.zeros_like(rgb))
    # the farther the distance from the surface the less the color matters
    scale = 1e-2
    weight = laplace_cdf(v, scale)
    print(weight.min().item(), weight.max().item()
    density = 1/scale * weight
    sigma_a = F.softplus(density-1)
    alpha = 1 - torch.exp(-sigma_a * 1e-1) # some small delta to measure change
    print(alpha.shape, rgb.shape)
    exit()
    rgb = rgb * alpha

    return torch.cat([v, rgb], dim=-1)

  # returns the SDF of this (and any children's SDF if nested)
  @property
  def sdf(self): return self.model

  @property
  def latent_size(self): return self.model.latent_size
  def set_assigned_latent(self, latent): return self.model.set_assigned_latent(latent)

  def vals_normal(self, x, view, latent=None):
    v, normal, sdf_latent = self.model.vals_normal(x, view, latent)
    rgb = self.refl(x, view, latent=sdf_latent)
    # return SDF latent here for consistency
    return torch.cat([v, rgb], dim=-1), normal, sdf_latent

  def set_to(self, sdf): self.model.set_to(sdf)

class MLP(nn.Module):
  def __init__(
    self,
    latent_size:int=32,
    output_latent_size:int=0,
    bounds:float=1.5,
  ):
    super().__init__()
    self.latent_size = latent_size
    self.assigned_latent = None
    self.mlp = SkipConnMLP(
      in_size=3, out=1 + output_latent_size,
      latent_size=latent_size,
      enc=FourierEncoder(input_dims=3),
      activation=torch.sin,
      num_layers=7, hidden_size=512,
      skip=3,
      xavier_init=True,
    )
    self.bounds = bounds
  def set_assigned_latent(self, latent): self.assigned_latent = latent
  def forward(self, x, view=None, latent=None):
    l = latent if latent is not None else self.assigned_latent.expand(*x.shape[:-1], -1)
    predicted = self.mlp(x, l)
    return predicted
    #if self.training: return predicted

    predicted, latent = predicted[...,0,None], predicted[..., 1:]

    v = x.abs() - self.bounds
    bounds = v.clamp(min=1e-6).norm(keepdim=True, dim=-1) + \
      v.max(dim=-1, keepdim=True)[0].clamp(max=-1e-6)
    return torch.cat([predicted.maximum(bounds), latent], dim=-1)
  @property
  def sdf(self): return self

  def vals_normal(self, x, view=None, latent=None):
    with torch.enable_grad():
      pts = x if x.requires_grad else x.requires_grad_()
      values = self(pts, view, latent)
      sdf = values[..., 0, None]
      latent = values[..., 1:]
      normals = autograd(pts, sdf)
      return sdf, normals, latent
  def set_to(self, sdf):
    opt = optim.Adam(self.parameters(), lr=1e-3)
    t = trange(1024)
    for i in t:
      opt.zero_grad()
      pts = torch.randn(1024, 3, device=device) * 3
      latent_noise = torch.randn(1024, self.latent_size, device=device)
      loss = F.mse_loss(self(pts, latent_noise)[..., 0], sdf(pts))
      loss.backward()
      opt.step()
      # TODO could add eikonal loss, not sure if worth since it's a blind copy.
      t.set_postfix(l2=f"{loss:.03f}")

# wraps just the SDF component of a volsdf to make it compatible with the GAN API
class VolSDFWrapper(nn.Module):
  def __init__(self, sdf, with_refl=True):
    super().__init__()
    self.sdf = sdf
    self.with_refl = with_refl
  def forward(self, pts, view, latent=None):
    sdf, latent = self.sdf.from_pts(pts)
    sdf = sdf.unsqueeze(-1)
    if self.with_refl and latent is not None: sdf = torch.cat([sdf, latent], dim=-1)
    return sdf

# renders an SDF using the given camera and crop=(top, left, width, height), of size of the
# image.
def render(
  G, cam, crop,
  size,
):
  ii, jj = torch.meshgrid(
    torch.arange(size, device=device, dtype=torch.float),
    torch.arange(size, device=device, dtype=torch.float),
  )
  positions = torch.stack([ii.transpose(-1, -2), jj.transpose(-1, -2)], dim=-1)
  t,l,h,w = crop
  positions = positions[t:t+h,l:l+w,:]
  rays = cam.sample_positions(positions, size=size, with_noise=0)

  r_o, r_d = rays.split([3,3], dim=-1)

  near = 2
  far = 6
  sdf = G.sdf
  pts, hits, best_pts, _ = bisect(sdf, r_o, r_d, eps=0, near=near, far=far)
  pts = pts.requires_grad_()

  sdf_latent = None
  with torch.enable_grad():
    vals, normals, sdf_latent = G.vals_normal(pts, r_d)
  normals = F.normalize(normals, dim=-1)
  normals = (normals+1)/2
  normals[~hits] = 0

  t = torch.linalg.norm(pts - r_o, dim=-1, keepdim=True)
  depths = (t - near)/(far - near)
  depths[~hits] = 0

  rgb = None
  if vals.shape[-1] == 4: rgb = vals[0, ..., 1:]

  return normals, depths, rgb

def save(model, disc, args):
  if args.nosave or args.epochs == 0: return
  torch.save(model, "models/G_sdf.pt")
  torch.save(disc, "models/D_sdf.pt")

def load_model(args):
  # binary classification
  feats = 4
  output_latent_size = 0
  r = None
  if args.refl_kind is not None:
    setattr(args, "feature_space", 3)
    setattr(args, "normal_kind", "raw")
    setattr(args, "light_kind", None)
    setattr(args, "sigmoid_kind", "normal")
    output_latent_size = 64 # TODO make arg # output latent_size to pass to refl
    r = refl.load(args, args.refl_kind, "identity", output_latent_size)
    feats += 3 # RGB
  model = MLP(bounds=args.bounds, output_latent_size=output_latent_size)

  if r is not None: model = SDFAndRefl(sdf=model, refl=r)

  discrim = PointNet(feature_size=feats, classes=1)
  return model.to(device), discrim.to(device)

def load_targets(args):
  if args.target == "sphere": return [sphere]
  elif args.target == "volsdf":
    volsdf = torch.load(args.volsdf_model)
    assert(isinstance(volsdf, VolSDF)), "Can only pass VolSDF model"
    sdf = volsdf.sdf
    if args.refl_kind is not None:
      return [SDFAndRefl(sdf=VolSDFWrapper(sdf), refl=sdf.refl)]
    else: return [VolSDFWrapper(sdf, with_refl=False)]
  else: raise NotImplementedError

def main():
  args = arguments()
  if not args.load:
    model, discrim = load_model(args)
  else:
    model, discrim = torch.load("models/G_sdf.pt"), torch.load("models/D_sdf.pt")
  opt_G = optim.Adam(model.parameters(), lr=args.G_lr)
  opt_D = optim.Adam(discrim.parameters(), lr=args.D_lr, weight_decay=args.D_decay)
  # select a set of target SDFs.
  targets = load_targets(args)

  train(targets, model, opt_G, discrim, opt_D, args)

  model.eval()
  sz = args.render_size
  with torch.no_grad():
    start_l = torch.randn(model.latent_size, device=device) * 3
    end_l = torch.randn(model.latent_size, device=device) * 3
    nts = args.num_test_samples
    for i in trange(nts):

      cam = OrthogonalCamera(
        pos = torch.tensor(
          # spinny camera (circle on x/y)
          [[3*math.cos(i * math.pi/64),3*math.sin(i * math.pi/64), 3]],
          device=device, dtype=torch.float,
        ),
        at = torch.tensor([[0,0,0]], device=device, dtype=torch.float),
        up = torch.tensor([[0,1,0]], device=device, dtype=torch.float),
        view_width=1.75,
      )

      # linearly interpolate between two random starting and ending points.
      t = i/nts
      model.set_assigned_latent(start_l * (1 - t) + end_l * t)

      # need to add learned reflectance here
      normals, depths, rgb = render(model, cam, (0,0,sz,sz), sz)
      items = [normals.squeeze(0), depths.squeeze(0)]
      if rgb is not None: items.append(rgb)
      save_plot(f"outputs/sdf_gan_{i:03}.png", *items)
  return

if __name__ == "__main__": main()
