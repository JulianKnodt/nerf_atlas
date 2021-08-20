import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import math

from .neural_blocks import ( SkipConnMLP, NNEncoder, FourierEncoder )
from .utils import ( autograd, eikonal_loss, dir_to_elev_azim, fat_sigmoid )
from .refl import ( LightAndRefl )

def load(args, shape, light_and_refl: LightAndRefl):
  assert(isinstance(light_and_refl, LightAndRefl)), "Need light and reflectance for integrator"

  if args.integrator_kind is None: return None
  elif args.integrator_kind == "direct": cons = Direct
  elif args.integrator_kind == "path": cons = Path
  else: raise NotImplementedError(f"load integrator: {args.integrator_kind}")
  ls = 0
  if hasattr(shape, "latent_size"): ls = shape.latent_size
  elif hasattr(shape, "total_latent_size"): ls = shape.total_latent_size()
  occ = load_occlusion_kind(args.occ_kind, ls)

  integ = cons(shape=shape, refl=light_and_refl.refl, occlusion=occ)

  return integ

def load_occlusion_kind(kind=None, latent_size:int=0):
  if kind is None: occ = lighting_wo_isect
  elif kind == "hard": occ = LightingWIsect()
  elif kind == "learned": occ = LearnedLighting(latent_size=latent_size)
  elif kind == "all-learned": occ = AllLearnedOcc(latent_size=latent_size)
  else: raise NotImplementedError(f"load occlusion: {args.occ_kind}")

  return occ

# no shadow
def lighting_wo_isect(pts, lights, isect_fn, latent=None, mask=None):
  dir, _, spectrum = lights(pts if mask is None else pts[mask], mask=mask)
  return dir, spectrum

# hard shadow lighting
class LightingWIsect(nn.Module):
  def __init__(self): super().__init__()
  def forward(self, pts, lights, isect_fn, latent=None, mask=None):
    pts = pts if mask is None else pts[mask]
    dir, dist, spectrum = lights(pts, mask=mask)
    far = dist.max().item() if mask.any() else 6
    visible = isect_fn(pts, dir, near=0.1, far=far)
    spectrum = torch.where(
      visible[...,None],
      spectrum,
      torch.zeros_like(spectrum)
    )
    return dir, spectrum

class LearnedLighting(nn.Module):
  def __init__(
    self,
    latent_size:int=0,
  ):
    super().__init__()
    in_size=6
    self.attenuation = SkipConnMLP(
      in_size=in_size, out=1, latent_size=latent_size, num_layers=5, hidden_size=128,
      enc=FourierEncoder(input_dims=in_size), xavier_init=True,
    )
  def forward(self, pts, lights, isect_fn, latent=None, mask=None):
    pts = pts if mask is None else pts[mask]
    dir, dist, spectrum = lights(pts, mask=mask)
    far = dist.max().item() if mask.any() else 6
    # TODO why doesn't this isect fn seem to work?
    visible = isect_fn(r_o=pts, r_d=dir, near=2e-3, far=3, eps=1e-3)
    att = self.attenuation(torch.cat([pts, dir], dim=-1), latent).sigmoid()
    spectrum = torch.where(visible.reshape_as(att), spectrum, spectrum * att)
    return dir, spectrum

class AllLearnedOcc(nn.Module):
  def __init__(
    self,
    latent_size:int=0,
  ):
    super().__init__()
    in_size=6
    self.attenuation = SkipConnMLP(
      in_size=in_size, out=1, latent_size=latent_size,
      enc=FourierEncoder(input_dims=in_size),
      num_layers=5, hidden_size=128, xavier_init=True,
    )
  def forward(self, pts, lights, isect_fn, latent=None, mask=None):
    pts = pts if mask is None else pts[mask]
    dir, _, spectrum = lights(pts, mask=mask)
    att = self.attenuation(torch.cat([pts, dir], dim=-1), latent).sigmoid()
    return dir, spectrum * att

class Renderer(nn.Module):
  def __init__(
    self,
    shape,
    refl,

    occlusion,
  ):
    super().__init__()
    self.shape = shape
    self.refl = refl
    self.occ = occlusion

  def forward(self, _rays): raise NotImplementedError()

class Direct(Renderer):
  def __init__(self, **kwargs):
    super().__init__(**kwargs)
  @property
  def sdf(self): return self.shape
  def total_latent_size(self): return self.shape.latent_size
  def set_refl(self, refl): self.refl = refl
  def forward(self, rays): return direct(self.shape, self.refl, self.occ, rays, self.training)

# Functional version of integration
def direct(shape, refl, occ, rays, training=True):
  r_o, r_d = rays.split([3, 3], dim=-1)

  pts, hits, tput, n = shape.intersect_w_n(r_o, r_d)
  _, latent = shape.from_pts(pts[hits])

  light_dir, light_val = occ(pts, refl.light, shape.intersect_mask, mask=hits, latent=latent)
  bsdf_val = refl(x=pts[hits], view=r_d[hits], normal=n[hits], light=light_dir, latent=latent)
  out = torch.zeros_like(r_d)
  out[hits] = bsdf_val * light_val
  if training: out = torch.cat([out, tput], dim=-1)
  return out

class Path(Renderer):
  def __init__(
    self,
    bounces:int=1,
    **kwargs,
  ):
    super().__init__(**kwargs)
    self.bounces = bounces
  def forward(self, rays):
    raise NotImplementedError()
