import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import math

from .neural_blocks import ( SkipConnMLP, FourierEncoder )
from .utils import ( autograd, eikonal_loss, dir_to_elev_azim )
from .refl import ( LightAndRefl )

def load(args, shape, light_and_refl: LightAndRefl):
  assert(isinstance(light_and_refl, LightAndRefl)), "Need light and reflectance for integrator"

  if args.integrator_kind is None: return None
  elif args.integrator_kind == "direct": cons = Direct
  elif args.integrator_kind == "path": cons = Path
  else: raise NotImplementedError(f"load integrator: {args.integrator_kind}")
  occ = load_occlusion_kind(args.occ_kind)

  integ = cons(shape=shape, refl=light_and_refl.refl, occlusion=occ)

  return integ

def load_occlusion_kind(kind=None):
  if kind is None: occ = lighting_wo_isect
  elif kind == "hard": occ = LightingWIsect()
  elif kind == "learned": occ = LearnedLighting()
  else: raise NotImplementedError(f"load occlusion: {args.occ_kind}")

  return occ

# no shadow
def lighting_wo_isect(pts, lights, isect_fn, mask=None):
  return lights(pts if mask is None else pts[mask], mask=mask)

# hard shadow lighting
class LightingWIsect(nn.Module):
  def __init__(self): super().__init__()
  def forward(self, pts, lights, isect_fn, mask=None):
    pts = pts if mask is None else pts[mask]
    dir, spectrum = lights(pts, mask=mask)
    visible = isect_fn(pts, -dir, near=1e-3, far=20)
    spectrum = torch.where(
      visible[...,None],
      spectrum,
      torch.zeros_like(spectrum)
    )
    return dir, spectrum

class LearnedLighting(nn.Module):
  def __init__(self):
    super().__init__()
    self.attenuation = SkipConnMLP(
      in_size=5, out=1,
      num_layers=5, hidden_size=256,
      xavier_init=True,
    )
  def forward(self, pts, lights, isect_fn, mask=None):
    pts = pts if mask is None else pts[mask]
    dir, spectrum = lights(pts, mask=mask)
    visible = isect_fn(pts, -dir, near=1e-3, far=20)
    att = self.attenuation(torch.cat([pts, dir_to_elev_azim(dir)], dim=-1)).sigmoid()
    spectrum = torch.where(visible.reshape_as(att), spectrum, spectrum * att)
    return dir, spectrum

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
  def forward(self, rays):
    r_o, r_d = rays.split([3, 3], dim=-1)

    pts, hits, _t, n = self.shape.intersect_w_n(r_o, r_d)
    _, latent = self.shape.from_pts(pts[hits])

    light_dir, light_val = self.occ(pts, self.refl.light, self.shape.intersect_mask, mask=hits)
    bsdf_val = self.refl(x=pts[hits], view=r_d[hits], normal=n[hits], light=light_dir, latent=latent)
    out = torch.zeros_like(r_d)
    out[hits] = bsdf_val * light_val
    if self.training:
      out = torch.cat([out, self.shape.throughput(r_o, r_d)], dim=-1)
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
