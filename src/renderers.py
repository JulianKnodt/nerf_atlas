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
  return lights(pts if mask is None else pts[mask], mask=mask)

# hard shadow lighting
class LightingWIsect(nn.Module):
  def __init__(self): super().__init__()
  def forward(self, pts, lights, isect_fn, latent=None, mask=None):
    pts = pts if mask is None else pts[mask]
    dir, spectrum = lights(pts, mask=mask)
    visible = isect_fn(pts, -dir, near=0.1, far=20)
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
    self.attenuation = SkipConnMLP(
      in_size=5, out=1, latent_size=latent_size,
      num_layers=5, hidden_size=256,
      xavier_init=True,
    )
  def forward(self, pts, lights, isect_fn, latent=None, mask=None):
    pts = pts if mask is None else pts[mask]
    dir, spectrum = lights(pts, mask=mask)
    visible = isect_fn(pts, -dir, near=0.1, far=20)
    att = self.attenuation(
      torch.cat([pts, dir_to_elev_azim(dir)], dim=-1),
      latent
    ).sigmoid()
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
      num_layers=5, hidden_size=256, xavier_init=True,
    )
  def forward(self, pts, lights, isect_fn, latent=None, mask=None):
    pts = pts if mask is None else pts[mask]
    dir, spectrum = lights(pts, mask=mask)
    visible = isect_fn(pts, -dir, near=0.1, far=20).unsqueeze(-1)
    elaz = dir_to_elev_azim(dir)
    # try squaring to encode the symmetry on both sides of asin
    att = self.attenuation(torch.cat([pts, elaz, visible], dim=-1), latent).sigmoid()
    spectrum = spectrum * att
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

    light_dir, light_val = self.occ(
      pts + n*1e-3, self.refl.light, self.shape.intersect_mask, mask=hits,
      latent=latent,
    )
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
