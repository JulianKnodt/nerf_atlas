import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import math

from .neural_blocks import ( SkipConnMLP, NNEncoder, FourierEncoder )
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

# no shadow
def lighting_wo_isect(pts, lights, isect_fn, latent=None, mask=None):
  dir, _, spectrum = lights(pts if mask is None else pts[mask], mask=mask)
  return dir, spectrum

# hard shadow lighting
class LightingWIsect(nn.Module):
  def __init__(self, latent_size:int): super().__init__()
  def forward(self, pts, lights, isect_fn, latent=None, mask=None):
    pts = pts if mask is None else pts[mask]
    dir, dist, spectrum = lights(pts, mask=mask)
    far = dist.max().item() if mask.any() else 6
    visible, _, _ = isect_fn(pts, dir, near=0.1, far=far)
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
    in_size=5
    self.attenuation = SkipConnMLP(
      in_size=in_size, out=1, latent_size=latent_size, num_layers=5, hidden_size=128,
      enc=FourierEncoder(input_dims=in_size), xavier_init=True,
    )
  def forward(self, pts, lights, isect_fn, latent=None, mask=None):
    pts = pts if mask is None else pts[mask]
    dir, dist, spectrum = lights(pts, mask=mask)
    far = dist.max().item() if mask.any() else 6
    # TODO why doesn't this isect fn seem to work?
    visible, _, _ = isect_fn(r_o=pts, r_d=dir, near=2e-3, far=far, eps=1e-3)
    elaz = dir_to_elev_azim(dir)
    att = self.attenuation(torch.cat([pts, elaz], dim=-1), latent).sigmoid()
    spectrum = torch.where(visible.reshape_as(att), spectrum, spectrum * att)
    return dir, spectrum

class LearnedConstantSoftLighting(nn.Module):
  def __init__(
    self,
    latent_size:int=0,
  ):
    super().__init__()
    in_size=5
    self.alpha = nn.Parameter(torch.tensor(0., requires_grad=True), requires_grad=True)
  def forward(self, pts, lights, isect_fn, latent=None, mask=None):
    pts = pts if mask is None else pts[mask]
    dir, dist, spectrum = lights(pts, mask=mask)
    far = dist.max().item() if mask and mask.any() else 6
    # TODO why doesn't this isect fn seem to work?
    visible, _, _ = isect_fn(r_o=pts, r_d=dir, near=1e-1, far=far, eps=1e-3)
    spectrum = torch.where(
      visible.unsqueeze(-1),
      spectrum,
      spectrum * self.alpha.sigmoid(),
    )
    return dir, spectrum

def elaz_and_3d(dir): return torch.cat([dir_to_elev_azim(dir), dir], dim=-1)

class AllLearnedOcc(nn.Module):
  def __init__(
    self,
    latent_size:int=0,
    with_dir=False,
  ):
    super().__init__()
    in_size=5 + (3 if with_dir else 0)
    # it seems that passing in the fourier encoder with just the elaz is good enough,
    # can also pass in the direction to handle the edge case.
    self.attenuation = SkipConnMLP(
      in_size=in_size, out=1, latent_size=latent_size,
      enc=FourierEncoder(input_dims=in_size),
      num_layers=6, hidden_size=512, xavier_init=True,
    )
    self.encode_dir = elaz_and_3d if with_dir else dir_to_elev_azim
  def forward(self, pts, lights, isect_fn, latent=None, mask=None):
    pts = pts if mask is None else pts[mask]
    dir, _, spectrum = lights(pts, mask=mask)
    att = self.attenuation(torch.cat([pts, self.encode_dir(dir)], dim=-1), latent).sigmoid()

    return dir, spectrum * att

# Learned approximate penumbra based on the SDF values based on how close nearby points
# are.
# Inspired by https://iquilezles.org/www/articles/rmshadows/rmshadows.htm
# Doesn't work so well, maybe worth continuing to experiment with it?
class ApproximateSmoothShadow(nn.Module):
  def __init__(
    self,
    latent_size:int=0,
  ):
    super().__init__()
    self.attenuation = SkipConnMLP(
      in_size=2, out=1, latent_size=latent_size,
      num_layers=6, hidden_size=180, xavier_init=True, skip=999,
    )
  def forward(self, pts, lights, isect_fn, latent=None, mask=None):
    pts = pts if mask is None else pts[mask]
    dir, dist, spectrum = lights(pts, mask=mask)
    far = dist.max().item()
    visible, min_sdf_val, point_of_min = \
      isect_fn(r_o=pts, r_d=dir, near=2e-3, far=far, eps=1e-3)
    assert(min_sdf_val is not None), "Cannot use Approx Smooth Shadow w/o throughput"
    dists = torch.linalg.norm(pts - point_of_min, dim=-1, ord=2, keepdim=True)
    att = self.attenuation(torch.cat([min_sdf_val, dists], dim=-1), latent).sigmoid()
    return dir, spectrum * att

occ_kinds = {
  None: lambda **kwargs: lighting_wo_isect,
  "hard": LightingWIsect,
  "learned": LearnedLighting,
  "learned-const": LearnedConstantSoftLighting,
  "all-learned": AllLearnedOcc,
  #"approx-soft": ApproximateSmoothShadow,
}

def load_occlusion_kind(kind=None, latent_size:int=0):
  con = occ_kinds.get(kind, -1)
  if con == -1: raise NotImplementedError(f"load occlusion: {args.occ_kind}")
  return con(latent_size=latent_size)

class Renderer(nn.Module):
  def __init__(
    self, shape, refl,
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
  def forward(s, rays): return direct(s.shape, s.refl, s.occ, rays, s.training)

# Functional version of integration
def direct(shape, refl, occ, rays, training=True):
  r_o, r_d = rays.split([3, 3], dim=-1)

  pts, hits, tput, n = shape.intersect_w_n(r_o, r_d)
  _, latent = shape.from_pts(pts[hits])

  out = torch.zeros_like(r_d)
  for light in refl.light.iter():
    light_dir, light_val = occ(pts, light, shape.intersect_mask, mask=hits, latent=latent)
    bsdf_val = refl(x=pts[hits], view=r_d[hits], normal=n[hits], light=light_dir, latent=latent)
    out[hits] = out[hits] + bsdf_val * light_val
  if training: out = torch.cat([out, tput], dim=-1)
  return out

def path(shape, refl, occ, rays, training=True):
  r_o, r_d = rays.split([3, 3], dim=-1)

  pts, hits, tput, n = shape.intersect_w_n(r_o, r_d)
  _, latent = shape.from_pts(pts[hits])

  out = torch.zeros_like(r_d)
  for light in refl.light.iter():
    light_dir, light_val = occ(pts, light, shape.intersect_mask, mask=hits, latent=latent)
    bsdf_val = refl(x=pts[hits], view=r_d[hits], normal=n[hits], light=light_dir, latent=latent)
    out[hits] = out[hits] + bsdf_val * light_val

  # TODO this should just be a random sample of pts in some range?
  pts_2nd_ord = pts.reshape(-1, 3)
  pts_2nd_ord = pts[torch.randint(high=pts_2nd_ord.shape[0], size=32, device=pts.device), :]
  with torch.no_grad():
    # compute light to set of points
    light_dir, light_val = occ(pts_2nd_ord, shape.intersect_mask, latent=latent)
    # compute dir from each of the 2nd order pts to the main points
    dirs = pts_2nd_ord - pts
    # TODO apply the learned occlusion here
    att = occ.attenuation(torch.cat([pts_2nd_ord, dirs], dim=-1), latent=latent)
    # TODO this should account for the BSDF when computing the reflected component
    out[hits] = out[hits] + att * light_val
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
