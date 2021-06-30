import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import math

from .nerf import ( CommonNeRF, compute_pts_ts )
from .neural_blocks import ( SkipConnMLP, FourierEncoder )
from .utils import ( autograd, eikonal_loss, elev_azim_to_dir )

class Renderer(nn.Module):
  def __init__(
    self,
    shape,
    bsdf,
    light,
  ):
    super().__init__()
    self.shape = shape
    self.light = light
    self.bsdf = bsdf
  def forward(self, _rays): raise NotImplementedError()

def sample_emitter_dir_w_isect(it, shapes, lights, sampler, active=True):
  ds, spectrum = lights.sample_direction(it, sampler=sampler, active=active)

  # ds.d is already in world space
  rays = torch.cat([it.p, ds.d], dim=-1)
  not_blocked = \
    shapes.intersect_test(rays, max_t=ds.dist.reshape_as(active)[..., None], active=active)
  spectrum[~not_blocked | ~active] = 0
  return ds, spectrum

class Direct(Renderer):
  def __init__(self, **kwargs):
    super(**kwargs).__init__()
  def forward(self, rays):
    # TODO define this intersect function
    r_o, r_d = rays.split([3, 3], dim=-1)
    pts = self.shape.intersect(rays)
    spectrum, light_dir = self.light(pts)
    blocked =
    bsdf_val = self.bsdf(pts, r_d)
    raise NotImplementedError()

class Path(Renderer):
  def __init__(
    self,
    bounces:int=1,
    **kwargs,
  ):
    super(**kwargs).__init__()
    self.bounces = 1
  def forward(self, rays):
    raise NotImplementedError()
