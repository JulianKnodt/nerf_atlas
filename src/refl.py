import torch
import torch.nn as nn
import torch.nn.functional as F
import random

from .nerf import ( CommonNeRF, compute_pts_ts )
from .neural_blocks import ( SkipConnMLP, NNEncoder )
from .utils import ( autograd, eikonal_loss, dir_to_elev_azim )

def load(args):
  if args.space_kind == "identity":
    space = IdentitySpace()
  elif args.space_kind == "surface":
    space = SurfaceSpace()
  raise NotImplementedError()

class SurfaceSpace(nn.Module):
  def __init__(self):
    super().__init__()
    self.encode = SkipConnMLP(
      in_size=3, out=2, activation=nn.Softplus(),
      num_layers=3, hidden_size=64,
    )
    # TODO see if there can be an activation here?
  def forward(self, x): return self.encode(x)
  @property
  def dims(self): return 2

class IdentitySpace(nn.Module):
  def __init__(self): super().__init__()
  def forward(self, x): return x
  @property
  def dims(self): return 3

class Reflectance(nn.Module):
  def __init__(self): super().__init__()
  def forward(self, r_d): raise NotImplementedError()

class BasicReflectance(Reflectance):
  def __init__(
    self,
    space=None,
  ):
    super().__init__()
    if space is None: space = SurfaceSpace()
    in_size = 2 + space.dims
    self.mlp = SkipConnMLP(
      in_size = in_size, out = 3,
      enc=NNEncoder(input_dims=in_size),
      xavier_init=True,
    )
    self.space = space
  def forward(self, x, r_d):
    elaz_rd = dir_to_elev_azim(r_d)
    x = self.space(x)
    v = torch.cat([x, elaz_rd], dim=-1)
    return self.mlp(v).sigmoid()
