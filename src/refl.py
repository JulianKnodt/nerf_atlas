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

  if args.refl_kind == "basic": cons = BasicReflectance
  elif args.refl_kind == "rusin": cons = RusinReflectance

  # TODO assign view, normal, lighting here?
  return cons(space=space)

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

def ident(x): return x
def empty(_): return None
def enc_norm_dir(kind=None):
  if kind is None: return 0, empty
  elif kind == "raw": return 3, ident
  elif kind == "elaz": return 2, dir_to_elev_azim
  else:
    raise NotImplementedError()

# basic reflectance takes a position and a direction
class BasicReflectance(Reflectance):
  def __init__(
    self,
    space=None,

    view="elaz",
    normal=None,
  ):
    super().__init__()
    if space is None: space = IdentitySpace()
    view_dims, self.view_enc = enc_norm_dir(view)
    normal_dims, self.normal_enc = enc_norm_dir(normal)
    in_size = view_dims + normal_dims + space.dims
    self.mlp = SkipConnMLP(
      in_size = in_size, out = 3,
      enc=NNEncoder(input_dims=in_size),
      xavier_init=True,
    )
    self.space = space
  def forward(self, x, view, normal=None):
    view = self.view_enc(view)
    normal = self.normal_enc(normal)
    x = self.space(x)
    v = torch.cat([v for v in [x, view, normal] if v is not None], dim=-1)
    return self.mlp(v).sigmoid()

def RusinReflectance(Reflectance):
  def __init__(
    self,
    space=None,
  ):
    super().__init__()
    if space is None: space = IdentitySpace()
    in_size = 3 + space.dims
    self.mlp = SkipConnMLP(
      in_size = in_size, out=3,
      enc=NNEncoder(input_dims=in_size),
      xavier_init=True,
    )
    self.space = space
  def forward(self, x, view, normal, light):
    rusin = rusin_params(view, normal, light)
    x = self.space(x)
    return self.mlp(torch.cat([x, rusin], dim=-1)).sigmoid()
