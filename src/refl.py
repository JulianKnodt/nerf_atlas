import torch
import torch.nn as nn
import torch.nn.functional as F
import random

from .nerf import ( CommonNeRF, compute_pts_ts )
from .neural_blocks import ( SkipConnMLP, NNEncoder, FourierEncoder )
from .utils import ( autograd, eikonal_loss, dir_to_elev_azim, rotate_vector )

def load(args):
  if args.space_kind == "identity":
    space = IdentitySpace()
  elif args.space_kind == "surface":
    space = SurfaceSpace()

  if args.refl_kind == "basic": cons = Basic
  elif args.refl_kind == "rusin": cons = Rusin

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
  def __init__(
    self,
    activation=torch.sigmoid,
  ):
    super().__init__()
    self.act = activation
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
class Basic(Reflectance):
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
      enc=FourierEncoder(input_dims=in_size),
      num_layers=3, hidden_size=64,
      xavier_init=True,
    )
    self.space = space
  def forward(self, x, view, normal=None):
    view = self.view_enc(view)
    normal = self.normal_enc(normal)
    x = self.space(x)
    v = torch.cat([v for v in [x, view, normal] if v is not None], dim=-1)
    return self.act(self.mlp(v))


class Diffuse(Reflectance):
  def __init__(
    self,
    space=None,
  ):
    if space is None: space = IdentitySpace()
    self.space = space

    in_size = space.dims
    self.diffuse_color = SkipConnMLP(
      in_size=in_size, out=3,
      #enc=NNEncoder(input_dims=in_size),
      num_layers=3, hidden_dims=64,
      xavier_init=True,
    )
  def forward(self, x, _view, normal, light):
    rgb = self.act(self.diffuse_color(self.space(x)))
    attenuation = (normal * light).sum(dim=-1, keepdim=True)
    return rgb * attenuation

def Rusin(Reflectance):
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
    frame = coordinate_system(normal)
    wo = to_local(frame, view)
    wi = to_local(frame, light)
    # have to move view and light into basis of normal
    rusin = rusin_params(wo, wi)
    x = self.space(x)
    raw = self.mlp(torch.cat([x, rusin], dim=-1))
    return self.act(raw)

def nonzero_eps(v, eps: float=1e-7):
  # in theory should also be copysign of eps, but so small it doesn't matter
  # and torch.jit.script doesn't support it
  return torch.where(v.abs() < eps, torch.full_like(v, eps), v)

# assumes wo and wi are already in local coordinates
@torch.jit.script
def rusin_params(wo, wi):
  wo = F.normalize(wo, dim=-1)
  wi = F.normalize(wi, dim=-1)
  e_1 = torch.tensor([0,1,0], device=wo.device, dtype=torch.float).expand_as(wo)
  e_2 = torch.tensor([0,0,1], device=wo.device, dtype=torch.float).expand_as(wo)

  H = F.normalize((wo + wi), dim=-1)

  cos_theta_h = H[..., 2]
  phi_h = torch.atan2(nonzero_eps(H[..., 1]), nonzero_eps(H[..., 0]))

  # v = -phi_h.unsqueeze(-1)
  r = nonzero_eps(H[..., 1]).hypot(nonzero_eps(H[..., 0])).clamp(min=1e-6)
  c = (H[..., 0]/r).unsqueeze(-1)
  s = -(H[..., 1]/r).unsqueeze(-1)
  tmp = F.normalize(rotate_vector(wi, e_2, c, s), dim=-1)
  #v = -theta_h.unsqueeze(-1)
  c = H[..., 2].unsqueeze(-1)
  s = -(1 - H[..., 2]).clamp(min=1e-6).sqrt().unsqueeze(-1)
  diff = F.normalize(rotate_vector(tmp, e_1, c, s), dim=-1)
  cos_theta_d = diff[..., 2]
  # rather than doing fmod, try cos to see if it can capture cyclicity better.
  cos_phi_d = torch.atan2(nonzero_eps(diff[..., 1]), nonzero_eps(diff[..., 0])).cos()

  return torch.stack([cos_phi_d, cos_theta_h, cos_theta_d], dim=-1)

# https://github.com/mitsuba-renderer/mitsuba2/blob/main/include/mitsuba/core/vector.h#L116
# had to be significantly modified in order to add numerical stability while back-propagating.
# returns a frame to be used for normalization
@torch.jit.script
def coordinate_system(n):
  n = F.normalize(n, eps=1e-7, dim=-1)
  x, y, z = n.split(1, dim=-1)
  sign = torch.where(z >= 0, 1., -1.)
  s_z = sign + z
  a = -torch.where(
    s_z.abs() < 1e-6,
    torch.tensor(1e-6, device=z.device),
    s_z,
  ).reciprocal()
  b = x * y * a

  s = torch.cat([
    (x * x * a * sign) + 1, b * sign, x * -sign,
  ], dim=-1)
  s = F.normalize(s, eps=1e-7, dim=-1)
  t = F.normalize(s.cross(n, dim=-1), eps=1e-7, dim=-1)
  s = F.normalize(n.cross(t, dim=-1), eps=1e-7, dim=-1)
  return torch.stack([s, t, n], dim=-1)

# frame: [..., 3, 3], wo: [..., 3], return a vector of wo in the reference frame
@torch.jit.script
def to_local(frame, wo):
  wo = wo.unsqueeze(-1)#.expand_as(frame) # TODO see if commenting out this expand_as works
  out = F.normalize((frame * wo).sum(dim=-2), eps=1e-7, dim=-1)
  return out

