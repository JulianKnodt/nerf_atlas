import torch
import torch.nn as nn
import torch.nn.functional as F
import random

from .neural_blocks import ( SkipConnMLP, NNEncoder, FourierEncoder )
from .utils import ( autograd, eikonal_loss, dir_to_elev_azim, rotate_vector )
import src.lights as lights

def load(args, latent_size):
  if args.space_kind == "identity": space = IdentitySpace
  elif args.space_kind == "surface": space = SurfaceSpace
  else: raise NotImplementedError()

  kwargs = {
    "latent_size": latent_size,
    "out_features": args.feature_space,
    "normal": args.normal_kind,
    "space": space(),
  }
  if args.refl_kind == "basic":
    cons = Basic
    if args.light_kind is not None: kwargs["light"] = "elaz"
  elif args.refl_kind == "rusin": cons = Rusin
  elif args.refl_kind == "view_dir" or args.refl_kind == "curr": cons = View
  elif args.refl_kind == "diffuse": cons = Diffuse
  else: raise NotImplementedError(f"refl kind: {args.refl_kind}")
  # TODO assign view, normal, lighting here?
  refl = cons(**kwargs)

  if args.light_kind is not None and refl.can_use_light:
    light = lights.load(args)
    refl = LightAndRefl(refl=refl,light=light)

  return refl

# a combination of light and reflectance models
class LightAndRefl(nn.Module):
  def __init__(self, refl, light):
    super().__init__()
    self.refl = refl
    self.light = light
    self.spectrum = None

  @property
  def can_use_normal(self): return self.refl.can_use_normal
  def forward(self, x, view=None, normal=None, light=None, latent=None, mask=None):
    if light is None: light, _spectrum = self.light(x, mask)
    return self.refl(x, view, normal, light, latent)

class SurfaceSpace(nn.Module):
  def __init__(
    self,
    act=nn.LeakyReLU(),
    final_act=nn.Identity(),
  ):
    super().__init__()
    self.encode = SkipConnMLP(
      in_size=3, out=2, activation=act,
      num_layers=5, hidden_size=64,
    )
    self.act = activation
  def forward(self, x): return self.act(self.encode(x))
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
    act=torch.sigmoid,
    latent_size:int = 0,
    out_features:int = 3,

    # delete unused
    normal=None,
    light=None,
  ):
    super().__init__()
    self.act = act
    self.latent_size = latent_size
    self.out_features = out_features
  def forward(self, x, view,normal=None,light=None,latent=None): raise NotImplementedError()
  @property
  def can_use_normal(self): return False
  @property
  def can_use_light(self): return False

def ident(x): return x
def empty(_): return None
def enc_norm_dir(kind=None):
  if kind is None: return 0, empty
  elif kind == "raw": return 3, ident
  elif kind == "elaz": return 2, dir_to_elev_azim
  else: raise NotImplementedError(f"enc_norm_dir: {kind}")

# basic reflectance takes a position and a direction and other components
class Basic(Reflectance):
  def __init__(
    self,
    space=None,

    view="elaz",
    normal=None,
    light=None,
    **kwargs,
  ):
    super().__init__(**kwargs)
    if space is None: space = IdentitySpace()
    view_dims, self.view_enc = enc_norm_dir(view)
    normal_dims, self.normal_enc = enc_norm_dir(normal)
    light_dims, self.light_enc = enc_norm_dir(light)
    in_size = view_dims + normal_dims + light_dims + space.dims
    self.mlp = SkipConnMLP(
      in_size=in_size, out=self.out_features, latent_size=self.latent_size,
      enc=FourierEncoder(input_dims=in_size),
      num_layers=5, hidden_size=128, xavier_init=True,
    )
    self.space = space

  @property
  def can_use_normal(self): return self.normal_enc != empty
  @property
  def can_use_light(self): return self.light_enc != empty

  def forward(self,x,view,normal=None,light=None,latent=None):
    x = self.space(x)
    view = self.view_enc(view)
    normal = self.normal_enc(normal)
    self.light_enc = empty
    light = self.light_enc(light)
    v = torch.cat([v for v in [x, view, normal, light] if v is not None], dim=-1)
    return self.act(self.mlp(v, latent))

# view reflectance takes a view direction and a latent vector, and nothing else.
class View(Reflectance):
  def __init__(
    self,
    space=None,

    view="elaz",
    **kwargs,
  ):
    super().__init__(**kwargs)
    view_dims, self.view_enc = enc_norm_dir(view)
    in_size = view_dims
    self.mlp = SkipConnMLP(
      in_size=in_size, out=self.out_features, latent_size=self.latent_size,
      enc=FourierEncoder(input_dims=in_size),
      num_layers=5, hidden_size=128, xavier_init=True,
    )
  def forward(self, x, view, normal=None, light=None, latent=None):
    v = self.view_enc(view)
    return self.act(self.mlp(v, latent))

class Diffuse(Reflectance):
  def __init__(
    self,
    space=None,
    **kwargs,
  ):
    super().__init__(**kwargs)
    if space is None: space = IdentitySpace()
    self.space = space

    in_size = space.dims
    self.diffuse_color = SkipConnMLP(
      in_size=in_size, out=self.out_features,
      latent_size=self.latent_size,
      enc=FourierEncoder(input_dims=in_size),
      num_layers=3, hidden_dims=64, xavier_init=True,
    )

  @property
  def can_use_normal(self): return True
  @property
  def can_use_light(self): return True

  def forward(self, x, view, normal, light):
    rgb = self.act(self.diffuse_color(self.space(x)))
    # TODO make this attentuation clamped to 0? Should be for realism.
    attenuation = (normal * light).sum(dim=-1, keepdim=True)
    return rgb * attenuation

class Rusin(Reflectance):
  def __init__(
    self,
    space=None,
    **kwargs,
  ):
    super().__init__(**kwargs)
    if space is None: space = IdentitySpace()
    in_size = 3 + space.dims
    self.mlp = SkipConnMLP(
      in_size=in_size, out=self.out_features, latent_size=self.latent_size,
      enc=NNEncoder(input_dims=in_size),
      xavier_init=True,

      num_layers=6, hidden_size=512,
    )
    self.space = space

  @property
  def can_use_normal(self): return True
  @property
  def can_use_light(self): return True

  def forward(self, x, view, normal, light, latent=None):
    frame = coordinate_system(normal)
    wo = to_local(frame, view)
    wi = to_local(frame, light)
    # have to move view and light into basis of normal
    rusin = rusin_params(wo, wi)
    x = self.space(x)
    raw = self.mlp(torch.cat([x, rusin], dim=-1), latent)
    return ((raw/2).sin()+1)/2

def nonzero_eps(v, eps: float=1e-7):
  # in theory should also be copysign of eps, but so small it doesn't matter
  # and torch.jit.script doesn't support it
  return torch.where(v.abs() < eps, torch.full_like(v, eps), v)

# assumes wo and wi are already in local coordinates
#@torch.jit.script
def rusin_params(wo, wi):
  wo = F.normalize(wo, eps=1e-6, dim=-1)
  wi = F.normalize(wi, eps=1e-6, dim=-1)
  e_1 = torch.tensor([0,1,0], device=wo.device, dtype=torch.float).expand_as(wo)
  e_2 = torch.tensor([0,0,1], device=wo.device, dtype=torch.float).expand_as(wo)

  H = F.normalize((wo + wi), eps=1e-6, dim=-1)

  cos_theta_h = H[..., 2]
  phi_h = torch.atan2(nonzero_eps(H[..., 1]), nonzero_eps(H[..., 0]))

  r = nonzero_eps(H[..., 1]).hypot(nonzero_eps(H[..., 0])).clamp(min=1e-6)
  c = (H[..., 0]/r).unsqueeze(-1)
  s = -(H[..., 1]/r).unsqueeze(-1)
  tmp = F.normalize(rotate_vector(wi, e_2, c, s), dim=-1)

  c = H[..., 2].unsqueeze(-1)
  s = -(1 - H[..., 2]).clamp(min=1e-6).sqrt().unsqueeze(-1)
  diff = F.normalize(rotate_vector(tmp, e_1, c, s), eps=1e-6, dim=-1)
  cos_theta_d = diff[..., 2]

  # rather than do `% pi/2`, take `cos` since both are cyclic but cos has better
  # properties.
  cos_phi_d = torch.atan2(nonzero_eps(diff[..., 1]), nonzero_eps(diff[..., 0])).cos()

  return torch.stack([cos_phi_d, cos_theta_h, cos_theta_d], dim=-1)

# https://github.com/mitsuba-renderer/mitsuba2/blob/main/include/mitsuba/core/vector.h#L116
# had to be significantly modified in order to add numerical stability while back-propagating.
# returns a frame to be used for normalization
#@torch.jit.script
def coordinate_system(n):
  n = F.normalize(n, eps=1e-6, dim=-1)
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
  s = F.normalize(s, eps=1e-6, dim=-1)
  t = F.normalize(s.cross(n, dim=-1), eps=1e-6, dim=-1)
  s = F.normalize(n.cross(t, dim=-1), eps=1e-6, dim=-1)
  return torch.stack([s, t, n], dim=-1)

# frame: [..., 3, 3], wo: [..., 3], return a vector of wo in the reference frame
#@torch.jit.script
def to_local(frame, wo):
  wo = wo.unsqueeze(-1)#.expand_as(frame) # TODO see if commenting out this expand_as works
  out = F.normalize((frame * wo).sum(dim=-2), eps=1e-7, dim=-1)
  return out

