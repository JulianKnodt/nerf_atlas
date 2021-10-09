import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import math

from .neural_blocks import ( SkipConnMLP, NNEncoder, FourierEncoder )
from .utils import ( autograd, eikonal_loss, dir_to_elev_azim, rotate_vector, load_sigmoid )
import src.lights as lights
from .spherical_harmonics import eval_sh

refl_kinds = [
  "curr", "pos", "view", "basic", "diffuse", "rusin", "multi_rusin", "sph-har",
  # meta refl models
  "weighted",
]

def load(args, refl_kind:str, space_kind:str, latent_size:int):
  if space_kind == "identity": space = IdentitySpace
  elif space_kind == "surface": space = SurfaceSpace
  elif space_kind == "none": space = NoSpace
  else: raise NotImplementedError()

  kwargs = {
    "latent_size": latent_size,
    "act": args.sigmoid_kind,
    "out_features": args.feature_space,
    "normal": args.normal_kind,
    "space": space(),
  }
  if refl_kind == "basic":
    cons = Basic
    if args.light_kind is not None: kwargs["light"] = "elaz"
  elif refl_kind == "rusin": cons = Rusin
  elif refl_kind == "pos": cons = Positional
  elif refl_kind == "view" or args.refl_kind == "curr": cons = View
  elif refl_kind == "diffuse": cons = Diffuse
  elif refl_kind == "sph-har":
    cons = SphericalHarmonic
    kwargs["order"] = args.spherical_harmonic_order
  elif refl_kind == "weighted":
    cons = WeightedChoice
    subs = args.weighted_subrefl_kinds
    # TODO should this warn if only one subrefl is used?
    assert(len(subs) > 1), "Specifying one subrefl is pointless."
    kwargs["choices"] = [load(args, c, "none", latent_size) for c in subs]
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
  @property
  def latent_size(self): return self.refl.latent_size

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

class NoSpace(nn.Module):
  def __init__(self): super().__init__()
  def forward(self, x): torch.empty((*x.shape[:-1], 0), device=x.device, dtype=torch.float)
  @property
  def dims(self): return 0

class Reflectance(nn.Module):
  def __init__(
    self,
    act="thin",
    latent_size:int = 0,
    out_features:int = 3,
    spherical_harmonic_order:int = 0,

    # delete unused
    normal=None,
    light=None,
  ):
    super().__init__()
    self.sho = sho = spherical_harmonic_order
    self.latent_size = latent_size
    self.out_features = out_features * (sho + 1) * (sho + 1)

    self.act = load_sigmoid(act)

  def forward(self, x, view,normal=None,light=None,latent=None): raise NotImplementedError()
  @property
  def can_use_normal(self): return False
  @property
  def can_use_light(self): return False

  def sph_ham(sh_coeffs, view):
    # not spherical harmonic coefficients
    if self.sho == 0: return sh_coeffs
    return eval_sh(
      self.sho,
      sh_coeffs.reshape(sh_coeffs.shape[:-1] + (self.out_features, -1)),
      F.normalize(view, dim=-1),
    )

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

# Positional only (no view dependence)
class Positional(Reflectance):
  def __init__(
    self,
    space=None,

    **kwargs,
  ):
    super().__init__(**kwargs)
    self.mlp = SkipConnMLP(
      in_size=self.latent_size, out=self.out_features, latent_size=0,
      #enc=FourierEncoder(input_dims=3),
      num_layers=5, hidden_size=256, xavier_init=True,
    )
  def forward(self, x, view, normal=None, light=None, latent=None):
    return self.act(self.mlp(latent))

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

  def forward(self, x, view, normal, light, latent=None):
    rgb = self.act(self.diffuse_color(self.space(x)))
    # TODO make this attentuation clamped to 0? Should be for realism.
    attenuation = (normal * light).sum(dim=-1, keepdim=True)
    return rgb * attenuation

class WeightedChoice(Reflectance):
  def __init__(
    self,
    choices:[Reflectance],
    space=None,

    **kwargs,
  ):
    super().__init__(**kwargs)
    if space is None: space = IdentitySpace()
    self.space = space

    for c in choices:
      assert(issubclass(type(c), Reflectance) or isinstance(c, LightAndRefl)), \
        f"Not refl: {type(c)}"

    self.choices = nn.ModuleList(choices)
    in_size = space.dims
    self.selection = SkipConnMLP(
      in_size=in_size, out=len(choices), latent_size=self.latent_size,
      xavier_init=True, enc=FourierEncoder(input_dims=in_size),
    )

  @property
  def can_use_normal(self): return True
  @property
  def can_use_light(self): return True

  def forward(self, x, view, normal, light, latent=None):
    weights = self.selection(self.space(x), latent)
    weights = F.softmax(weights,dim=-1).unsqueeze(-2)
    subs = torch.stack([
      c(x, view, normal, light, latent) for c in self.choices
    ], dim=-1)
    return (weights * subs).sum(dim=-1)

class Rusin(Reflectance):
  def __init__(
    self,
    space=None,
    **kwargs,
  ):
    super().__init__(**kwargs)
    if space is None: space = IdentitySpace()
    _pos_size = space.dims
    rusin_size = 3
    self.rusin = SkipConnMLP(
      in_size=rusin_size, out=self.out_features, latent_size=self.latent_size,
      enc=FourierEncoder(input_dims=rusin_size),
      xavier_init=True,

      num_layers=3, hidden_size=256,
    )
    self.space = space
    # add a prior of a BSDF to the model so that it will learn a diffuse color at a specific
    # point
    self.diffuse_color = SkipConnMLP(
      in_size=3, out=self.out_features, latent_size=self.latent_size,
      enc=FourierEncoder(input_dims=rusin_size),
      xavier_init=True,

      num_layers=3, hidden_size=256,
    )

  @property
  def can_use_normal(self): return True
  @property
  def can_use_light(self): return True

  # returns the raw results given rusin parameters
  def raw(self, rusin_params, latent=None):
    return self.act(self.rusin(rusin_params.cos(), latent))

  def forward(self, x, view, normal, light, latent=None):
    # TODO would it be good to detach the normal? is it trying to fix the surface
    # to make it look better?
    frame = coordinate_system(normal.detach())
    # have to move view and light into basis of normal
    wo = to_local(frame, F.normalize(view, dim=-1))
    wi = to_local(frame, light)
    rusin = rusin_params(wo, wi)
    learned = self.act(self.rusin(rusin, latent))
    diffuse = self.act(self.diffuse_color(x, latent)) * \
      (normal * light).sum(keepdim=True, dim=-1)
    return learned + diffuse

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

  phi_d = torch.atan2(nonzero_eps(diff[..., 1]), nonzero_eps(diff[..., 0]))
  phi_d = phi_d.cos()
  #phi_d = torch.remainder(phi_d, math.pi)

  return torch.stack([phi_d, cos_theta_h, cos_theta_d], dim=-1)

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
    torch.copysign(torch.tensor(1e-6, device=z.device), s_z),
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
@torch.jit.script
def to_local(frame, wo):
  return F.normalize((frame * wo.unsqueeze(-1)).sum(dim=-2), eps=1e-7, dim=-1)

# Spherical Harmonics computes reflectance of a given viewing direction using the spherical
# harmonic basis.
class SphericalHarmonic(Reflectance):
  def __init__(
    self,
    space=None,
    order:int=2,
    view="elaz",

    **kwargs,
  ):
    super().__init__(**kwargs)
    assert(order >= 0 and order <= 4)
    in_size, self.view_enc = enc_norm_dir(view)
    self.order = order
    self.mlp = SkipConnMLP(
      in_size=in_size, out=self.out_features*((order+1)*(order+1)), latent_size=self.latent_size,
      enc=FourierEncoder(input_dims=in_size),
      num_layers=5, hidden_size=128, xavier_init=True,
    )
  def forward(self, x, view, normal=None, light=None, latent=None):
    v = self.view_enc(view)
    sh_coeffs = self.mlp(v, latent)
    rgb = eval_sh(
      self.order,
      sh_coeffs.reshape(sh_coeffs.shape[:-1] + (self.out_features, -1)),
      F.normalize(view, dim=-1),
    )
    return self.act(rgb)
