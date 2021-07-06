import torch
import torch.nn as nn
import torch.nn.functional as F
import random

from .neural_blocks import (
  SkipConnMLP, UpdateOperator, FourierEncoder, PositionalEncoder, NNEncoder
)
from .utils import ( dir_to_elev_azim, autograd )
import src.refl as refl
from .renderers import ( lighting_w_isect )

@torch.jit.script
def cumuprod_exclusive(t):
  cp = torch.cumprod(t, dim=0)
  cp = torch.roll(cp, 1, dims=0)
  cp[0, ...] = 1.0
  return cp

#@torch.jit.script # cannot jit script cause of tensordot :)
def compute_pts_ts(
  rays, near, far, steps, lindisp=False,
  perturb: float = 0,
):
  r_o, r_d = rays.split([3,3], dim=-1)
  device = r_o.device
  if lindisp:
    t_vals = torch.linspace(0, 1, steps, device=device, dtype=r_o.dtype)
    ts = 1/(1/max(near, 1e-10) * (1-t_vals) + 1/far * (t_vals))
  else:
    ts = torch.linspace(near, far, steps=steps, device=device, dtype=r_o.dtype)

  if perturb > 0:
    mids = 0.5 * (ts[:-1] + ts[1:])
    lower = torch.cat([mids, ts[-1:]])
    upper = torch.cat([ts[:1], mids])
    rand = torch.rand_like(lower) * perturb
    ts = lower + (upper - lower) * rand
  pts = r_o.unsqueeze(0) + torch.tensordot(ts, r_d, dims = 0)
  return pts, ts, r_o, r_d

# given a set of densities, and distances between the densities,
# compute alphas from them.
@torch.jit.script
def alpha_from_density(
  density, ts, r_d,
  softplus: bool = True,
):
  device=density.device

  if softplus: sigma_a = F.softplus(density-1)
  else: sigma_a = F.relu(density)

  end_val = torch.full_like(ts[..., :1], 1e10)
  dists = torch.cat([ts[..., 1:] - ts[..., :-1], end_val], dim=-1)
  while len(dists.shape) < 4: dists = dists[..., None]
  dists = dists * torch.linalg.norm(r_d, dim=-1)
  alpha = 1 - torch.exp(-sigma_a * dists)
  weights = alpha * cumuprod_exclusive(1.0 - alpha + 1e-10)
  return alpha, weights

# TODO probably move sigmoids to utils

# sigmoids which shrink or expand the total range to prevent gradient vanishing,
# or prevent it from representing full density items.
# fat sigmoid has no vanishing gradient, but thin sigmoid leads to better outlines.
def fat_sigmoid(v, eps: float = 1e-3): return v.sigmoid() * (1+2*eps) - eps
def thin_sigmoid(v, eps: float = 1e-2): return fat_sigmoid(v, -eps)
def cyclic_sigmoid(v, eps:float=-1e-2,period:int=5):
  return ((v/period).sin()+1)/2 * (1+2*eps) - eps

# perform volumetric integration of density with some other quantity
# returns the integrated 2nd value over density at timesteps.
@torch.jit.script
def volumetric_integrate(weights, other):
  return torch.sum(weights[..., None] * other, dim=0)


# bg functions, need to be here for pickling
def black(_elaz_r_d, _weights): return 0
def white(_, weights): 1-weights.sum(dim=0).unsqueeze(-1)
# having a random color will probably help prevent any background
def random_color(_elaz_r_d, weights):
  # TODO need to think this through more
  # This will make it so that there never is a background.
  summed = (1-weights.sum(dim=0).unsqueeze(-1))
  return torch.rand_like(summed) * summed

class CommonNeRF(nn.Module):
  def __init__(
    self,

    steps: int = 64,

    #out_features: int = 3, # 3 is for RGB
    t_near: float = 0,
    t_far: float = 1,
    density_std: float = 0.01,
    noise_std: int = 1e-2,
    mip = None,
    instance_latent_size: int = 0,
    per_pixel_latent_size: int = 0,
    per_point_latent_size: int = 0,
    unisurf_loss: bool = False,

    sigmoid_kind: str = "thin",
    bg: str = "black",

    record_depth: bool = False,

    device="cuda",
  ):
    super().__init__()
    self.empty_latent = torch.zeros(1,1,1,1,0, device=device, dtype=torch.float)

    self.t_near = t_near
    self.t_far = t_far
    self.steps = steps
    self.mip = mip

    self.per_pixel_latent_size = per_pixel_latent_size
    self.per_pixel_latent = None

    self.instance_latent_size = instance_latent_size
    self.instance_latent = None

    self.per_pt_latent_size = per_point_latent_size
    self.per_pt_latent = None

    self.alpha = None
    self.noise_std = 0.2
    # TODO add activation for using sigmoid or fat sigmoid

    self.rec_unisurf_loss = unisurf_loss
    self.unisurf_loss = 0

    self.set_bg(bg)
    self.set_sigmoid(sigmoid_kind)

    self.record_depth = record_depth
    self.depth = None

  def forward(self, _x): raise NotImplementedError()
  def set_bg(self, bg="black"):
    if bg == "black":
      self.sky_color = black
    elif bg == "white":
      self.sky_color = white
    elif bg == "mlp":
      self.sky_mlp = SkipConnMLP(
        in_size=2, out=3, enc=NNEncoder(in_size=2,out=3),
        num_layers=3, hidden_size=32, device=device, xavier_init=True,
      )
      self.sky_color = self.sky_from_mlp
    elif bg == "random":
      self.sky_color = random_color
    else:
      raise NotImplementedError(f"Unexpected bg: {bg}")

  def set_sigmoid(self, kind="thin"):
    if kind == "thin": self.feat_act = thin_sigmoid
    elif kind == "fat": self.feat_act = fat_sigmoid
    elif kind == "normal": self.feat_act = torch.sigmoid
    elif kind == "cyclic": self.feat_act = cyclic_sigmoid
    elif kind == "softmax": self.feat_act = nn.Softmax(dim=-1)
    else: raise NotImplementedError(f"Unknown sigmoid kind({kind})")
  def sky_from_mlp(self, elaz_r_d, weights):
    return (1-weights.sum(dim=0)).unsqueeze(-1) * fat_sigmoid(self.sky_mlp(elaz_r_d))
  def record_unisurf_loss(self, pts, alpha):
    if not self.rec_unisurf_loss or not self.training: return
    alpha = alpha.unsqueeze(-1)
    normals, = torch.autograd.grad(
      inputs=pts, outputs=alpha, grad_outputs=torch.ones_like(alpha), create_graph=True
    )
    self.unisurf_loss, = torch.autograd.grad(
      inputs=normals, outputs=alpha, grad_outputs=torch.ones_like(alpha),
      only_inputs=True, allow_unused=True,
    )
    self.unisurf_loss = self.unisurf_loss or 0
    return self.unisurf_loss

  def total_latent_size(self) -> int:
    return self.mip_size() + \
      self.per_pixel_latent_size + \
      self.instance_latent_size + \
      self.per_pt_latent_size
  def set_per_pt_latent(self, latent):
    assert(latent.shape[-1] == self.per_pt_latent_size), \
      f"expected latent in [T, B, H, W, L={self.per_pixel_latent_size}], got {latent.shape}"
    assert(len(latent.shape) == 5), \
      f"expected latent in [T, B, H, W, L], got {latent.shape}"
    self.per_pt_latent = latent
  def set_per_pixel_latent(self, latent):
    assert(latent.shape[-1] == self.per_pixel_latent_size), \
      f"expected latent in [B, H, W, L={self.per_pixel_latent_size}], got {latent.shape}"
    assert(len(latent.shape) == 4), \
      f"expected latent in [B, H, W, L], got {latent.shape}"
    self.per_pixel_latent = latent
  def set_instance_latent(self, latent):
    assert(latent.shape[-1] == self.instance_latent_size), "expected latent in [B, L]"
    assert(len(latent.shape) == 2), "expected latent in [B, L]"
    self.instance_latent = latent

  # produces a segmentation mask of sorts, using the alpha for occupancy.
  def acc(self): return self.alpha.max(dim=0)[0]
  def acc_smooth(self): return torch.sum(self.weights,dim=0).unsqueeze(-1)
  def set_refl(self, refl):
    if hasattr(self, "refl"): self.refl = refl

  def depths(self, depths):
    with torch.no_grad():
      print(self.alpha.shape, depths.shape)
      exit()
      return volumetric_integrate(self.alpha, depths[..., None, None, None])

  @property
  def nerf(self): return self

  def mip_size(self): return 0 if self.mip is None else self.mip.size() * 6
  def mip_encoding(self, r_o, r_d, ts):
    if self.mip is None: return None
    end_val = torch.tensor([1e10], device=ts.device, dtype=ts.dtype)
    ts = torch.cat([ts, end_val], dim=-1)
    return self.mip(r_o, r_d, ts[..., :-1], ts[..., 1:])

  # gets the current latent vector for this NeRF instance
  def curr_latent(self, pts_shape) -> ["T", "B", "H", "W", "L_pp + L_inst"]:
    curr = self.empty_latent.expand(pts_shape[:-1] + (0,)) if self.per_pt_latent is None \
      else self.per_pt_latent

    if self.per_pixel_latent is not None:
      ppl = self.per_pixel_latent[None, ...].expand(pts.shape[:-1] + (-1,))
      curr = torch.cat([curr, ppl], dim=-1)

    if self.instance_latent is not None:
      il = self.instance_latent[None, :, None, None, :].expand_as(pts.shape[:-1] + (-1,))
      curr = torch.cat([curr, il], dim=-1)

    return curr

class TinyNeRF(CommonNeRF):
  # No frills, single MLP NeRF
  def __init__(
    self,
    out_features: int = 3,
    device="cuda",
    **kwargs,
  ):
    super().__init__(**kwargs, device=device)
    self.estim = SkipConnMLP(
      in_size=3, out=1 + out_features,
      latent_size = self.total_latent_size(),
      num_layers=6, hidden_size=128,

      xavier_init=True,
    )

  def forward(self, rays):
    pts, ts, r_o, r_d = compute_pts_ts(
      rays, self.t_near, self.t_far, self.steps,
      perturb = 1 if self.training else 0,
    )
    return self.from_pts(pts, ts, r_o, r_d)

  def from_pts(self, pts, ts, r_o, r_d):
    latent = self.curr_latent(pts.shape)
    mip_enc = self.mip_encoding(r_o, r_d, ts)
    if mip_enc is not None: latent = torch.cat([latent, mip_enc], dim=-1)

    density, feats = self.estim(pts, latent).split([1, 3], dim=-1)

    self.alpha, self.weights = alpha_from_density(density, ts, r_d)
    return volumetric_integrate(self.weights, self.feat_act(feats)) + \
      self.sky_color(None, self.weights)

# A plain old nerf
class PlainNeRF(CommonNeRF):
  def __init__(
    self,
    intermediate_size: int = 32,
    out_features: int = 3,

    device: torch.device = "cuda",

    **kwargs,
  ):
    super().__init__(**kwargs, device=device)
    self.latent_size = self.total_latent_size()

    self.first = SkipConnMLP(
      in_size=3, out=1 + intermediate_size, latent_size=self.latent_size,
      enc=FourierEncoder(input_dims=3, device=device),

      num_layers = 6, hidden_size = 128, xavier_init=True,
    )

    self.refl = refl.View(
      out_features=out_features, act=self.feat_act,
      latent_size=self.latent_size+intermediate_size,
    )

  def forward(self, rays):
    pts, ts, r_o, r_d = compute_pts_ts(
      rays, self.t_near, self.t_far, self.steps, perturb = 1 if self.training else 0,
    )
    return self.from_pts(pts, ts, r_o, r_d)

  def from_pts(self, pts, ts, r_o, r_d):
    latent = self.curr_latent(pts.shape)

    mip_enc = self.mip_encoding(r_o, r_d, ts)

    # If there is a mip encoding, stack it with the latent encoding.
    if mip_enc is not None: latent = torch.cat([latent, mip_enc], dim=-1)

    first_out = self.first(pts, latent if latent.shape[-1] != 0 else None)

    density = first_out[..., 0]
    if self.training and self.noise_std > 0:
      density = density + torch.randn_like(density) * self.noise_std

    intermediate = first_out[..., 1:]

    #n = None
    #if self.refl.can_use_normal: n = autograd(pts, density)

    view = r_d[None, ...].expand_as(pts)
    rgb = self.refl(
      x=pts, view=view,
      latent=torch.cat([latent, intermediate], dim=-1),
    )

    self.alpha, self.weights = alpha_from_density(density, ts, r_d)
    return volumetric_integrate(self.weights, rgb) + self.sky_color(view, self.weights)

# NeRF with a thin middle layer, for encoding information
class NeRFAE(CommonNeRF):
  def __init__(
    self,
    intermediate_size: int = 32,
    out_features: int = 3,

    encoding_size: int = 32,
    normalize_latent: bool = True,

    device="cuda",
    **kwargs,
  ):
    super().__init__(**kwargs, device=device)

    self.latent_size = self.total_latent_size()

    self.encode = SkipConnMLP(
      in_size=3, out=encoding_size,
      latent_size=self.latent_size,
      num_layers=5, hidden_size=128,
      enc=FourierEncoder(input_dims=3, device=device),
      xavier_init=True,
    )

    self.density_tform = SkipConnMLP(
      in_size=encoding_size, out=1+intermediate_size, latent_size=0,
      num_layers=5, hidden_size=64, xavier_init=True,
    )

    self.refl = refl.View(
      out_features=out_features, act=self.feat_act,
      latent_size=encoding_size+intermediate_size,
    )
    self.encoding_size = encoding_size
    self.regularize_latent = False
    self.normalize_latent = normalize_latent

  def set_regularize_latent(self):
    self.regularize_latent = True
    self.latent_l2_loss = 0
  def forward(self, rays):
    pts, ts, r_o, r_d = compute_pts_ts(
      rays, self.t_near, self.t_far, self.steps,
      perturb = 1 if self.training else 0,
    )
    return self.from_pts(pts, ts, r_o, r_d)

  def from_pts(self, pts, ts, r_o, r_d):
    encoded = self.compute_encoded(pts, ts, r_o, r_d)
    if self.regularize_latent:
      self.latent_l2_loss = torch.linalg.norm(encoded, dim=-1).square().mean()
    return self.from_encoded(encoded, ts, r_d, pts)

  def compute_encoded(self, pts, ts, r_o, r_d):
    latent = self.curr_latent(pts.shape)

    mip_enc = self.mip_encoding(r_o, r_d, ts)

    # If there is a mip encoding, stack it with the latent encoding.
    if mip_enc is not None: latent = torch.cat([latent, mip_enc], dim=-1)

    return self.encode(pts, latent if latent.shape[-1] != 0 else None)
  def from_encoded(self, encoded, ts, r_d, pts):
    if self.normalize_latent: encoded = F.normalize(encoded, dim=-1)

    first_out = self.density_tform(encoded)
    density = first_out[..., 0]
    intermediate = first_out[..., 1:]

    if self.training and self.noise_std > 0:
      density = density + torch.randn_like(density) * self.noise_std

    rgb = self.refl(
      x=pts, view=r_d[None,...].expand_as(pts),
      latent=torch.cat([encoded,intermediate],dim=-1),
    )

    self.alpha, self.weights = alpha_from_density(density, ts, r_d)
    self.record_unisurf_loss(pts, self.alpha)

    color = volumetric_integrate(self.weights, rgb)
    sky = self.sky_color(elev_azim_r_d, self.weights)
    return color + sky


# https://arxiv.org/pdf/2106.12052.pdf
class VolSDF(CommonNeRF):
  def __init__(
    self, sdf,
    # how many features to pass from density to RGB
    intermediate_size: int = 32, out_features: int = 3,
    device: torch.device = "cuda",

    direct_lighting=False,
    **kwargs,
  ):
    super().__init__(**kwargs, device=device)
    self.sdf = sdf
    # the reflectance model is in the SDF, so don't encode it here.
    # TODO add a bounding radius here for preventing spheres reaching infinite length?
    self.scale = nn.Parameter(torch.tensor(0.1, requires_grad=True, device=device))
    self.secondary = None
    if direct_lighting:
      assert(isinstance(self.sdf.refl, refl.LightAndRefl)), \
        "Must use light w/ direct illumination"
      self.secondary = self.direct
  def direct(self, pts, view, normal):
    light = self.sdf.refl.light
    light_dir, light_val = lighting_w_isect(pts, light, self.sdf.intersect_mask)
    bsdf_val = self.sdf.refl(x=pts,view=view, normal=normal, light=light_dir)
    return bsdf_val * light_val
  def forward(self, rays):
    pts, ts, r_o, r_d = compute_pts_ts(
      rays, self.t_near, self.t_far, self.steps, perturb = 1 if self.training else 0,
    )
    return self.from_pts(pts, ts, r_o, r_d)
  def set_refl(self, refl): self.sdf.refl = refl
  def from_pts(self, pts, ts, r_o, r_d):
    latent = self.curr_latent(pts.shape)
    mip_enc = self.mip_encoding(r_o, r_d, ts)
    if mip_enc is not None: latent = torch.cat([latent, mip_enc], dim=-1)

    sdf_vals, latent = self.sdf.from_pts(pts)
    density = 1/self.scale * self.laplace_cdf(-sdf_vals)
    self.alpha, self.weights = alpha_from_density(density, ts, r_d, softplus=False)

    n = None
    if self.sdf.refl.can_use_normal or self.secondary is not None:
      n = self.sdf.normals(pts)

    view = r_d.unsqueeze(0).expand_as(pts)
    if self.secondary is None:rgb = self.sdf.refl(x=pts, view=view, normal=n, latent=latent)
    else: rgb = self.secondary(pts, view, n)

    return volumetric_integrate(self.weights, rgb)

  def laplace_cdf(self, sdf_vals):
    scaled = sdf_vals/self.scale
    return torch.where(
      scaled <= 0,
      # clamps are necessary to prevent NaNs, even though the values should get filtered out
      # later. They should be noops.
      scaled.clamp(max=0).exp()/2,
      1 - scaled.clamp(min=0).neg().exp()/2
    )



# Dynamic NeRF for multiple frams
class DynamicNeRF(nn.Module):
  def __init__(self, canonical: CommonNeRF, gru_flow:bool=False, device="cuda"):
    super().__init__()
    self.canonical = canonical.to(device)

    if gru_flow:
      self.delta_estim = UpdateOperator(in_size=4, out_size=3, hidden_size=32)
    else:
      self.delta_estim = SkipConnMLP(
        # x,y,z,t -> dx, dy, dz
        in_size=4, out=3,

        num_layers = 5, hidden_size = 128,
        enc=NNEncoder(input_dims=4, device=device),
        activation=nn.Softplus(),
        zero_init=True,
      )
    self.time_noise_std = 1e-2
    self.smooth_delta = False
    self.delta_smoothness = 0

  @property
  def nerf(self): return self.canonical
  def set_smooth_delta(self): setattr(self, "smooth_delta", True)
  def forward(self, rays_t):
    rays, t = rays_t
    device=rays.device
    pts, ts, r_o, r_d = compute_pts_ts(
      rays, self.canonical.t_near, self.canonical.t_far, self.canonical.steps,
      perturb = 1 if self.training else 0,
    )
    # small deviation for regularization
    if self.training and self.time_noise_std > 0:
      t = t + self.time_noise_std * torch.randn_like(t)

    t = t[None, :, None, None, None].expand(pts.shape[:-1] + (1,))

    pts_t = torch.cat([pts, t], dim=-1)
    dp = self.delta_estim(pts_t)
    dp = torch.where(t.abs() < 1e-6, torch.zeros_like(pts), dp)
    if self.training and self.smooth_delta:
      self.delta_smoothness = self.delta_estim.l2_smoothness(pts_t, dp)
    pts = pts + dp
    return self.canonical.from_pts(pts, ts, r_o, r_d)

# Dynamic NeRFAE for multiple framss with changing materials
class DynamicNeRFAE(nn.Module):
  def __init__(self, canonical: NeRFAE, gru_flow: bool=False, device="cuda"):
    super().__init__()
    assert(isinstance(canonical, NeRFAE)), "Must use NeRFAE for DynamicNeRFAE"
    self.canon = canonical.to(device)

    self.delta_estim = SkipConnMLP(
      # x,y,z,t -> dx, dy, dz
      in_size=4, out=3 + canonical.encoding_size,
      num_layers = 6, hidden_size = 128,
      enc=NNEncoder(input_dims=4, device=device),

      activation=nn.Softplus(), zero_init=True,
    )

    self.smooth_delta = False
    self.tf_smoothness = 0
    self.time_noise_std = 1e-3

  @property
  def nerf(self): return self.canon
  def set_smooth_delta(self): setattr(self, "smooth_delta", True)
  def forward(self, rays_t):
    rays, t = rays_t
    device=rays.device

    pts, ts, r_o, r_d = compute_pts_ts(
      rays, self.canon.t_near, self.canon.t_far, self.canon.steps,
    )
    if self.canon.rec_unisurf_loss: pts.requires_grad_()
    # small deviation for regularization
    if self.training and self.time_noise_std > 0: t = t + self.time_noise_std * torch.randn_like(t)
    t = t[None, :, None, None, None].expand(pts.shape[:-1] + (1,))
    # compute encoding using delta positions at a given time
    pts_t = torch.cat([pts, t], dim=-1)
    delta = self.delta_estim(pts_t)
    #delta = torch.where(t.abs() < 1e-6, torch.zeros_like(delta), delta)
    dp, d_enc = delta.split([3, self.canon.encoding_size], dim=-1)
    encoded = self.canon.compute_encoded(pts + dp, ts, r_o, r_d)

    # TODO is this best as a sum, or is some other kind of tform better?
    return self.canon.from_encoded(encoded + d_enc, ts, r_d, pts)


# unisurf, currently doesn't really work because no secant search
class Unisurf(CommonNeRF):
  def __init__(
    self,
    intermediate_size: int = 32, out_features: int = 3,
    device: torch.device = "cuda",
    **kwargs,
  ):
    super().__init__(**kwargs, device=device)
    self.latent_size = self.total_latent_size()

    self.alpha_pred = SkipConnMLP(
      in_size=3, out=1 + intermediate_size, latent_size=self.latent_size,
      enc=NNEncoder(input_dims=3, device=device),
      num_layers = 6, hidden_size = 128, xavier_init=True,
    )

    self.refl = refl.View(
      out_features=out_features, act=self.feat_act,
      latent_size=self.latent_size+intermediate_size,
    )

  def forward(self, rays):
    pts, ts, r_o, r_d = compute_pts_ts(
      rays, self.t_near, self.t_far, self.steps, perturb = 1 if self.training else 0,
    )
    return self.from_pts(pts, ts, r_o, r_d)
  def from_pts(self, pts, ts, r_o, r_d):
    latent = self.curr_latent(pts.shape)
    mip_enc = self.mip_encoding(r_o, r_d, ts)
    if mip_enc is not None: latent = torch.cat([latent, mip_enc], dim=-1)

    first_out = self.alpha_pred(pts, latent if latent.shape[-1] != 0 else None)

    self.alpha = fat_sigmoid(first_out[..., 0])
    intermediate = first_out[..., 1:]

    rgb = self.refl(
      x=pts, view=r_d.squeeze(0).expand_as(pts),
      latent=torch.cat([intermediate, latent], dim=-1),
    )
    self.weights = self.alpha * cumuprod_exclusive(1.0 - self.alpha + 1e-10)
    return volumetric_integrate(self.weights, rgb)

class SinglePixelNeRF(nn.Module):
  def __init__(
    self,
    canon: CommonNeRF,
    encoder,
    img,

    device: torch.device = "cuda",
  ):
    super().__init__()
    self.canon = canon
    self.encoder = encoder
    # encode image
    self.encoder(img)

    self.device = device

  @property
  def nerf(self): return self.canon
  def forward(self, rays_uvs):
    rays, uvs = rays_uvs
    latent = self.encoder.sample(uvs)
    self.canon.set_per_pixel_latent(latent)
    return self.canon(rays)

class MPI(nn.Module):
  # Multi Plane Imaging.
  def __init__(
    self,
    canonical: CommonNeRF,

    position = [0,0,0],
    normal = [0,0,-1],
    delta=0.1,

    n_planes: int = 6,

    device="cuda",
  ):
    super().__init__()

    self.n_planes = torch.linspace(canon.t_near, canon.t_far, steps=n_planes, device=device)
    self.position = torch.tensor(position, device=device, dtype=torch.float)
    self.normal = torch.tensor(normal, device=device, dtype=torch.float)
    self.delta = delta

    self.canonical = canonical.to(device)
  def forward(self, rays):
    r_o, r_d = rays.split([3,3], dim=-1)
    device = r_o.device

    n = self.normal.expand_as(r_d)
    denom = (n * r_d).sum(dim=-1, keepdim=True)
    centers = self.position.unsqueeze(0) + torch.tensordot(
      self.delta * torch.arange(self.n_planes, device=device, dtype=torch.float),
      -self.normal, dims=0,
    )
    ts = ((centers - r_o) * n).sum(dim=-1, keepdim=True)/denom
    # if denom is too small it will have numerical instability because it's near parallel.
    hits = torch.where(denom.abs() > 1e-3, ts, torch.zeros_like(denom))
    pts = r_o.unsqueeze(0) + r_d.unsqueeze(0) * hits

    return self.canonical.from_pts(pts, ts, r_o, r_d)

  @property
  def nerf(self): return self.canon
  def from_pts(self, pts, ts, r_o, r_d):
    density, feats = self.estim(pts).split([1, 3], dim=-1)

    alpha, weights = alpha_from_density(density, ts, r_d)
    return volumetric_integrate(weights, self.feat_act(feats))


# TODO test this as well
def metropolis_sampling(
  density_estimator,
  ts_init, r_o, r_d,
  iters: int = 6,
):
  # need to make this the shape of r_d exit with last dim 1
  curr = ts_init
  print(r_o.shape)
  exit()
  with torch.no_grad():
    candidates = torch.rand_like(curr) + curr
    curr_density = density_estimator(candidates)
    for i in range(iters):
      candidates = torch.randn_like(curr) + curr
      density = density_estimator(candidates)
      acceptance = density/curr_density
      alphas = torch.rand_like(density)
      mask = acceptance <= alphas
      curr = torch.where(mask, candidates, curr)
      curr_density = torch.where(mask, density, curr_density)
  return curr, r_o + curr * r_d

# TODO need to test this more, doesn't seem to work that well
def inverse_sample(
  density_estimator,
  pts, ts, r_o, r_d,
):
  with torch.no_grad():
    _, weights = alpha_from_density(density_estimator(pts.squeeze(-1)), ts, r_d)
    weights = weights.clamp(min=1e-10)
    pdf = weights/weights.sum(dim=0,keepdim=True)
    cdf = torch.cumsum(pdf, dim=0)
    N = ts.shape[0]
    # XXX this only works because we assume that the number of samples (N) is the same.
    #u = torch.rand_like(cdf)
    u = torch.linspace(0, 1, N, device=cdf.device)\
      [..., None, None, None].expand_as(cdf)
    # XXX this operates on innermost dimension, so need to do this transpose
    inds = torch.searchsorted(
      cdf.transpose(0, -1).contiguous(), u.transpose(0, -1).contiguous(), right=True
    ).transpose(0, -1)
    below = (inds - 1).clamp(min=0)
    above = inds.clamp(max=N-1)
    inds_g = torch.stack([below, above], dim=-1)

    # TODO what is the right dimension to add here?
    cdf_g = torch.gather(cdf.unsqueeze(1).expand_as(inds_g), 0, inds_g)
    bins_g = torch.gather(ts[:, None, None, None, None].expand_as(inds_g), 0, inds_g)

    denom = cdf_g[..., 1] - cdf_g[..., 0]
    denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
    t = (u - cdf_g[..., 0]) / denom
    samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])
    new_pts = r_o + samples.unsqueeze(-1) * r_d
  return samples, new_pts
