import torch
import torch.nn as nn
import torch.nn.functional as F
import random

from .neural_blocks import ( SkipConnMLP, FourierEncoder, PositionalEncoder )
from .utils import ( dir_to_elev_azim, autograd, eikonal_loss )

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

def sample_pdf(
  bins, weights, n_samples: int = 64,
):
  print(bins.shape, weights.shape)
  exit()
  # TODO test this
  device=weights.device
  weights = weights.clamp(min=1e-6)
  pdf = weights / weights.sum(dim=0, keepdim=True)
  cdf = torch.cumsum(pdf, dim=0)
  cdf = torch.cat([torch.zeros_like(cdf[:1]), cdf], dim=0)

  u = torch.rand(cdf.shape[:-1] + (num_samples,), device=device)

  inds = torch.searchsorted(cdf, u, right=True)

  below = (inds-1).clamp(min=0)
  above = inds.clamp(max=inds.shape[0]-1)
  inds_g = torch.cat([below, above], dim=-1)
  gather_shape = (*inds_g.shape[:2], cdf.shape[-1])
  cdf_g = torch.gather(cdf.unsqueeze(1).expand(gather_shape), 2, inds_g)
  bins_g = torch.gather(bins.unsqueeze(1).expand(gather_shape), 2, inds_g)

  denom = cdf_g[..., 1] - cdf_g[..., 0]
  denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
  t = (u - cdf_g[..., 0]) / denom
  samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])

  return samples

@torch.jit.script
def alpha_from_density(
  density, ts, r_d,
  shifted_softplus: bool = True,
):
  device=density.device

  if shifted_softplus: sigma_a = F.softplus(density-1)
  else: sigma_a = F.relu(density)

  end_val = torch.tensor([1e10], device=device, dtype=torch.float)
  dists = torch.cat([ts[1:] - ts[:-1], end_val], dim=-1)
  dists = dists[:, None, None, None] * torch.linalg.norm(r_d, dim=-1)
  alpha = 1 - torch.exp(-sigma_a * dists)
  weights = alpha * cumuprod_exclusive(1.0 - alpha + 1e-10)
  return alpha, weights

def fat_sigmoid(v, eps: float = 1e-3): return v.sigmoid() * (1+2*eps) - eps

# perform volumetric integration of density with some other quantity
# returns the integrated 2nd value over density at timesteps.
@torch.jit.script
def volumetric_integrate(weights, other):
  return torch.sum(weights[..., None] * other, dim=0)

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
    use_fat_sigmoid: bool =True,
    eikonal_loss: bool = False,

    with_sky_mlp: bool = False,

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
    self.feat_act = torch.sigmoid
    if use_fat_sigmoid: self.feat_act = fat_sigmoid

    self.rec_eikonal_loss = eikonal_loss
    self.eikonal_loss = 0

    self.with_sky_mlp = with_sky_mlp
    if with_sky_mlp:
      self.sky_mlp = SkipConnMLP(
        in_size=2, out=3,
        num_layers=3, hidden_size=32, device=device, xavier_init=True,
      ).to(device)

    self.record_depth = record_depth
    self.depth = None

  def forward(self, _x): raise NotImplementedError()

  def record_eikonal_loss(self, pts, density):
    self.eikonal_loss = eikonal_loss(autograd(pts, density))
    return self.eikonal_loss

  def total_latent_size(self) -> int:
    return self.mip_size() + \
      self.per_pixel_latent_size + self.instance_latent_size + self.per_pt_latent_size
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
  def sky_color(self, elev_azim_r_d):
    if not self.with_sky_mlp: return torch.zeros_like(r_d)
    # TODO use feat act here?
    return fat_sigmoid(self.sky_mlp(elev_azim_r_d))

  # produces a segmentation mask of sorts, using the alpha for occupancy.
  def acc(self): return self.alpha.max(dim=0)[0]

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
    self, out_features: int = 3,

    device="cuda",
    **kwargs,
  ):
    super().__init__(**kwargs, device=device)
    self.estim = SkipConnMLP(
      in_size=3, out=1 + out_features,
      latent_size = self.total_latent_size(),

      num_layers=6, hidden_size=80,

      xavier_init=True,

      device=device,
    ).to(device)

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
    if self.rec_eikonal_loss: self.record_eikonal_loss(pts, density)

    self.alpha, weights = alpha_from_density(density, ts, r_d)
    return volumetric_integrate(weights, self.feat_act(feats))

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
      #enc=PositionalEncoder(3, 10, N=16),
      enc=FourierEncoder(input_dims=3, device=device),

      num_layers = 6, hidden_size = 128, xavier_init=True,

      device=device,
    ).to(device)

    self.second = SkipConnMLP(
      in_size=2, out=out_features, latent_size=self.latent_size + intermediate_size,
      #enc=PositionalEncoder(2, 4, N=16),
      enc=FourierEncoder(input_dims=2, device=device),

      num_layers=5, hidden_size=64, xavier_init=True,

      device=device,
    ).to(device)

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
    if self.rec_eikonal_loss: self.record_eikonal_loss(pts, density)
    if self.training and self.noise_std > 0:
      density = density + torch.randn_like(density) * self.noise_std

    intermediate = first_out[..., 1:]

    elev_azim_r_d = dir_to_elev_azim(r_d)[None, ...].expand(pts.shape[:-1]+(2,))
    rgb = self.feat_act(self.second(
      elev_azim_r_d, torch.cat([intermediate, latent], dim=-1)
    ))

    self.alpha, weights = alpha_from_density(density, ts, r_d)
    return volumetric_integrate(weights, rgb)

# NeRF with a thin middle layer, for encoding information
class NeRFAE(CommonNeRF):
  def __init__(
    self,
    latent_size: int = 0,
    intermediate_size: int = 32,
    out_features: int = 3,

    encoding_size: int = 16,
    sigma=1<<5,

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

      device=device,
    ).to(device)

    self.density_tform = SkipConnMLP(
      # a bit hacky, but pass it in with no encodings since there are no additional inputs.
      in_size=encoding_size, out=1, latent_size=0,
      num_layers=5, hidden_size=64, xavier_init=True,

      device=device,
    ).to(device)

    self.rgb = SkipConnMLP(
      in_size=2, out=out_features, latent_size=encoding_size,

      num_layers=5, hidden_size=64, xavier_init=True,
      enc=FourierEncoder(input_dims=2, device=device),

      device=device,
    ).to(device)
    self.encoding_size = encoding_size

  def forward(self, rays):
    pts, ts, r_o, r_d = compute_pts_ts(
      rays, self.t_near, self.t_far, self.steps,
      perturb = 1 if self.training else 0,
    )
    return self.from_pts(pts, ts, r_o, r_d)

  def from_pts(self, pts, ts, r_o, r_d):
    encoded = self.compute_encoded(pts, ts, r_o, r_d)
    return self.from_encoded(encoded, ts, r_d)

  def compute_encoded(self, pts, ts, r_o, r_d):
    latent = self.curr_latent(pts.shape)

    mip_enc = self.mip_encoding(r_o, r_d, ts)

    # If there is a mip encoding, stack it with the latent encoding.
    if mip_enc is not None: latent = torch.cat([latent, mip_enc], dim=-1)

    return self.encode(pts, latent if latent.shape[-1] != 0 else None)
  def from_encoded(self, encoded, ts, r_d):
    density = self.density_tform(encoded).squeeze(-1)
    if self.rec_eikonal_loss: self.record_eikonal_loss(pts, density)
    if self.noise_std > 0 and self.training:
      density = density + torch.randn_like(density) * self.noise_std

    elev_azim_r_d = dir_to_elev_azim(r_d)[None, ...].expand(encoded.shape[:-1]+(2,))

    rgb = self.feat_act(self.rgb(elev_azim_r_d, encoded))

    self.alpha, weights = alpha_from_density(density, ts, r_d)
    return volumetric_integrate(weights, rgb)

# Dynamic NeRF for multiple frams
class DynamicNeRF(nn.Module):
  def __init__(self, canonical: CommonNeRF, device="cuda"):
    super().__init__()

    self.delta_estim = SkipConnMLP(
      # x,y,z,t -> dx, dy, dz
      in_size=4, out=3,

      num_layers = 5,
      hidden_size = 128,

      device=device,
    ).to(device)
    self.time_noise_std = 1e-2
    self.canonical = canonical.to(device)

  @property
  def nerf(self): return self.canonical
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

    dp = self.delta_estim(torch.cat([pts, t], dim=-1))
    dp = torch.where(t.abs() < 1e-6, torch.zeros_like(pts), dp)
    pts = pts + dp
    return self.canonical.from_pts(pts, ts, r_o, r_d)

# Dynamic NeRFAE for multiple framss with changing materials
class DynamicNeRFAE(nn.Module):
  def __init__(self, canonical: NeRFAE, device="cuda"):
    super().__init__()
    assert(isinstance(canonical, NeRFAE)), "Must use NeRFAE for DynamicNeRFAE"
    self.canonical = canonical.to(device)

    self.delta_estim = SkipConnMLP(
      # x,y,z,t -> dx, dy, dz
      in_size=4, out=3,
      num_layers = 5, hidden_size = 128,

      device=device,
    ).to(device)

    self.delta_enc_estim = SkipConnMLP(
      in_size=4, out=canonical.encoding_size,

      num_layers = 3, hidden_size = 128,

      device=device,
      # start assuming that there is no transformation between material type
      zero_init=True,
    )


  def forward(self, rays_t):
    rays, t = rays_t
    device=rays.device

    pts, ts, r_o, r_d = compute_pts_ts(
      rays, self.canonical.t_near, self.canonical.t_far, self.canonical.steps,
    )
    t = t[None, :, None, None, None].expand(pts.shape[:-1] + (1,))
    # compute encoding using delta positions at a given time
    pts_t = torch.cat([pts, t], dim=-1)
    dp = self.delta_estim(pts_t)
    dp = torch.where(t.abs() < 1e-6, torch.zeros_like(pts), dp)
    encoded = self.canonical.compute_encoded(pts + dp, ts, r_o, r_d)

    # compute encoding delta at a given time
    d_enc = self.delta_enc_estim(pts_t)
    d_enc = torch.where(t.abs() < 1e-6, torch.zeros_like(d_enc), d_enc)

    # TODO is this best as a sum, or is some other kind of tform better?
    return self.canonical.from_encoded(encoded + d_enc, ts, r_d)

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
  # [WIP] Multi Plane Images.
  def __init__(
    self,

    canon: CommonNeRF,

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
