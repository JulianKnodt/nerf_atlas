import torch
import torch.nn as nn
import torch.nn.functional as F
import random

from .neural_blocks import ( SkipConnMLP )
from .utils import dir_to_elev_azim

@torch.jit.script
def cumuprod_exclusive(t):
  cp = torch.cumprod(t, dim=0)
  cp = torch.roll(cp, 1, dims=0)
  cp[0, ...] = 1
  return cp

def compute_pts_ts(
  rays, near, far, steps, with_noise=False, lindisp=False,
  perturb: float = 1.,
):
  r_o, r_d = rays.split([3,3], dim=-1)
  device = r_o.device
  # TODO add step jitter: self.steps + random.randint(0,5),
  t_vals = torch.linspace(0, 1, steps, device=device, dtype=r_o.dtype)
  if lindisp:
    ts = 1./(1./max(near, 1e-10) * (1.-t_vals) + 1./far * (t_vals))
  else:
    ts = near * (1.-t_vals) + far * (t_vals)
  if perturb > 0.:
    mids = 0.5 * (ts[:-1] + ts[1:])
    lower = torch.cat([mids, ts[-1:]])
    upper = torch.cat([ts[:1], mids])
    rand = torch.rand_like(lower) * perturb
    ts = lower + (upper - lower) * rand
  pts = r_o.unsqueeze(0) + torch.tensordot(ts, r_d, dims = 0)
  return pts, ts, r_o, r_d

# perform volumetric integration of density with some other quantity
@torch.jit.script
def volumetric_integrate(density, other, ts):
  device=density.device

  sigma_a = F.relu(density)

  end_val = torch.tensor([1e10], device=device, dtype=torch.float)
  dists = torch.cat([ts[1:] - ts[:-1], end_val], dim=-1)
  dists = dists[:, None, None, None]

  alpha = 1.0 - torch.exp(-sigma_a * dists)
  weights = alpha * cumuprod_exclusive((1.0 - alpha).clamp(min=1e-10))
  return (weights[..., None] * other).sum(dim=0)

class CommonNeRF(nn.Module):
  def __init__(
    self,

    steps: int = 64,
    #out_features: int = 3, # 3 is for RGB
    t_near: float = 0,
    t_far: float = 1,
    density_std: float = 0.01,
    mip = None,
  ):
    super().__init__()
    self.t_near = t_near
    self.t_far = t_far
    self.steps = steps
    self.mip = mip
  def forward(self, x): raise NotImplementedError()
  def mip_size(self): return 0 if self.mip is None else self.mip.size() * 6
  def mip_encoding(self, r_o, r_d, ts):
    if self.mip is None: return None
    end_val = torch.tensor([1e10], device=ts.device, dtype=ts.dtype)
    ts = torch.cat([ts, end_val], dim=-1)
    t0 = ts[..., :-1]
    t1 = ts[..., 1:]
    return self.mip(r_o, r_d, t0, t1)


class TinyNeRF(CommonNeRF):
  def __init__(
    self,
    out_features: int = 3,

    device="cuda",
    **kwargs,
  ):
    super().__init__(**kwargs)
    self.estim = SkipConnMLP(
      in_size=3, out=1 + out_features,

      num_layers=6,
      hidden_size=80,
      latent_size=0,

      device=device,

      xavier_init=True,
    ).to(device)

  def forward(self, rays):
    pts, ts, r_o, r_d = compute_pts_ts(rays, self.t_near, self.t_far, self.steps)
    return self.from_pts(pts, ts, r_o, r_d)

  def from_pts(self, pts, ts, r_o, r_d):
    density, feats = self.estim(pts).split([1, 3], dim=-1)

    return volumetric_integrate(density, feats, ts)

# A plain old nerf
class PlainNeRF(CommonNeRF):
  def __init__(
    self,
    latent_size: int = 0,
    intermediate_size: int = 32,
    out_features: int = 3,

    device: torch.device = "cuda",

    **kwargs,
  ):
    super().__init__(**kwargs)
    self.latent = None
    if latent_size == 0:
      self.latent = torch.tensor([[]], device=device, dtype=torch.float)
    self.latent_size = latent_size + self.mip_size()

    self.first = SkipConnMLP(
      in_size=3, out=1 + intermediate_size,
      latent_size=self.latent_size,

      num_layers = 6,
      hidden_size = 128,

      device=device,

      xavier_init=True,
    ).to(device)

    self.second = SkipConnMLP(
      in_size=2, out=out_features,
      latent_size=self.latent_size + intermediate_size,

      num_layers=5,
      hidden_size=64,
      device=device,

      xavier_init=True,
    ).to(device)

  def assign_latent(self, latent):
    assert(latent.shape[-1] == self.latent_size)
    assert(len(latent.shape) == 2), "expected latent in [B, L]"
    self.latent = latent

  def forward(self, rays, lights=None):
    pts, ts, r_o, r_d = compute_pts_ts(rays, self.t_near, self.t_far, self.steps, with_noise=0.5)
    return self.from_pts(pts, ts, r_o, r_d)

  def from_pts(self, pts, ts, r_o, r_d):
    latent = self.latent[None, :, None, None, :].expand(pts.shape[:-1] + (-1,))

    mip_enc = self.mip_encoding(r_o, r_d, ts)

    # If there is a mip encoding, stack it with the latent encoding.
    if mip_enc is not None: latent = torch.cat([latent, mip_enc], dim=-1)

    first_out = self.first(pts, latent if latent.shape[-1] != 0 else None)

    density = first_out[..., 0]
    intermediate = first_out[..., 1:]

    elev_azim_r_d = dir_to_elev_azim(r_d)[None, ...].expand(pts.shape[:-1]+(2,))
    rgb = self.second(
      elev_azim_r_d,
      torch.cat([intermediate, latent], dim=-1)
    )

    return volumetric_integrate(density, rgb, ts)

# NeRF with a thin middle layer, for encoding information
class NeRFAE(CommonNeRF):
  def __init__(
    self,
    latent_size: int = 0,
    intermediate_size: int = 32,
    out_features: int = 3,

    encoding_size: int = 32,

    device="cuda",
    **kwargs,
  ):
    super().__init__(**kwargs)
    self.latent = None
    self.latent_size = latent_size
    if self.latent_size == 0:
      self.latent = torch.tensor([[]], device=device)

    self.encode = SkipConnMLP(
      in_size=3, out=encoding_size,
      latent_size=latent_size,
      num_layers=5,
      hidden_size=64,
      device=device,
      xavier_init=True,
    ).to(device)

    self.density = SkipConnMLP(
      in_size=encoding_size, out=1,
      latent_size=latent_size,
      num_layers=3,
      hidden_size=32,

      freqs=0,
      device=device,
      xavier_init=True,
    ).to(device)

    self.rgb = SkipConnMLP(
      in_size=2, out=out_features,
      latent_size=encoding_size,

      num_layers=3,
      hidden_size=32,

      device=device,
      xavier_init=True,
    ).to(device)

  def assign_latent(self, latent):
    assert(latent.shape[-1] == self.latent_size)
    assert(len(latent.shape) == 2), "expected latent in [B, L]"
    self.latent = latent
  def forward(self, rays):
    pts, ts, r_o, r_d = compute_pts_ts(rays, self.t_near, self.t_far, self.steps, with_noise=0.1)
    return self.from_pts(pts, ts, r_o, r_d)

  def from_pts(self, pts, ts, r_o, r_d):
    latent = self.latent[None, :, None, None, :].expand(pts.shape[:-1] + (-1,))

    encoded = self.encode(pts)

    density = self.density(encoded).squeeze(-1)
    density = density + torch.randn_like(density) * 1e-2

    elev_azim_r_d = dir_to_elev_azim(r_d)[None, ...].expand(pts.shape[:-1]+(2,))

    rgb = self.rgb(
      elev_azim_r_d,
      encoded,
    )

    return volumetric_integrate(density, rgb, ts)

# Dynamic NeRF for multiple frams
class DynamicNeRF(nn.Module):
  def __init__(
    self,
    canonical: nn.Module,
    device="cuda",
  ):
    super().__init__()

    self.delta_estim = SkipConnMLP(
      # x,y,z,t -> dx, dy, dz
      in_size=4, out=3,

      num_layers = 5,
      hidden_size = 128,

      device=device,
    ).to(device)

    self.canonical = canonical.to(device)
  def assign_latent(self, latent):
    assert(latent.shape[-1] == self.latent_size)
    assert(len(latent.shape) == 2), "expected latent in [B, L]"
    self.latent = latent
  def forward(self, rays_t):
    rays, t = rays_t
    device=rays.device
    pts, ts, r_o, r_d = compute_pts_ts(
      rays,
      self.canonical.t_near,
      self.canonical.t_far,
      self.canonical.steps,
    )
    t = t[None, :, None, None, None].expand(pts.shape[:-1] + (1,))
    dp = self.delta_estim(torch.cat([pts, t], dim=-1))
    dp = torch.where(
      t == 0,
      torch.zeros_like(pts),
      dp,
    )
    pts = pts + dp
    return self.canonical.from_pts(pts, ts, r_o, r_d)

