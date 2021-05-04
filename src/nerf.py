import torch
import torch.nn as nn
import torch.nn.functional as F
import random

from .neural_blocks import ( SkipConnMLP )
from .utils import dir_to_elev_azim

# A plain old nerf
class PlainNeRF(nn.Module):
  def __init__(
    self,
    latent_size: int = 32,
    intermediate_size: int = 32,
    steps = 64,
    device="cuda",
  ):
    super().__init__()
    self.latent = None
    self.latent_size = latent_size
    if self.latent_size == 0:
      self.latent = torch.tensor([[]], device=device)

    self.steps = steps

    self.first = SkipConnMLP(
      in_size=3, out=1 + intermediate_size,
      latent_size=latent_size,

      num_layers = 5,
      hidden_size = 128,

      device=device,
    ).to(device)
    self.second = SkipConnMLP(
      in_size=2, out=3,
      latent_size=latent_size + intermediate_size,

      num_layers=5,
      hidden_size=32,
      device=device,
    ).to(device)

    self.t_near = 0.4
    self.t_far = 2
  def assign_latent(self, latent):
    assert(latent.shape[-1] == self.latent_size)
    assert(len(latent.shape) == 2), "expected latent in [B, L]"
    self.latent = latent
  def forward(self, rays, lights=None):
    r_o, r_d = rays.split([3,3], dim=-1)
    device = r_o.device
    # time steps
    ts = torch.linspace(self.t_near, self.t_far, self.steps, device=device)
    pts = r_o.unsqueeze(0) + ts[:, None, None, None, None] * r_d.unsqueeze(0)
    latent = self.latent[None, :, None, None, :].expand(pts.shape[:-1] + (-1,))

    first_out = self.first(pts, latent if self.latent_size != 0 else None)

    alpha = first_out[..., 0]
    intermediate = first_out[..., 1:]

    elev_azim_r_d = dir_to_elev_azim(r_d)[None, ...].expand(pts.shape[:-1]+(2,))
    rgb = self.second(
      elev_azim_r_d,
      torch.cat([intermediate, latent], dim=-1)
    ).sigmoid()
    #if True: alpha = alpha + torch.randn_like(alpha) * 1e-3
    end_val = torch.tensor([1e10], device=device, dtype=torch.float)
    dists = torch.cat([ts[1:] - ts[:-1], end_val], dim=-1)
    sigma_a = F.relu(alpha)
    alpha = 1.0 - torch.exp(-sigma_a * dists[:, None, None, None].expand_as(sigma_a))
    cp = torch.cumprod(1-alpha + 1e-10,dim=0)
    cp = torch.roll(cp, 1, 0)
    last = cp[-1, ...]
    cp[-1, ...] = 1
    weights = alpha * cp
    rgb_out = (weights[..., None] * rgb).sum(dim=0)
    return rgb_out

# NeRF which decomposes different components into different functions
class PartialNeRF(nn.Module):
  def __init__(
    self,
    latent_size: int = 32,
    intermediate_size: int = 32,
    first = {
      "layers": 4,
      "hidden_size": 32,
    },
    second = {
      "layers": 4,
      "hidden_size": 32,
    },
    device="cuda",
  ):
    super().__init__()
    self.latent = None
    self.latent_size = latent_size

    self.first = SkipConnMLP(
      in_size=3, out=1 + intermediate_size,
      latent_size=latent_size,
      xavier_init=True,

      num_layers = first['layers'],
      hidden_size= first['hidden_size'],

      device=device,
    ).to(device)
    self.second = SkipConnMLP(
      in_size=2, out=3,
      latent_size=latent_size + intermediate_size,

      num_layers = second['layers'],
      hidden_size= second['hidden_size'],
      xavier_init=True,

      device=device,
    ).to(device)
  def assign_latent(self, latent):
    assert(latent.shape[0] == self.latent_size)
    self.latent = latent
  def forward(self, rays, steps=16):
    assert(self.latent is not None)
    r_o, r_d = rays.split([3,3], dim=-1)
    device = r_o.device
    ts = torch.linspace(0.4, 1.5 + random.random()*0.01, steps, device=device)
    pts = r_o.unsqueeze(0) + torch.tensordot(ts, r_d,dims=0)

    first_out = self.first(pts)

    # get alpha from first layer
    alpha = first_out[..., 0, None]
    intermediate = first_out[..., 1:]

    elev_azim_r_d = dir_to_elev_azim(r_d)
    rgb = self.second(
      elev_azim_r_d[None, ...].expand(latent.shape[:-1]+(2,)),
      intermediate,
    )
    return alpha, rgb
  @classmethod
  def volumetric_integrate(alpha, rgb, ts):
    noise = 0
    #if True: noise = torch.randn_like(alpha) * 0.01
    sigma_a = F.relu(alpha + noise).squeeze(-1)
    alpha = 1 - torch.exp(-sigma_a * ts[:, None, None].expand_as(sigma_a))
    cp = torch.cumprod((1-alpha).clamp(min=1e-10),dim=0)
    cp = torch.roll(cp, 1, 0)
    cp[-1, ...] = 1
    weights = alpha * cp
    rgb_out = (weights[..., None] * rgb.sigmoid()).sum(dim=0)
    return rgb_out

