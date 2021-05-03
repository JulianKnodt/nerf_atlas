import torch
import torch.nn as nn
import torch.nn.functional as F
from ..neural_blocks import ( SkipConnMLP )
import random
from ..utils import dir_to_elev_azim

# A plain old nerf
class PlainNeRF(nn.Module):
  def __init__(
    self,
    latent_size: int = 32,
    intermediate_size: int = 32,
    steps = 32,
    device="cuda",
  ):
    super().__init__()
    self.latent = None
    self.latent_size = latent_size

    self.steps = steps

    self.first = SkipConnMLP(
      in_size=3, out=1 + intermediate_size,
      latent_size=latent_size,

      num_layers = 5,
      hidden_size=32,

      device=device,
    ).to(device)
    self.second = SkipConnMLP(
      in_size=2, out=3,
      latent_size=latent_size + intermediate_size,

      num_layers=5,
      hidden_size=32,
      device=device,
    ).to(device)
    self.sky = SkipConnMLP(
      in_size=2, out=3,

      latent_size=latent_size,
      num_layers=3,
      hidden_size=16,
    )
    #self.t_near = 0.4
    #self.t_far = 2
  def assign_latent(self, latent):
    assert(latent.shape[-1] == self.latent_size)
    assert(len(latent.shape) == 2), "expected latent in [B, L]"
    self.latent = latent
  def forward(self, rays, lights):
    assert(self.latent is not None)
    r_o, r_d = rays.split([3,3], dim=-1)
    device = r_o.device
    ts = torch.linspace(0.4, 2 + random.random()*0.1, self.steps, device=device)
    pts = r_o.unsqueeze(0) + torch.tensordot(ts, r_d,dims=0)
    latent = self.latent[None, :, None, None, None, :].expand(pts.shape[:-1] + (-1,))

    first_out = self.first(pts, latent)

    # get alpha from first layer
    alpha = first_out[..., 0, None]
    intermediate = first_out[..., 1:]

    elev_azim_r_d = dir_to_elev_azim(r_d)[None, ...].expand(latent.shape[:-1]+(2,))
    rgb = self.second(
      elev_azim_r_d,
      torch.cat([intermediate, latent], dim=-1)
    ).tanh()
    noise = 0
    if True: noise = torch.randn_like(alpha) * 1e-3
    sigma_a = F.relu(alpha + noise).squeeze(-1)
    alpha = 1 - torch.exp(-sigma_a * ts[:, None, None, None, None].expand_as(sigma_a))
    cp = torch.cumprod((1-alpha).clamp(min=1e-10),dim=0)
    cp = torch.roll(cp, 1, 0)
    last = cp[-1, ...]
    cp[-1, ...] = 1
    weights = alpha * cp
    sky_color = self.sky(elev_azim_r_d[0], latent[0]).tanh()
    rgb_out = (weights[..., None] * rgb).sum(dim=0) + sky_color
    return (rgb_out+1)/2

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
    latent = self.latent[None, None, None, None, None, :].expand(pts.shape[:-1] + (-1,))

    first_out = self.first(pts, latent)

    # get alpha from first layer
    alpha = first_out[..., 0, None]
    intermediate = first_out[..., 1:]

    elev_azim_r_d = dir_to_elev_azim(r_d)
    rgb = self.second(
      elev_azim_r_d[None, ...].expand(latent.shape[:-1]+(2,)),
      torch.cat([intermediate, latent], dim=-1)
    )
    return alpha, rgb
  @classmethod
  def volumetric_integrate(alpha, rgb):
    noise = 0
    #if True: noise = torch.randn_like(alpha) * 0.01
    sigma_a = F.relu(alpha + noise).squeeze(-1)
    alpha = 1 - torch.exp(-sigma_a * ts[:, None, None, None, None].expand_as(sigma_a))
    cp = torch.cumprod((1-alpha).clamp(min=1e-10),dim=0)
    cp = torch.roll(cp, 1, 0)
    cp[-1, ...] = 1
    weights = alpha * cp
    rgb_out = (weights[..., None] * rgb).sum(dim=0)
    return rgb_out.sigmoid()


# NeRF with environment lighting and point light emitter. I shoved them together for
# convenience.
class NeRFLE(nn.Module):
  def __init__(
    self,
    envmap=False,
    bins=4,
    device="cuda",
  ):
    super(NeRFLE, self).__init__()
    self.latent_size = 64
    self.first = SkipConnMLP(
      num_layers = 5,
      hidden_size=128,
      in_size=3, out=1 + self.latent_size,
      device=device,
    ).to(device)
    self.bins = bins
    self.second = SkipConnMLP(
      in_size=self.latent_size + (6 if not envmap else 3 + bins*bins * 3), out=3,
      device=device,
    ).to(device)
    self.envmap = envmap

  def forward(self, rays, lights):
    r_o, r_d = rays.split([3,3], dim=-1)
    device=r_o.device
    ts = torch.linspace(0, 2 + random.random()*0.1, 64, device=device)
    pts = r_o.unsqueeze(0) + torch.tensordot(ts,r_d,dims=0)
    first_out = self.first(pts)
    latent = first_out[..., 1:]
    alpha = first_out[..., 0, None]
    light_encode = None
    if getattr(self, "envmap", False):
      points = torch.stack(
        torch.meshgrid(
          torch.linspace(0, 180, self.bins, device=device),
          torch.linspace(0, 45, self.bins, device=device)
        ),
      dim=-1).reshape(-1, 2)
      from ..utils import elev_azim_to_dir
      # assume that bounding sphere is unit sphere?
      light_encode = lights.envmap(elev_azim_to_dir(points))
      _, B, _, _, _, _three = latent.shape
      light_encode = light_encode.reshape(1, B, 1, 1, 1, -1).expand(latent.shape[:-1] + (-1,))
    else:
      light_encode = lights.location[None, :, None, None, None, :].expand(latent.shape[:-1]+(3,))

    rgb = self.second(torch.cat([
      latent,
      r_d[None, ...].expand(latent.shape[:-1]+(3,)),
      light_encode
    ], dim=-1)).sigmoid()
    noise = 0
    #if True: noise = torch.randn_like(alpha) * 0.01
    sigma_a = F.relu(alpha + noise).squeeze(-1)
    alpha = 1 - torch.exp(-sigma_a * ts[:, None, None, None, None].expand_as(sigma_a))
    cp = torch.cumprod((1-alpha).clamp(min=1e-10),dim=0)
    cp = torch.roll(cp, 1, 0)
    cp[-1, ...] = 1
    # alpha shape is wrong and so is cp
    weights = alpha * cp
    rgb_out = (weights[..., None] * rgb).sum(dim=0)
    return rgb_out

class MPI(nn.Module):
  num_planes: int = 10
  min_t: float
  max_t: float
  device: torch.device
  def __init__(
    self,
    center,
    num_planes=10,
    point=[0,0,0],
    normal=[0,0,-1],
    min_t=1e-1,
    max_t=2,
    device="cuda",
  ):
    self.num_planes = num_planes
    self.min_t = min_t
    self.max_t = max_t
    self.device = torch.device(device)
    self.point = torch.tensor(point, device=device)
    self.normal = torch.tensor(normal, device=device)

  def forward(self, rays, camera=None, active=True, primary=True):
    assert(camera is not None)
    r_o, r_d = torch.split(rays, 3, dim=-1)
    _, prim_r_d = camera.sample_positions(
      positions_samples = torch.tensor([0.5, 0.5], dim=-1, device=self.device),
      size=1,
      sampler=None,
      with_noise=False,
      bundle_size=1,
      N=1,
    ).split(3, dim=-1)
    prim_r_d = F.normalize(prim_r_d, dim=-1)
    s = torch.linspace(self.min_t, self.max_t, steps=self.num_planes, device=self.device)
    lens = torch.tensordot(s, prim_r_d)
    print(lens.shape, r_d.shape)
    exit()
    points = r_d * (lens * r_d).sum(dim=-1, keepdim=True)

    si = SurfaceInteraction(
      p=None,
      t=None,
      obj=self,
    )
    #si.set_normals(n)
    #si.wi = si.to_local(-r_d)
    return si, True

# Convolutional NeRF
class ConvNeRF(nn.Module):
  def __init__(
    self,
    steps = 32,
    latent_size=32,
    intermediate_size=32,

    device="cuda",
  ):
    super().__init__()
    self.latent = None
    self.latent_size = latent_size

    self.steps = steps

    self.firsts = nn.ModuleList([
      SkipConnMLP(
        in_size=3, out=1 + intermediate_size,
        latent_size=latent_size,

        num_layers = 4,
        hidden_size=32,

        device=device,
      ),
      SkipConnMLP(
        in_size=3, out=1 + intermediate_size,
        latent_size=latent_size,

        num_layers = 5,
        hidden_size=32,

        device=device,
      ),
      SkipConnMLP(
        in_size=3, out=1 + intermediate_size,
        latent_size=latent_size,

        num_layers = 5,
        hidden_size=64,

        device=device,
      ),
    ])

    self.second = SkipConnMLP(
      in_size=2, out=3,
      latent_size=latent_size + intermediate_size,

      num_layers=5,
      hidden_size=32,
      device=device,
    ).to(device)

    self.sky = SkipConnMLP(
      in_size=2, out=3,

      latent_size=latent_size,
      num_layers=3,
      hidden_size=16,
    )
    self.t_near = 0.4
    self.t_far = 2
  def assign_latent(self, latent):
    assert(latent.shape[-1] == self.latent_size)
    assert(len(latent.shape) == 2), "expected latent in [B, L]"
    self.latent = latent
  def forward(self, camera, s=512):
    assert(self.latent is not None)
    device = self.latent.device
    out = None
    ts = torch.linspace(
      self.t_near, self.t_far, self.steps,
      device=device
    )
    # TODO add something which limits output to certain number of layers
    for i, layer in enumerate(self.firsts):
      off = (1 << (len(self.firsts) - i + 1))
      shrunk = s // off
      u = torch.arange(0, s, device=device, dtype=torch.float)
      uv = torch.stack(torch.meshgrid(u, u), dim=-1)
      r_o, r_d = camera.sample_positions(
        uv, None, with_noise=1e-3, bundle_size=1, size=shrunk, N=len(camera),
      ).squeeze(-2).split([3,3], dim=-1)
      # [D, N, W, H, 3]
      pts = r_o.unsqueeze(0) + torch.tensordot(ts, r_d,dims=0)
      latent = self.latent[None, :, None, None, :].expand(pts.shape[:-1] + (-1,))
      v = layer(pts, latent)
      # v in [D, N, W, H, 1 + MID]

      if out is None:
        out = v
      else:
        # [N, *, D, W, H]
        interp = F.interpolate(out.permute(1, 4, 0, 2, 3), (pts.shape[0],) + pts.shape[2:4])
        interp = interp.permute(2, 0, 3, 4, 1)
        out = v + interp

    # get alpha from first layer
    alpha = out[..., 0, None]
    intermediate = out[..., 1:]

    elev_azim_r_d = dir_to_elev_azim(r_d)[None, ...].expand(latent.shape[:-1]+(2,))
    rgb = self.second(
      elev_azim_r_d,
      torch.cat([intermediate, latent], dim=-1)
    ).tanh()
    noise = 0
    if True: noise = torch.randn_like(alpha) * 1e-3
    sigma_a = F.relu(alpha + noise).squeeze(-1)
    alpha = 1 - torch.exp(-sigma_a * ts[:, None, None, None].expand_as(sigma_a))
    cp = torch.cumprod((1-alpha).clamp(min=1e-10),dim=0)
    last = cp[-1, ...]
    cp = torch.roll(cp, 1, 0)
    cp[-1, ...] = 1
    weights = alpha * cp
    #sky_color = last * self.sky(elev_azim_r_d[0], latent[0]).tanh()
    rgb_out = (weights[..., None] * rgb).sum(dim=0) #+ sky_color
    return (rgb_out+1)/2
