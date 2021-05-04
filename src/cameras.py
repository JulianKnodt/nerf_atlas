import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from .utils import rotate_vector
import random

# generates random positions on  a grid
def generate_positions(size:int, sample_size:int, device="cuda"):
  return torch.randint(0, size, (sample_size, 2), device=device)

# generate random
def generate_continuous(size:int, sample_size:int, device="cuda"):
  u = random.randint(0, max(size-sample_size, 0))
  v = random.randint(0, max(size-sample_size, 0))
  return torch.stack(torch.meshgrid(
    torch.arange(v, min(v+sample_size, size), device=device),
    torch.arange(u, min(u+sample_size, size), device=device),
  ), dim=-1).reshape(-1, 2)

# General Camera interface
@dataclass
class Camera(nn.Module):
  camera_to_world = None
  world_to_camera = None
  # samples from positions in [0,1] screen space to global
  def sample_positions(self, positions):
    raise NotImplementedError()

# A camera made specifically for generating rays from NeRF models
@dataclass
class NeRFCamera(Camera):
  cam_to_world:torch.tensor = None
  focal: float=None
  device:str ="cuda"
  def __len__(self): return self.cam_to_world.shape[0]

  # support indexing to get sub components of a camera
  def __getitem__(self, v):
    return NeRFCamera(cam_to_world=self.cam_to_world[v], focal=self.focal, device=self.device)

  def sample_positions(
    self,
    position_samples,
    size=512,
    with_noise=False,
  ):
    device=self.device
    u,v = position_samples.split([1,1], dim=-1)
    # u,v each in range [0, size]
    if with_noise:
      u = u + (torch.rand_like(u)-0.5)*with_noise
      v = v + (torch.rand_like(v)-0.5)*with_noise

    d = torch.stack(
      [
        # W
        (u - size * 0.5) / self.focal,
        # H
        -(v - size * 0.5) / self.focal,
        -torch.ones_like(u),
      ],
      dim=-1,
    )
    r_d = torch.sum(
      d[..., None, :] * self.cam_to_world[..., :3, :3], dim=-1
    )
    # normalize direction and exchange [N,B,3] -> [B,N,3]
    r_d = F.normalize(r_d, dim=-1).permute(2,0,1,3)
    r_o = self.cam_to_world[..., :3, -1][:, None, None, :].expand_as(r_d)
    return torch.cat([r_o, r_d], dim=-1)

# The camera described in the NeRF-- paper
@dataclass
class NeRFMMCamera(Camera):
  # position
  t: torch.tensor = None
  # angle of rotation about axis
  angle: torch.tensor = None
  axis:torch.tensor = None
  # intrinsic focal positions
  focals: torch.tensor = None
  device:str ="cuda"
  def __len__(self): return self.t.shape[0]
  def parameters(self): return [angle, axis, t, focals]

  def sample_positions(
    self,
    position_samples,
    sampler,
    size=512,
    with_noise=False,
    N=1,
  ):
    device=self.device
    u,v = position_samples.split(1, dim=-1)
    # u,v each in range [0, size]
    if with_noise:
      u = u + (torch.rand_like(u)-0.5)*with_noise
      v = v + (torch.rand_like(v)-0.5)*with_noise

    d = torch.stack(
      [
        # W
        (u - size * 0.5) / self.focals[..., 0],
        # H
        -(v - size * 0.5) / self.focals[..., 1],
        -torch.ones_like(u),
      ],
      dim=-1,
    )
    r_d = rotate_vector(d, self.axis, self.angle.cos(), self.angle.sin())
    # normalize direction and exchange [W,H,B,3] -> [B,W,H,1,3]
    r_d = F.normalize(r_d, dim=-1).permute(2,0,1,3).unsqueeze(-2)
    r_o = self.t[:, None, None, None, :].expand_as(r_d)
    return torch.cat([r_o, r_d], dim=-1)
