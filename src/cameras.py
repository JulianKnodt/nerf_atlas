import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from .utils import rotate_vector
import random

# General Camera interface
@dataclass
class Camera(nn.Module):
  camera_to_world = None
  world_to_camera = None
  # samples from positions in [0,1] screen space to global
  def sample_positions(self, positions): raise NotImplementedError()

# A camera made specifically for generating rays from NeRF models
@dataclass
class NeRFCamera(Camera):
  cam_to_world:torch.tensor = None
  focal: float=None
  device:str ="cuda"
  near: float = None
  far: float = None

  def __len__(self): return self.cam_to_world.shape[0]

  @classmethod
  def identity(cls, batch_size: int, device="cuda"):
    c2w = torch.tensor([
      [1,0,0, 0],
      [0,1,0, 0],
      [0,0,1, 0],
    ], device=device).unsqueeze(0).expand(batch_size, 3, 4)
    return cls(cam_to_world=c2w, focal=0.5, device=device)

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
    r_d = r_d.permute(2,0,1,3)#F.normalize(r_d, dim=-1).permute(2,0,1,3)
    r_o = self.cam_to_world[..., :3, -1][:, None, None, :].expand_as(r_d)
    return torch.cat([r_o, r_d], dim=-1)

def vec2skew(vec):
  zero = torch.zero(vec.shape[:-1], device=vec.device, dtype=vec.dtype)
  print(vec.shape)
  exit()
  return torch.stack([
    torch.cat([zero, -v[..., 2:3], v[..., 1:2]], dim=-1),
    torch.cat([v[..., 2:3], zero, -v[..., 0:1]], dim=-1),
    torch.cat([-v[..., 1:2], v[..., 0:1], zero], dim=-1),
  ], dim=-2)

# The camera described in the NeRF-- paper
@dataclass
class NeRFMMCamera(Camera):
  # position
  t: torch.tensor = None
  # angle of rotation about axis
  r: torch.tensor = None
  # intrinsic focal positions
  focals: torch.tensor = None
  device:str ="cuda"

  def __len__(self): return self.t.shape[0]

  @classmethod
  def identity(cls, batch_size: int, device="cuda"):
    t = torch.tensor([0, 0, 0], dtype=torch.float, device=device, requires_grad=True)\
      .unsqueeze(0).expand(batch_size, 3)
    r = torch.tenso
    # focals are for all the cameras and thus don't have batch dim
    focals = torch.tensor([0.7, 0.7], dtype=torch.float, device=device, requires_grad=True)
    return cls(t=t, angle=angle, axis=axis, focal=focals, device=device)

  def parameters(self): return [angle, axis, t, focals]

  def __getitem__(self, v):
    return NeRFMMCamera(
      t=self.t[v], angle=self.angle[v],axis=self.axis[v], focals=self.focals,
    )

  def sample_positions(
    self,
    position_samples,
    size=512,
    with_noise=False,
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
