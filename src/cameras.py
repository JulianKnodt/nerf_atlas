import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from .utils import rotate_vector
from .neural_blocks import ( SkipConnMLP )
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
    r_d = r_d.permute(2,0,1,3)
    r_o = self.cam_to_world[..., :3, -1][:, None, None, :].expand_as(r_d)
    return torch.cat([r_o, r_d], dim=-1)

def vec2skew(v):
  zero = torch.zeros(v.shape[:-1] + (1,), device=v.device, dtype=v.dtype)
  return torch.stack([
    torch.cat([zero, -v[..., 2:3], v[..., 1:2]], dim=-1),
    torch.cat([v[..., 2:3], zero, -v[..., 0:1]], dim=-1),
    torch.cat([-v[..., 1:2], v[..., 0:1], zero], dim=-1),
  ], dim=-2)

def exp(r):
  skew_r = vec2skew(r)
  norm_r = torch.linalg.norm(r).clamp(min=1e-6)
  eye = torch.eye(3, dtype=r.dtype, device=r.device)
  R = eye + \
    (norm_r.sin()/norm_r) * skew_r + \
    ((1 - norm_r.cos())/norm_r.square()) * (skew_r @ skew_r)
  return R

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
    t = torch.zeros(batch_size, 3, dtype=torch.float, device=device, requires_grad=True)
    r = torch.zeros_like(t, requires_grad=True)
    # focals are for all the cameras and thus don't have batch dim
    focals = torch.tensor([0.7, 0.7], dtype=torch.float, device=device, requires_grad=True)
    return cls(t=t, r=r, focals=focals, device=device)

  def parameters(self): return [self.r, self.t, self.focals]

  def __getitem__(self, v):
    return NeRFMMCamera(t=self.t[v],r=self.r[v],focals=self.focals)

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
    R = exp(self.r)
    r_d = torch.sum(d[..., None, :] * R, dim=-1)
    # normalize direction and exchange [W,H,B,3] -> [B,W,H,3]
    r_d = F.normalize(r_d, dim=-1).permute(2,0,1,3)
    r_o = self.t[:, None, None, :].expand_as(r_d)
    return torch.cat([r_o, r_d], dim=-1)

# learned time varying camera
class NeRFMMTimeCamera(Camera):
  def __init__(
    self,
    batch_size,
    # position
    translate: torch.tensor,
    # angle of rotation about axis
    rot: torch.tensor,
    # intrinsic focal positions
    focals: torch.tensor,
    delta_params: SkipConnMLP = None,
    device:str ="cuda"
  ):
    ...
    if delta_params is None:
      delta_params = SkipConnMLP(
        in_size=1, out=6,
        zero_init=True,
      )
    self.delta_params = delta_params
    self.focals = nn.Parameter(focals.requires_grad_())
    self.rot = nn.Parameter(translate.requires_grad_())
    self.translate = nn.Parameter(translate.requires_grad_())
  def __len__(self): return self.t.shape[0]
  def __getitem__(self, v):
    raise NotImplementedError()
    return NeRFMMTimeCamera(
      cam_to_world=self.cam_to_world[v], focal=self.focal, device=self.device
    )
  def sample_positions(
    self,
    position_samples,
    t,
    size=512,
    with_noise=False,
  ):
    ...
