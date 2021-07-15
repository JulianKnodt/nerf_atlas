import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import math

from .neural_blocks import ( SkipConnMLP, FourierEncoder )
from .utils import ( autograd, elev_azim_to_dir )

def load(args):
  if args.light_kind == "field": cons = Field
  elif args.light_kind == "point": cons = Point
  elif args.light_kind == "dataset": cons = lambda **kwargs: None
  else: raise NotImplementedError(f"light kind: {args.light_kind}")

  return cons()

class Light(nn.Module):
  def __init__(
    self
  ):
    super().__init__()
  def __getitem__(self, _v): return self
  @property
  def can_sample(self): return False
  def forward(self, x): raise NotImplementedError()

class Field(Light):
  def __init__(self):
    super().__init__()
    self.mlp = SkipConnMLP(in_size=3, out=5, enc=FourierEncoder(input_dims=3))
  def forward(self, x):
    intensity, elaz = self.mlp(x).split([3,2], dim=-1)
    r_d = elev_azim_to_dir(elaz.tanh() * (math.pi-1e-6))
    return intensity.sigmoid(), r_d

class Point(Light):
  def __init__(
    self,
    center = [0,0,0],
    train_center=False,
    intensity=[1],
    train_intensity=False,
  ):
    super().__init__()
    if type(center) == torch.Tensor: self.center = center
    else:
      self.center = nn.Parameter(torch.tensor(
        center, requires_grad=train_center, dtype=torch.float,
      ))
    self.train_center = train_center

    if type(intensity) == torch.Tensor: self.intensity = intensity
    else:
      if len(intensity) == 1: intensity = intensity * 3
      intensity = torch.tensor(
        intensity, requires_grad=train_intensity, dtype=torch.float,
      ).expand([self.center.shape[0], -1])
      self.intensity = nn.Parameter(intensity)
    self.train_intensity = train_intensity
  def __getitem__(self, v):
    return Point(
      center=self.center[v], train_center=self.train_center,
      intensity=self.intensity[v], train_intensity=self.train_intensity,
    )
  @property
  def can_sample(self): return True
  def forward(self, x, mask=None):
    loc = self.center[:, None, None, :]
    if mask is not None: loc = loc.expand(mask.shape + (3,))[mask]
    d = x - loc
    dist = torch.linalg.norm(d, dim=-1)
    dir = F.normalize(d, eps=1e-6, dim=-1)
    decay = dist.square()
    intn = self.intensity[:, None, None, :]
    if mask is not None: intn = intn.expand(mask.shape + (3,))[mask]
    spectrum = intn/decay.clamp(min=1e-6).unsqueeze(-1)

    return dir, spectrum
