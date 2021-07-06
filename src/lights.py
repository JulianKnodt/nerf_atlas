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
  else: raise NotImplementedError(f"light kind: {args.light_kind}")

  return cons()

class Light(nn.Module):
  def __init__(
    self
  ):
    super().__init__()

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
    self.center = nn.Parameter(torch.tensor(
      center, requires_grad=train_center, dtype=torch.float,
    ))
    self.intensity = nn.Parameter(torch.tensor(
      intensity, requires_grad=train_intensity, dtype=torch.float,
    ))
  @property
  def can_sample(self): return True
  def forward(self, x):
    print(x.shape, self.center.shape)
    d = x - self.center
    dist = torch.linalg.norm(d, dim=-1)
    dir = F.normalize(d, dim=-1)
    decay = dist.square()
    print(decay.shape, self.intensity.shape)
    spectrum = F.normalize(self.intensity, dim=-1)/dist.clamp(min=1e-6)

    return dir, spectrum
