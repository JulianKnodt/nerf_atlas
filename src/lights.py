import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import math

from .nerf import ( CommonNeRF, compute_pts_ts )
from .neural_blocks import ( SkipConnMLP, FourierEncoder )
from .utils import ( autograd, eikonal_loss, elev_azim_to_dir )

def load(args):
  if args.light_kind == "field": cons = Field
  if args.light_kind == "point": cons = Point
  else: raise NotImplementedError()

  return cons()

class Light(nn.Module):
  def __init__(self):
    super().__init__()
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
    center = [0,0,0]
    train_center=False,
    intensity=1,
    train_intensity=False,
  ):
    super().__init__()
    self.center = nn.Parameter(torch.tensor(
      center, requires_grad=train_center, dtype=torch.float,
    ))
    self.intensity = nn.Parameter(torch.tensor(
      intensity, requires_grad=train_intensity, dtype=torch.float,
    ))
  def forward(self, x):
    raise NotImplementedError()
