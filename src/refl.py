import torch
import torch.nn as nn
import torch.nn.functional as F
import random

from .nerf import ( CommonNeRF, compute_pts_ts )
from .neural_blocks import ( SkipConnMLP, NNEncoder )
from .utils import ( autograd, eikonal_loss, dir_to_elev_azim )

def load(args):
  raise NotImplementedError()

class Reflectance(nn.Module):
  def __init__(self): super().__init__()
  def forward(self, r_d): raise NotImplementedError()

class BasicReflectance(Reflectance):
  def __init__(self):
    super().__init__()
    self.mlp = SkipConnMLP(
      in_size = 5, out = 3,
      enc=NNEncoder(input_dims=5),
      xavier_init=True,
    )
  def forward(self, x, r_d):
    elaz_rd = dir_to_elev_azim(r_d)
    v = torch.cat([x, elaz_rd], dim=-1)
    return self.mlp(v)
