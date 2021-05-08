import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import (
  fourier, create_fourier_basis,
)
from itertools import chain
from typing import Optional

class SkipConnMLP(nn.Module):
  "MLP with skip connections and fourier encoding"
  def __init__(
    self,

    num_layers = 5,
    hidden_size = 128,
    in_size=3,
    out=3,

    skip=3,
    freqs = 16,
    sigma=2<<4,
    device="cuda",
    activation = nn.LeakyReLU(inplace=True),

    latent_size=0,

    zero_init = False,
    xavier_init = False,
  ):
    super(SkipConnMLP, self).__init__()
    self.in_size = in_size
    assert(type(freqs) == int)
    self.basis_p, map_size = create_fourier_basis(
      freqs, features=in_size, freq=sigma, device=device
    )

    self.dim_p = map_size + latent_size
    self.skip = skip
    self.latent_size = latent_size
    skip_size = hidden_size + self.dim_p

    # with skip
    hidden_layers = [
      nn.Linear(
        skip_size if (i % skip) == 0 and i != num_layers-1 else hidden_size,
        hidden_size,
      ) for i in range(num_layers)
    ]

    self.init =  nn.Linear(self.dim_p, hidden_size)
    self.layers = nn.ModuleList(hidden_layers)
    self.out = nn.Linear(hidden_size, out)
    weights = [
      self.init.weight,
      self.out.weight,
      *[l.weight for l in self.layers],
    ]
    biases = [
      self.init.bias,
      self.out.bias,
      *[l.bias for l in self.layers],
    ]
    if zero_init:
      for t in weights: nn.init.zeros_(t)
      for t in biases: nn.init.zeros_(t)
    if xavier_init:
      for t in weights: nn.init.xavier_uniform_(t)
      for t in biases: nn.init.zeros_(t)

    self.activation = activation

  def forward(self, p, latent: Optional[torch.Tensor]=None):
    batches = p.shape[:-1]
    init = fourier(p.reshape(-1, self.in_size), self.basis_p)
    if latent is not None:
      init = torch.cat([init, latent.reshape(-1, self.latent_size)], dim=-1)
    x = self.init(init)
    for i, layer in enumerate(self.layers):
      if i != len(self.layers)-1 and (i % self.skip) == 0:
        x = torch.cat([x, init], dim=-1)
      x = layer(self.activation(x))
    out_size = self.out.out_features
    return self.out(self.activation(x)).reshape(batches + (out_size,))


class Upsampler(nn.Module):
  def __init__(
    self,
    in_size: int,
    out: int,

    in_features:int = 3,
    out_features:int = 3,

    hidden_size=15,

    num_layers=5,
    activation = nn.LeakyReLU(inplace=True),
  ):
    super().__init__()
    self.layers = nn.Sequential(
      nn.Conv2d(in_features, hidden_size, 3, 1, 1),
      activation,
      *list(chain.from_iterable([
        (nn.Conv2d(hidden_size, hidden_size, 3, 1, 1), activation)
        for _ in range(max(num_layers-2, 0))
      ])),
      nn.Conv2d(hidden_size, out_features, 3, 1, 1),
    )
    self.out = out
  def forward(self, x):
    img = x.permute(0,3,1,2)
    out = F.interpolate(img, size=(self.out, self.out), mode="bilinear", align_corners=False)
    out = self.layers(out).permute(0,2,3,1)
    return out.sigmoid()
