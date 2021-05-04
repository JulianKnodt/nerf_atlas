import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import (
  fourier, create_fourier_basis,
)

class SkipConnMLP(nn.Module):
  "MLP with skip connections and fourier encoding"
  def __init__(
    self,

    num_layers = 8,
    hidden_size=64,
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

  def forward(self, p, latent=None):
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
  # sets this MLP to always just return its own input
  def prime_identity(
    self,
    lr=1e-4,
    iters=50_000,
    batches=4096,
    device="cuda",
  ):
    opt = torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=0)
    for i in range(iters):
      opt.zero_grad()
      x = torch.rand_like(batches, self.in_size, device=device)
      y = self(x)
      loss = F.mse_loss(x, y)
      loss.backward()
      opt.step()
