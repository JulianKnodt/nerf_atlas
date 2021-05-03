import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import (
  weak_sigmoid, log_polar_indices,
  cartesian_indices, cartesian_indices_avgs,

  fourier2, create_fourier_basis2,
)
import torch.distributions as D

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
    self.basis_p, map_size = create_fourier_basis2(
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
    init = fourier2(p.reshape(-1, self.in_size), self.basis_p)
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

class AutoDecoder(nn.Module):
  def __init__(
    self,
    in_size=3,
    out=3,
    num_layers=4,
    w_in=True,
    code_size=64,
    freqs=[2**4, 2**4, 2**5, 2**5, 2**6, 2**6, 2**7, 2**7],
    hidden_size=64,
    skip=3,
    device="cuda",

    activation = F.leaky_relu,
  ):
    super(AutoDecoder, self).__init__()
    self.code = torch.rand(code_size, device=device, requires_grad=True)

    self.in_size = in_size
    self.w_in = w_in = in_size if w_in else 0

    n_p = len(freqs)
    self.basis_p = create_fourier_basis([w_in], freqs, n_p, device)

    self.dim_p = code_size + w_in + 2 * n_p
    self.skip = skip
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

    self.activation = activation

  def forward(self, p):
    batches = p.shape[:-1]
    init = fourier(p.reshape(-1, self.in_size), self.basis_p)
    init = torch.cat([
      self.code.expand(init.shape[0], -1),
      init,
    ], dim=-1)
    x = self.init(init)
    for i, layer in enumerate(self.layers):
      if i != len(self.layers)-1 and (i % self.skip) == 0:
        x = torch.cat([x, init], dim=-1)
      x = layer(self.activation(x))
    return self.out(self.activation(x)).reshape(*batches + (-1,))
  def latent_parameters(self): return [self.code]
  def randomize_code(self): self.code = torch.randn_like(self.code, requires_grad=True)
  def set_code(self, code):
    assert(self.code.shape == code.shape)
    self.code = code

class DensityEstimator(nn.Module):
  def __init__(
    self,
    in_size=2,
    dists=2<<4,
    device="cuda",
  ):
    super().__init__()
    self.centers = torch.zeros(dists, in_size, device=device, requires_grad=True)
    self.centers = nn.Parameter(self.centers)
    self.vars = torch.zeros((in_size * (in_size+1))//2, device=device, requires_grad=True)\
      .unsqueeze(0)\
      .repeat(dists, 1)\
      .detach()
    self.vars = nn.Parameter(self.vars)
    self.in_size = in_size
    self.weights = nn.Parameter(torch.zeros(dists, device=device, requires_grad=True))
  def forward(self, shape):
    a, d0, d1 = self.vars.split(1, dim=-1)
    z = torch.zeros_like(a)
    scale_tril = torch.cat([
      d0.exp(), z,
      a, d1.exp(),
    ], dim=-1).reshape(-1, self.in_size, self.in_size)
    dist = D.MultivariateNormal(self.centers, scale_tril=scale_tril)
    out = dist.rsample(shape)
    k = F.softmax(self.weights, dim=-1)
    out = out.permute(4, 0, 1, 2, 3, 5)
    val = out * k[:, None, None, None, None, None].expand_as(out)
    val = val.sum(dim=0)
    pdf = (dist.log_prob(val).exp() * k[None, None, None, :]).sum(dim=-1)
    assert((pdf <= 1.).all())
    assert((pdf >= 0.).all())
    return val, pdf
  def pdf(self, val):
    a, d0, d1 = self.vars.split(1, dim=-1)
    z = torch.zeros_like(a)
    scale_tril = torch.cat([
      d0.exp(), z,
      a, d1.exp(),
    ], dim=-1).reshape(-1, self.in_size, self.in_size)
    dist = D.MultivariateNormal(self.centers, scale_tril=scale_tril)
    k = F.softmax(self.weights, dim=-1)
    pdf_indiv = dist.log_prob(val.unsqueeze(-2)).exp()
    pdf = (pdf_indiv * k.expand_as(pdf_indiv)).sum(dim=-1,keepdim=True)
    return pdf


# Given an image returns a latent code for it
class Embedder(nn.Module):
  def __init__(
    self,
  ):
    super().__init__()
  def forward(self, img, word):
    # TODO some number of convolutional layers then MLP to return feature vector
    raise NotImplementedError()

# simple gan discriminator
class Discriminator(nn.Module):
  def __init__(
    self,
    num_features = 64,
    num_channel = 3,
    device="cuda",
  ):
    super().__init__()
    self.main = nn.Sequential(
      # input is (nc) x 64 x 64
      #nn.Conv2d(num_channel, num_features, 4, 2, 1, bias=False),
      #nn.LeakyReLU(0.2, inplace=True),
      # state size. (ndf) x 32 x 32
      nn.Conv2d(num_channel, num_features * 2, 4, 2, 1, bias=False),
      nn.BatchNorm2d(num_features * 2),
      nn.LeakyReLU(0.2, inplace=True),
      # state size. (ndf*2) x 16 x 16
      nn.Conv2d(num_features * 2, num_features * 4, 4, 2, 1, bias=False),
      nn.BatchNorm2d(num_features * 4),
      nn.LeakyReLU(0.2, inplace=True),
      # state size. (ndf*4) x 8 x 8
      nn.Conv2d(num_features * 4, num_features * 8, 4, 2, 1, bias=False),
      nn.BatchNorm2d(num_features * 8),
      nn.LeakyReLU(0.2, inplace=True),
      # state size. (ndf*8) x 4 x 4
      nn.Conv2d(num_features * 8, 1, 4, 1, 0, bias=False),
    )
  def forward(self, x):
    assert(len(x.shape) == 4)
    _, C, W, H = x.shape
    #assert(C == 3 and W == 64 and H == 64)
    return self.main(x).reshape(-1)

def equal_conv(*args, **kwargs):
  conv = nn.Conv2d(*args, **kwargs)
  conv.weight.data.normal_()
  conv.bias.data.zero_()
  return conv

class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size,
        padding,
        kernel_size2=None,
        padding2=None,
        downsample=False,
    ):
        super().__init__()
        pad1 = pad2 = padding
        pad2 = padding2 if padding2 is not None else padding

        kernel1 = kernel_size
        kernel2 = kernel_size2 if kernel_size2 is not None else kernel_size

        conv1 = nn.Sequential(
          equal_conv(in_channel, out_channel, kernel1, padding=pad1),
          nn.LeakyReLU(0.2)
        )

        if downsample:
            self.conv2 = nn.Sequential(
              #Blur(out_channel),
              equal_conv(out_channel, out_channel, kernel2, padding=pad2),
              nn.AvgPool2d(2), nn.LeakyReLU(0.2),
            )
        else:
            self.conv2 = nn.Sequential(
                equal_conv(out_channel, out_channel, kernel2, padding=pad2),
                nn.LeakyReLU(0.2),
            )
        self.layers = nn.Sequential(conv1, conv2)
    def forward(self, x): return self.layers(x)


class MultiDiscriminator(nn.Module):
  def __init__(
    self,
    num_features = 64,
    num_channel = 3,
    device="cuda",
  ):
    self.conv_layers = nn.ModuleList(
    )
  def forward(self):
    raise NotImplementedError()



















