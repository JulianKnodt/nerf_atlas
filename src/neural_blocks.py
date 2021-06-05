import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from itertools import chain
from typing import Optional

from .utils import ( fourier, create_fourier_basis, )

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
    sigma=1<<5,
    device="cuda",
    activation = nn.LeakyReLU(inplace=True),
    latent_size=0,

    # Record the last layers activation
    last_layer_act = False,

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
        skip_size if (i % skip) == 0 and i != num_layers-1 else hidden_size, hidden_size,
      ) for i in range(num_layers)
    ]

    self.init =  nn.Linear(self.dim_p, hidden_size)
    self.layers = nn.ModuleList(hidden_layers)
    self.out = nn.Linear(hidden_size, out)
    weights = [
      self.init.weight, self.out.weight,
      *[l.weight for l in self.layers],
    ]
    biases = [
      self.init.bias, self.out.bias,
      *[l.bias for l in self.layers],
    ]
    if zero_init:
      for t in weights: nn.init.zeros_(t)
      for t in biases: nn.init.zeros_(t)
    if xavier_init:
      for t in weights: nn.init.xavier_uniform_(t)
      for t in biases: nn.init.zeros_(t)

    self.activation = activation
    self.last_layer_act = last_layer_act

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
    if self.last_layer_act: setattr(self, "last_layer_out", x.reshape(batches + (-1,)))
    out_size = self.out.out_features
    return self.out(self.activation(x)).reshape(batches + (out_size,))

class Upsampler(nn.Module):
  def __init__(
    self,

    in_size: int,
    out: int,

    kernel_size:int = 3,

    repeat:int = 6,
    in_features:int = 3,
    out_features:int = 3,
    feat_decay: float = 2,
    activation = nn.LeakyReLU(inplace=True),
  ):
    super().__init__()
    step_size = (out - in_size)//repeat
    self.sizes = list(range(in_size + step_size, out+step_size, step_size))
    self.sizes = self.sizes[:repeat]
    self.sizes[-1] = out
    assert(kernel_size % 2 == 1), "Must provide odd kernel upsampling"

    feat_sizes = [
      max(out_features, int(in_features // (feat_decay**i))) for i in range(repeat+1)
    ]

    self.base = nn.Conv2d(in_features, out_features, kernel_size, 1, (kernel_size-1)//2)

    self.convs = nn.ModuleList([
      nn.Sequential(
        nn.Conv2d(fs, nfs, kernel_size, 1, (kernel_size-1)//2),
        nn.Dropout2d(0.1, inplace=True),
        nn.LeakyReLU(inplace=True)
      )
      for fs, nfs in zip(feat_sizes, feat_sizes[1:])
      # Move from one feature size to the next
    ])

    self.combine = nn.ModuleList([
      nn.Conv2d(feat_sizes, out_features, kernel_size, 1, (kernel_size-1)//2)
      for feat_sizes in feat_sizes[1:]
    ])

    self.rgb_up_kind="bilinear"
    self.feat_up_kind="nearest"

  def forward(self, x):
    curr = x.permute(0,3,1,2)
    upscaled = self.base(curr) # (N, H_in, W_in, 3)

    for s, conv, combine in zip(self.sizes, self.convs, self.combine):
      resized_old=F.interpolate(upscaled,size=(s,s),mode=self.rgb_up_kind,align_corners=False)

      curr = conv(F.interpolate(curr, size=(s, s), mode=self.feat_up_kind))
      upscaled = resized_old + combine(curr)
    return upscaled.permute(0,2,3,1)

class SpatialEncoder(nn.Module):
  # Encodes an image into a latent vector, for use in PixelNeRF
  def __init__(
    self,
    latent_size: int =64,
    num_layers: int = 4,
  ):
    super().__init__()
    self.latent = None
    self.model = torchvision.models.resnet34(pretrained=True).eval()
    self.latent_size = latent_size
    self.num_layers = num_layers
  # (B, C = 3, H, W) -> (B, L, H, W)
  def forward(self, img):
    img = img.permute(0,3,1,2)
    l_sz = img.shape[2:]
    x = self.model.conv1(img)
    x = self.model.bn1(x)
    x = self.model.relu(x)
    latents = [x]
    # TODO other latent layers?

    ls = [F.interpolate(l, l_sz, mode="bilinear", align_corners=True) for l in latents]
    # necessary to detach here because we don't gradients to resnet
    # TODO maybe want latent gradients? Have to experiment with it.
    self.latents = torch.cat(latents, dim=1).detach().requires_grad_(False)
    return self.latents
  def sample(self, uvs: torch.Tensor, mode="bilinear", align_corners=True):
    assert(self.latents is not None), "Expected latents to be assigned in encoder"
    latents = F.grid_sample(
      self.latents,
      uvs.unsqueeze(0),
      mode=mode,
      align_corners=align_corners,
    )
    return latents.permute(0,2,3,1)

