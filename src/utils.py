import math
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from pytorch_msssim import ( ssim, ms_ssim )
import matplotlib.pyplot as plt

def create_fourier_basis(batch_size, features=3, freq=40, device="cuda"):
  B = freq * torch.randn(batch_size, features, device=device).T
  out_size = batch_size * 2 + features
  return B, out_size

@torch.jit.script
def fourier(x, B):
  mapped = x @ B
  return torch.cat([x, mapped.sin(), mapped.cos()], dim=-1)

@torch.jit.script
def expected_sin(x, x_var):
  y = (-0.5 * x_var).exp() * x.sin()
  y_var = (0.5 * (1 - (-2 * x_var).exp() * (2 * x).cos()) - y.square()).clamp(min=0)
  return y, y_var

def integrated_pos_enc_diag(x, x_cov, min_deg:int, max_deg:int):
  scales = torch.exp2(torch.arange(min_deg, max_deg, device=x.device, dtype=x.dtype))
  out_shape = x.shape[:-1] + (-1,)
  y = (x[..., None, :] * scales[..., None]).reshape(out_shape)
  y_var = (x_cov[..., None, :] * scales[..., None].square()).reshape(out_shape)
  return expected_sin(
    torch.cat([y, y + 0.5 * math.pi], dim=-1),
    torch.cat([y_var, y_var], dim=-1),
  )[0]

#@torch.jit.script
def lift_gaussian(r_d, t_mean, t_var, r_var):
  mean = r_d[..., None] * t_mean[..., None, :]

  magn_sq = r_d.square().sum(dim=-1, keepdim=True).clamp(min=1e-10)
  outer_diag = r_d.square()
  null_outer_diag = 1 - outer_diag / magn_sq

  t_cov_diag = t_var[..., None] * outer_diag[..., None, :]
  xy_cov_diag = r_var[..., None] * null_outer_diag[..., None, :]
  cov_diag = t_cov_diag + xy_cov_diag

  # the movedim moves the time dimension to the front
  return mean.movedim(-1, 0), cov_diag.movedim(-1, 0)


# Computes radius along the x-axis
@torch.jit.script
def radii_x(r_d):
  dx = (r_d[..., :-1, :, :] - r_d[..., 1:, :, :]).square().sum(dim=-1).sqrt()
  dx = torch.cat([dx, dx[:, -2:-1, :]], dim=-2)

  return dx[..., None] * 2 / math.sqrt(12)

def conical_frustrum_to_gaussian(r_d, t0, t1, rad:float):
  mu = (far + near) / 2
  hw = (far - near) / 2
  mu2 = mu * mu
  hw2 = hw * hw
  t_mean = mu + (2 * mu * hw2) / (3 * mu2 + hw2)
  t_var = hw / 3 - (4 / 15) * ((hw2*hw2 * (12 * mu2 - hw2)) / (3 * mu2 + hw2).square())
  r_var = rad*rad * (mu2 / 4 + (5 / 12) * hw2 - 4 / 15 * (hw2*hw2) / (3 * mu2 + hw2))

  return lift_gaussian(r_d, t_mean, t_var, r_var)

def cylinder_to_gaussian(r_d, t0, t1, rad:float):
  t_mean = (t1 + t0) / 2
  r_var = rad * rad / 4
  t_var = (t1 - t0).square() / 12

  return lift_gaussian(r_d, t_mean, t_var, r_var)

class CylinderGaussian(nn.Module):
  def __init__(
    self,
    min_deg: int = 0,
    max_deg: int = 16,
  ):
    super().__init__()
    self.min_deg = min_deg
    self.max_deg = max_deg
  def size(self): return self.max_deg - self.min_deg
  def forward(self, r_o, r_d, t0, t1):
    rad = radii_x(r_d)
    mean, cov = cylinder_to_gaussian(r_d, t0, t1, rad)
    mean = mean + r_o
    return integrated_pos_enc_diag(mean, cov, self.min_deg, self.max_deg)

# TODO integrated pos enc w/o diag? It's never used so don't need to have it

def save_image(name, img): plt.imsave(name, img.detach().cpu().clamp(0,1).numpy())
def save_plot(name, expected, got):
  fig = plt.figure()
  fig.add_subplot(1, 2, 1)
  plt.imshow(got.detach().squeeze().cpu().numpy())
  plt.grid("off");
  plt.axis("off");
  fig.add_subplot(1, 2, 2)
  plt.imshow(expected.detach().squeeze().cpu().numpy())
  plt.grid("off");
  plt.axis("off");
  plt.savefig(name)
  plt.close(fig)

# https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula
# c = cosine of theta, s = sine of theta
#@torch.jit.script
def rotate_vector(v, axis, c, s):
  return v * c \
         + axis * (v * axis).sum(dim=-1, keepdim=True) * (1-c) \
         + torch.cross(axis, v, dim=-1) * s

#@torch.jit.script
def nonzero_eps(v, eps: float=1e-7):
  # in theory should also be copysign of eps, but so small it doesn't matter
  # and torch.jit.script doesn't support it
  return torch.where(
    v.abs() < eps,
    torch.tensor(eps, device=v.device),
    #torch.copysign(torch.full_like(v, eps, device=v.device), v),
    v
  )


def mse2psnr(x): return -10 * torch.log10(x)

def count_parameters(params): return sum(p.numel() for p in params)

def load_image(src, resize=None):
  img = Image.open(src)
  if resize is not None:
    img = img.resize(resize)
  return torch.from_numpy(np.array(img, dtype=float)/255).float()

# [-1, 1] -> [-pi/2, pi/2]
#@torch.jit.script
def uv_to_elev_azim(uv):
  u, v = uv.clamp(min=-1+1e-7, max=1-1e-7).split(1, dim=-1)
  elev = v.asin()
  azim = torch.atan2(u, (1 - u.square() - v.square()).clamp(min=1e-8).sqrt())
  return torch.cat([elev, azim], dim=-1)

#@torch.jit.script
def elev_azim_to_uv(elev_azim):
  elev, azim = elev_azim.split(1, dim=-1)
  u = elev.cos() * azim.sin()
  v = elev.sin()
  return torch.cat([u, v], dim=-1)

#[-pi, pi]^2 -> [0,1]^3
#@torch.jit.script
def elev_azim_to_dir(elev_azim):
  limit = math.pi-1e-7
  elev, azim = elev_azim.clamp(min=-limit, max=limit).split(1, dim=-1)
  direction = torch.cat([
    azim.sin() * elev.cos(),
    azim.cos() * elev.cos(),
    elev.sin(),
  ], dim=-1)
  return direction


#@torch.jit.script
def dir_to_elev_azim(direc):
  x, y, z = F.normalize(direc, dim=-1).clamp(min=-1+1e-10, max=1-1e-10).split([1,1,1], dim=-1)
  elev = z.asin()
  azim = torch.atan2(x, (1 - x.square() - z.square()).clamp(min=1e-10).sqrt())
  return torch.cat([elev, azim], dim=-1)

# [-1, 1]x2 -> [-1, 1]x3 (direction) sum (component of dir)^2 = 1
#@torch.jit.script
def uv_to_dir(uv): return elev_azim_to_dir(uv_to_elev_azim(uv))

#@torch.jit.script
def dir_to_uv(d):
  elaz = dir_to_elev_azim(d)
  return elev_azim_to_uv(elaz)


# gets a set of spherical transforms about the origin
def spherical_positions(
  min_elev=0, max_elev=45, min_azim=-135, max_azim=135,

  n_elev:int=8, n_azim:int=8, dist=1, device="cuda",
):
  from pytorch3d.renderer import look_at_view_transform
  Rs, Ts = [], []
  for elev in torch.linspace(min_elev, max_elev, n_elev):
    for azim in torch.linspace(min_azim, max_azim, n_azim):
      R,T = look_at_view_transform(dist=dist, elev=elev, azim=azim, device=device)
      Rs.append(R)
      Ts.append(T)
  return torch.cat(Rs, dim=0), torch.cat(Ts, dim=0)




