import math
import numpy as np
import random
import torch
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

def save_image(name, img): plt.imsave(name, img.detach().cpu().clamp(0,1).numpy())

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

@torch.jit.script
def cartesian_to_log_polar(p, eps: float=1e-6):
  r = p.square().sum(keepdim=True, dim=-1).clamp(min=eps)
  x, y, z = p.split(1, dim=-1)
  phi = torch.atan2(nonzero_eps(y, eps), nonzero_eps(x, eps))
  theta = torch.atan2((x*x + y*y).clamp(min=eps).sqrt(), nonzero_eps(z, eps))
  # take the derivative w.r.t. the log in order to better differentiate small coordinates
  return torch.cat([r.log(), phi, theta], dim=-1)

pi = math.pi
two_pi = 2 * math.pi
# returns log polar indices, as well as local log polar coordinates
def log_polar_indices(lp, max_lr=15, n_lr=8, n_phi=8, n_theta=8):
  lr, phi, theta = lp.split(1, dim=-1)
  # anything below 0 will be in the first bucket since it's such a small region
  # anything above max_lr will be in the last bucket
  lr = lr.clamp(max=max_lr, min=0)
  phi = (phi + pi).clamp(max=two_pi, min=0)
  theta = (theta + pi).clamp(max=two_pi, min=0)
  lr_idx = (lr * (n_lr/max_lr)).clamp(max=n_lr)
  phi_idx = (phi * (n_phi/two_pi)).clamp(max=n_phi)
  theta_idx = (phi * (n_phi/two_pi)).clamp(max=n_phi)
  idx = torch.cat([lr_idx, phi_idx, theta_idx], dim=-1).floor().long() - 1

  local = torch.cat([
    lr - (lr_idx * max_lr/n_lr),
    phi - (phi_idx * pi/n_phi),
    theta - (theta_idx * pi/n_phi),
  ], dim=-1)
  return idx, local

def almost_identity(x, thresh=1e-1):
  x_abs = x.abs()
  t = x_abs/thresh
  v = (x_abs - thresh) * t * t + thresh
  return torch.where(x_abs > thresh, x, x.sign() * v)

DEF_BOUND = 3
# returns xyz index of at most n per dimension, as well as local coordinates in range [0, 1).
def cartesian_indices(xyz, bound: int =DEF_BOUND, n:int =8, bound2:int=DEF_BOUND*2):
  xyz = xyz.clamp(min=-bound, max=bound) + bound
  inv_region_size = (n-1)/bound2
  xyz_idx = (xyz * inv_region_size).long().clamp(max=n-1, min=0)
  local = xyz * inv_region_size - xyz_idx
  return xyz_idx, local


DEF_EPS = 1e-5
# returns xyz index of at most n per dimension, local coordinates in range [0, 1), and an
# averaging function for items near the edge.
def cartesian_indices_avgs(
  xyz,
  bound=DEF_BOUND, bound2=DEF_BOUND * 2, n=8, eps=DEF_EPS, epsm1=1-DEF_EPS,
):
  xyz = xyz.clamp(min=-bound, max=bound) + bound
  inv_region_size = n/bound2
  xyz_idx = (xyz * inv_region_size).clamp(max=n, min=0).long()
  local = xyz * inv_region_size - xyz_idx
  xyz_idx = xyz_idx - 1
  lo_mask = ((local < eps) & (xyz_idx > 0))
  hi_mask = ((local > epsm1) & (xyz_idx < n-1))
  #assert(not (lo_mask & hi_mask).any())
  mask = (lo_mask | hi_mask).any(dim=-1)

  avg_idx = xyz_idx + (hi_mask.long() - lo_mask.long())

  avg_local = lo_mask * epsm1 + hi_mask * eps

  extra_local = avg_local[mask, :]
  extra_idx = avg_idx[mask, :]

  def reaverage(out: ["batches", -1]):
    out, extra = out.split([ local.shape[0], extra_local.shape[0] ], dim=0)
    extra_buf = torch.zeros_like(mask, dtype=torch.float)
    extra_buf[mask] = extra.squeeze(-1)
    out = out.squeeze(-1) + extra_buf
    return torch.where(mask, out/2, out)
  return torch.cat([ xyz_idx, extra_idx ]), torch.cat([ local, extra_local ]), reaverage

# Tracks losses for a fixed number of samples and probabilistically samples from item with
# highest loss
class LossSampler():
  def __init__(self, N, default = 1e5, likelihood_inc=1.00001):
    self.losses = np.array([default] * N)
    # this increases likelihood of items which have not been seen in a while
    self.l_inc = likelihood_inc
  def update(self, idx, loss):
    self.losses *= self.l_inc
    self.losses[idx] = loss + 1
  def sample(self, n=1, replace=False):
    sqr_losses = self.losses * self.losses
    p = sqr_losses/sqr_losses.sum()
    return np.random.choice(len(self.losses), replace=replace, size=n, p=p)
  def update_idxs(self, idxs, loss):
    for idx in idxs: self.update(idx, loss)

# https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula
# c = cosine of theta, s = sine of theta
#@torch.jit.script
def rotate_vector(v, axis, c, s):
  return v * c \
         + axis * (v * axis).sum(dim=-1, keepdim=True) * (1-c) \
         + torch.cross(axis, v, dim=-1) * s

# https://github.com/facebookresearch/QuaterNet/blob/master/common/quaternion.py
def qmul(q, r):
    assert(q.shape[-1] == 4)
    assert(r.shape[-1] == 4)
    original_shape = q.shape

    # Compute outer product
    terms = torch.bmm(r.view(-1, 4, 1), q.view(-1, 1, 4))

    w = terms[:, 0, 0] - terms[:, 1, 1] - terms[:, 2, 2] - terms[:, 3, 3]
    x = terms[:, 0, 1] + terms[:, 1, 0] - terms[:, 2, 3] + terms[:, 3, 2]
    y = terms[:, 0, 2] + terms[:, 1, 3] + terms[:, 2, 0] - terms[:, 3, 1]
    z = terms[:, 0, 3] - terms[:, 1, 2] + terms[:, 2, 1] + terms[:, 3, 0]
    return torch.stack((w, x, y, z), dim=1).view(original_shape)

# http://paulbourke.net/geometry/rotate/
def quat_rot(v, axis, theta):
  q_1 = torch.cat([
    torch.zeros(axis.shape[:-1] + (1,), device=v.device), v,
  ], dim=-1)
  t_2 = (theta/2).expand(axis.shape[:-1] + (1,))
  q_2 = torch.cat([
    t_2.cos(),
    t_2.sin() * axis,
  ], dim=-1)
  q_2_inv = q_2
  q_2_inv[..., 1:] = -q_2_inv[..., 1:]
  out = qmul(qmul(q_2, q_1), q_2_inv)
  return out[..., 1:]

# bases of coordinate system, for finding normal to any vector
e_0 = torch.tensor([1,0,0], device="cuda", dtype=torch.float)
e_1 = torch.tensor([0,1,0], device="cuda", dtype=torch.float)
e_2 = torch.tensor([0,0,1], device="cuda", dtype=torch.float)
norm_eps = 1e-7
#@torch.jit.script
def param_rusin(n, wo, wi):
  n =  F.normalize(n, eps=norm_eps, dim=-1)
  wo = F.normalize(wo, eps=norm_eps, dim=-1)
  wi = F.normalize(wi, eps=norm_eps, dim=-1)
  # only run this on valid items (those which don't have all 0s in the last dimension)
  midway = F.normalize((n + e_2)/2, eps=norm_eps, dim=-1)

  wo = rotate_vector(wo, midway, -torch.ones_like(wo), torch.zeros_like(wo))
  wi = rotate_vector(wi, midway, -torch.ones_like(wi), torch.zeros_like(wi))

  # halfway vector between the two light directions
  H = F.normalize((wo + wi)/2, eps=norm_eps, dim=-1)
  #assert(H.isfinite().all()), H[~H.isfinite()]

  cos_theta_h = H[..., 2].unsqueeze(-1).clamp(min=-1, max=1)
  phi_h = torch.atan2(nonzero_eps(H[..., 1]), nonzero_eps(H[..., 0]))

  binormal = e_1

  v = -phi_h.unsqueeze(-1)
  tmp = F.normalize(rotate_vector(wi, n, v.cos(), v.sin()), eps=norm_eps, dim=-1)

  # this is actually sin(-theta_h) = -sin(theta_h)
  # and since cos(-theta_h) = cos(-theta_h), that can be ignored
  sin_theta_h = -(1-cos_theta_h.square()).clamp(min=1e-6).sqrt()
  diff = F.normalize(
    rotate_vector(tmp, binormal.expand(tmp.shape), cos_theta_h, sin_theta_h),
    eps=norm_eps,
    dim=-1,
  )

  # clamp to remove NaN, it probably breaks something somewhere
  # Instead we can just use the raw value since it's more numerically stable and 1-1
  cos_theta_d = diff[..., 2]#.clamp(min=-0.95, max=0.95).acos()
  #phi_d = torch.fmod(torch.atan2(nonzero_eps(diff[..., 1]), nonzero_eps(diff[..., 0])), math.pi)
  phi_d = torch.atan2(nonzero_eps(diff[..., 1]), nonzero_eps(diff[..., 0]))

  return torch.stack([phi_d, cos_theta_h.squeeze(-1), cos_theta_d], dim=-1)

# assumes wo and wi are already in local coordinates
@torch.jit.script
def param_rusin2(wo, wi):
  wo = F.normalize(wo, dim=-1)
  wi = F.normalize(wi, dim=-1)
  e_1 = torch.tensor([0,1,0], device=wo.device, dtype=torch.float).expand_as(wo)
  e_2 = torch.tensor([0,0,1], device=wo.device, dtype=torch.float).expand_as(wo)

  H = F.normalize((wo + wi), dim=-1)

  cos_theta_h = H[..., 2]
  phi_h = torch.atan2(nonzero_eps(H[..., 1]), nonzero_eps(H[..., 0]))

  # v = -phi_h.unsqueeze(-1)
  r = nonzero_eps(H[..., 1]).hypot(nonzero_eps(H[..., 0])).clamp(min=1e-6)
  c = (H[..., 0]/r).unsqueeze(-1)
  s = -(H[..., 1]/r).unsqueeze(-1)
  tmp = F.normalize(rotate_vector(wi, e_2, c, s), dim=-1)
  #v = -theta_h.unsqueeze(-1)
  c = H[..., 2].unsqueeze(-1)
  s = -(1 - H[..., 2]).clamp(min=1e-6).sqrt().unsqueeze(-1)
  diff = F.normalize(rotate_vector(tmp, e_1, c, s), dim=-1)
  cos_theta_d = diff[..., 2]
  # rather than doing fmod, try cos to see if it can capture cyclicity better.
  cos_phi_d = torch.atan2(nonzero_eps(diff[..., 1]), nonzero_eps(diff[..., 0])).cos()

  return torch.stack([cos_phi_d, cos_theta_h, cos_theta_d], dim=-1)

# This is for converting a number in the range [-1, 1] to a range which is approximately
# infinite, based on eps. eps != 0 for numerical stability at the edges.
def inverse_tan_activation(x: [-1, 1], eps=1e-1):
  return torch.tan(x * math.pi/(2 + eps))

def gaussian_kernel(n, sigma=3):
  # move this import here because it breaks in some envs
  from scipy.ndimage import gaussian_filter
  kernel = np.zeros((2*n+1,2*n+1))
  kernel[n,n] = 1
  return torch.from_numpy(gaussian_filter(kernel, sigma=sigma))

@torch.jit.script
def weak_sigmoid(x, k:float=4e-4, sqrt_k:float=2e-2, eps:float=1e-6):
  return torch.where(x.abs() < k,
    x/sqrt_k,
    x.sign() * (x.abs() + eps).sqrt(),
  )

# positive symmetric weak sigmoid
def pos_weak_sigmoid(x, k=4e-4, k_pow_3_2=2e-2, eps=1e-5):
  return torch.where(x.abs() <= k, x.square()/k_pow_3_2, (x.abs() + eps).sqrt())

# Compute some very close rays for some input ray, which are different in orthogonal dimensions.
def finite_diff_ray(r_d, eps=1e-5):
  x, y, z = r_d.split(1, dim=-1)
  sign = z.sign()
  a = (-(sign + z) + 1e-5).reciprocal()
  b = x * y * a

  s = torch.cat([(x * x * a * sign) + 1, b * sign, x * -sign], dim=-1)
  t = torch.cat([b, sign + y * y * a, -y], dim=-1)
  return r_d + s * eps, r_d + t * eps

@torch.jit.script
def eikonal_loss(grad): return (torch.norm(grad, dim=-1) - 1).square().mean()

@torch.jit.script
def edge_detection(tensor):
  kernel = torch.tensor([[
    [-1, -1, -1],
    [-1,  8, -1],
    [-1, -1, -1],
  ]], dtype=torch.float, device=tensor.device).expand(3, 3, 3)
  return F.conv2d(tensor.permute(2,1,0).unsqueeze(0), kernel.unsqueeze(0), stride=1).squeeze(0)

#@torch.jit.script
def masked_loss(
  got, exp,
  throughput, exp_mask,
  eps:float=1e-10, trim:int=0, mask_weight:float=1,
  with_logits: bool=True,
  tone_mapping: bool = False,
):
  active = ((throughput > 0) & (exp_mask == 1)).squeeze(-1)
  misses = ~active

  color_loss = 0
  if active.any():
    if tone_mapping:
      got_active = got * active[..., None]
      exp_active = exp * active[..., None]
      got_active = (got_active)/(1+got_active)
      exp_active = (exp_active)/(1+exp_active)
      l1_loss = F.l1_loss(got_active, exp_active)
      l2_loss = F.mse_loss(got_active, exp_active)
      rmse_loss = l2_loss.clamp(min=1e-10).sqrt()

      ssim_loss = -ssim(
        got_active.permute(0,3,1,2),
        exp_active.permute(0,3,1,2),
        data_range=1,
        size_average=True
      ).log()

      color_loss = l2_loss + rmse_loss + l1_loss + ssim_loss
    else:
      got_active = got * active[..., None]
      exp_active = exp * active[..., None]
      l1_loss = F.l1_loss(got_active, exp_active)
      l2_loss = F.mse_loss(got_active, exp_active)
      rmse_loss = l2_loss.clamp(min=1e-10).sqrt()
      ssim_loss = -ssim(
        got_active.permute(0,3,1,2),
        exp_active.permute(0,3,1,2),
        data_range=1,
        size_average=True
      ).log()
      color_loss = l2_loss + rmse_loss + l1_loss + ssim_loss

  # This case is hit if the mask intersects nothing
  mask_loss = 0
  if misses.any():
    loss_fn = F.binary_cross_entropy
    if with_logits: loss_fn = F.binary_cross_entropy_with_logits
    mask_loss = loss_fn(
      throughput[misses].reshape(-1, 1), exp_mask[misses].reshape(-1, 1),
    )
  out = mask_weight * mask_loss + 10*color_loss
  return out

def mse2psnr(x): return -10 * torch.log10(x)

def count_parameters(params): return sum(p.numel() for p in params)

def load_image(src, resize=None):
  img = Image.open(src)
  if resize is not None:
    img = img.resize(resize)
  return torch.from_numpy(np.array(img, dtype=float)/255).float()

# crops an image with top left corner u, v, and first two dimensions X, Y
def crop(img, u, v, size): return img[u:u+size,v:v+size,...]
# returns a random u,v (integer)

def rand_uv(w: int, h: int, size: int):
  return random.randint(0, w-size), random.randint(0, h-size)

def rand_uv_mask(mask, size:int):
  half_size = int(math.ceil(size/2))
  valid = mask[half_size:-half_size-size, half_size:-half_size-size, ...]
  (p, q) = valid.nonzero(as_tuple=True)
  idx = random.randint(0, len(p)-1)
  return p[idx], q[idx]

@torch.jit.script
def smooth_min(v, k:float=32, dim:int=0):
  return -torch.exp(-k * v).sum(dim).clamp(min=1e-4).log()/k

def sphere_render_bsdf(bsdf,integrator=None,device="cuda", size=256, chunk_size=128, scale=100):
  from pytorch3d.pathtracer.shapes import Sphere
  from pytorch3d.pathtracer.bsdf import Diffuse
  from pytorch3d.pathtracer import pathtrace
  from pytorch3d.renderer import (
    look_at_view_transform, OpenGLPerspectiveCameras, PointLights,
  )
  import pytorch3d.pathtracer.integrators as integrators
  sphere = Sphere([0,0,0], 1, device=device)
  R, T = look_at_view_transform(dist=2., elev=0, azim=0)
  cameras = OpenGLPerspectiveCameras(device=device, R=R, T=T)
  lights = PointLights(device=device, location=[[0., 1., 4.]], scale=scale)
  if integrator is None:
    integrator = integrators.Direct()
  return pathtrace(
    sphere, cameras=cameras, lights=lights, chunk_size=chunk_size, size=size,
    bsdf=bsdf, integrator=integrator,
    device=device, silent=True,
  )[0]

def sphere_examples(bsdf, device="cuda", size=256, chunk_size=128, scale=100):
  from pytorch3d.pathtracer.shapes import Sphere
  from pytorch3d.pathtracer.bsdf import Diffuse
  from pytorch3d.pathtracer import pathtrace
  from pytorch3d.renderer import (
    look_at_view_transform, OpenGLPerspectiveCameras, PointLights,
  )
  import pytorch3d.pathtracer.integrators as integrators
  sphere = Sphere([0,0,0], 1, device=device)
  R, T = look_at_view_transform(dist=2., elev=0, azim=0)
  cameras = OpenGLPerspectiveCameras(device=device, R=R, T=T)
  lights = PointLights(device=device, location=[[0., 1., 4.]], scale=scale)
  out = []

  for basis in bsdf.bsdfs:
    expected = pathtrace(
      sphere, cameras=cameras, lights=lights,
      chunk_size=chunk_size, size=size,
      bsdf=basis, integrator=integrators.Direct(),
      device=device, silent=True,
    )[0]
    out.append(expected)
  return out

# draws a heightmap of a given warp using u,v in [0,1] coordinates at a given granularity
def heightmap(warp, size=256, device="cuda"):
  u,v = torch.meshgrid(
    torch.linspace(0, 1, size, device=device),
    torch.linspace(0, 1, size, device=device),
  )
  return warp.pdf(torch.stack([u,v],dim=-1))

@torch.jit.script
def depth_image(img):
  l, m = img.split(1, dim=-1)
  l = (l/l.max())
  return torch.cat([l, l, l, m], dim=-1)


#@torch.jit.script
def fwidth(v):
  n, w, h, b, _ = v.shape
  dx = v[:, 1:, ...] - v[:, :-1, ...]
  dx = torch.cat([
    dx, dx[:, -1:, ...],
  ], dim=1)
  dy = v[:, :, 1:, ...] - v[:, :, :-1, ...]
  dy = torch.cat([
    dy, dy[:, :, -1:, ...]
  ], dim=2)
  return dx.abs() + dy.abs()

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
  x, y, z = F.normalize(direc, dim=-1).clamp(min=-1+1e-7, max=1-1e-7).split(1, dim=-1)
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




