import torch
import torch.nn as nn
import torch.nn.functional as F
import random

from .nerf import ( CommonNeRF, compute_pts_ts )
from .neural_blocks import ( SkipConnMLP )
from .utils import ( autograd, eikonal_loss )
from .refl import ( BasicReflectance )

def load(args):
  if args.sdf_kind == "spheres":
    model = SmoothedSpheres()
  elif args.sdf_kind == "siren":
    model = SIREN()
  else: raise NotImplementedError()
  # TODO need to add BSDF model and lighting here
  sdf = SDF(
    model,
    BasicReflectance(),
    t_near=args.near,
    t_far=args.far,
  )

  return sdf

class SDFModel(nn.Module):
  def __init__(self):
    super().__init__()
  def forward(self, _pts): raise NotImplementedError()
  def normals(self, pts, values = None):
    with torch.enable_grad():
      if not pts.requires_grad: autograd_pts = pts.requires_grad_()
      else: autograd_pts = pts

      if values is None: values = self(autograd_pts)
      normals = autograd(autograd_pts, values)
      assert(normals.isfinite().all())
    return normals

class SDF(nn.Module):
  def __init__(
    self,
    underlying: SDFModel,
    reflectance: "fn(x, dir) -> RGB",
    t_near: float,
    t_far: float,
    alpha:int = 100,
  ):
    super().__init__()
    assert(isinstance(underlying, SDFModel))
    self.underlying = underlying
    self.refl = reflectance
    self.far = t_far
    self.near = t_near
    self.alpha = alpha
  def forward(self, rays, with_throughput=True):
    r_o, r_d = rays.split([3,3], dim=-1)
    pts, hit, t = sphere_march(self.underlying, r_o, r_d, near=self.near, far=self.far)
    rgb = self.refl(pts, r_d)
    out = torch.where(hit, rgb, torch.zeros_like(rgb))
    if with_throughput:
      tput, _best_pos = throughput(self.underlying, r_o, r_d, self.far, self.near)
      out = torch.cat([out, -self.alpha*tput.unsqueeze(-1)], dim=-1)
    return out

@torch.jit.script
def smooth_min(v, k:float=32, dim:int=0):
  return -torch.exp(-k * v).sum(dim).clamp(min=1e-6).log()/k

class SmoothedSpheres(SDFModel):
  def __init__(
    self,
    n:int=32,
  ):
    super().__init__()
    self.centers = nn.Parameter(0.3 * torch.rand(n,3, requires_grad=True) - 0.15)
    self.radii = nn.Parameter(0.2 * torch.rand(n, requires_grad=True) - 0.1)

    self.tfs = nn.Parameter(torch.zeros(n, 3, 3, requires_grad=True))

  @torch.jit.export
  def transform(self, p):
    tfs = self.tfs + torch.eye(3, device=p.device).unsqueeze(0)
    return torch.einsum("ijk,ibk->ibj", tfs, p.expand(tfs.shape[0], -1, -1))
  def forward(self, p):
    q = self.transform(p.reshape(-1, 3).unsqueeze(0)) - self.centers.unsqueeze(1)
    sd = q.norm(p=2, dim=-1) - self.radii.unsqueeze(-1)
    out = smooth_min(sd, k=32.).reshape(p.shape[:-1])
    return out

def siren_act(v): return (3*v).sin()
class SIREN(SDFModel):
  def __init__(self):
    super().__init__()
    self.siren = SkipConnMLP(
      in_size=3, out=1, enc=None,
      hidden_size=96,
      activation=siren_act,
      # Do not have skip conns
      skip=1000,
    )
  def forward(self, x): return self.siren(x).squeeze(-1)

class SDFNeRF(nn.Module):
  def __init__(
    self,
    nerf: CommonNeRF,
    sdf: SDF,
  ):
    super().__init__()
    self.nerf = nerf
    self.sdf = sdf
    self.min_along_rays = None
  def forward(self, rays):
    pts, ts, r_o, r_d = compute_pts_ts(rays, self.nerf.t_near, self.nerf.t_far, self.nerf.steps)
    sdf_vals = self.sdf(pts)
    # record mins along rays for backprop
    self.min_along_rays = sdf_vals.min(dim=0)[0]
    # values (useful for density), normals (useful for view), latent (useful for ???)
    sdf_latent = torch.cat([sdf_vals, self.sdf.normals, self.sdf.mlp.last_layer_out], dim=-1)
    self.nerf.set_per_pt_latent(sdf_latent)
    return self.nerf.from_pts(pts, ts, r_o, r_d)
  @property
  def density(self): return self.nerf.density
  def render(self, rays):
    r_o, r_d = rays.split([3,3], dim=-1)
    pts, hits, ts = self.sdf.sphere_march(r_o, r_d, near=self.nerf.t_near, far=self.nerf.t_far)
    # TODO convert vals to some RGB value
    vals = torch.ones_like(pts)
    return torch.where(hits, vals, torch.zeros_like(vals))

# sphere_march is a traditional sphere marching algorithm on the SDF.
# It returns the (pts: R^3s, mask: bools, t: step along rays)
def sphere_march(
  self,
  r_o, r_d,
  iters: int = 32,
  eps: float = 5e-3,
  near: float = 0, far: float = 1,
):
  device = r_o.device
  with torch.no_grad():
    hits = torch.zeros(r_o.shape[:-1] + (1,), dtype=torch.bool, device=device)
    curr_dist = torch.full_like(hits, near, dtype=torch.float)
    for i in range(iters):
      curr = r_o + r_d * curr_dist
      dist = self(curr).reshape_as(curr_dist)
      hits = hits | ((dist < eps) & (curr_dist >= near) & (curr_dist <= far))
      curr_dist = torch.where(~hits, curr_dist + dist, curr_dist)
      # this saves some memory? but slows it down?
      #if hits.all(): break

  curr = r_o + r_d * curr_dist
  return curr, hits, curr_dist

def throughput(
  self,
  r_o, r_d,
  far: float,
  near: float,
  batch_size:int = 32,
):
  # some random jitter I guess?
  max_t = far+random.random()*(2/batch_size)
  step = max_t/batch_size
  with torch.no_grad():
    sd = self(r_o).squeeze(-1)
    curr_min = sd
    idxs = torch.zeros_like(sd, dtype=torch.long, device=r_d.device)
    for i in range(batch_size):
      t = step * (i+1)
      sd = self(r_o + t * r_d).squeeze(-1)
      idxs = torch.where(sd < curr_min, i+1, idxs)
      curr_min = torch.minimum(curr_min, sd)
  idxs = idxs.unsqueeze(-1)
  best_pos = r_o  + idxs * step * r_d
  return self(best_pos), best_pos


#@torch.jit.script
def masked_loss(img_loss=F.mse_loss):
  # masked loss takes some image loss, such as l2, l1 or ssim, and then applies it with an
  # additional loss on the alpha channel.
  def aux(
    # got and exp have 4 channels, where the last are got_mask and exp_mask
    got,
    exp,
    mask_weight:float=15,
  ):
    got, got_mask = got.split([3,1],dim=-1)
    exp, exp_mask = exp.split([3,1],dim=-1)
    active = ((got_mask > 0) & (exp_mask > 0)).squeeze(-1)
    misses = ~active

    color_loss = 0
    if active.any():
      got_active = got * active[..., None]
      exp_active = exp * active[..., None]
      color_loss = img_loss(got_active, exp_active)

    # this case is hit if the mask intersects nothing
    mask_loss = 0
    if misses.any():
      loss_fn = F.binary_cross_entropy_with_logits
      mask_loss = loss_fn(got_mask[misses].reshape(-1, 1), exp_mask[misses].reshape(-1, 1))
    return mask_weight * mask_loss + color_loss
  return aux
