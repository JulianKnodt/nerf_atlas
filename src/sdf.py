import torch
import torch.nn as nn
import torch.nn.functional as F
import random

from .nerf import ( CommonNeRF, compute_pts_ts )
from .neural_blocks import ( SkipConnMLP, FourierEncoder )
from .utils import ( autograd, eikonal_loss )
import src.refl as refl

def load(args):
  if args.sdf_kind == "spheres": model = SmoothedSpheres()
  elif args.sdf_kind == "siren": model = SIREN()
  elif args.sdf_kind == "local": model = Local()
  elif args.sdf_kind == "mlp": model = MLP()
  else: raise NotImplementedError()
  # refl inst may also have a nested light
  refl_inst = refl.load(args, model.latent_size)

  sdf = SDF(
    model, refl_inst,
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
      #assert(normals.isfinite().all())
    return normals

class SDF(nn.Module):
  def __init__(
    self,
    underlying: SDFModel,
    reflectance: refl.Reflectance,
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

  @property
  def sdf(self): return self

  def normals(self, pts, values = None): return self.underlying.normals(pts, values)
  def from_pts(self, pts):
    raw = self.underlying(pts)
    latent = raw[..., 1:]
    return raw[..., 0], latent if latent.shape[-1] != 0 else None

  def intersect_w_n(self, r_o, r_d):
    pts, hit, t = sphere_march(
      self.underlying, r_o, r_d, near=self.near, far=self.far,
      iters=32 if self.training else 64,
    )
    return pts, hit, t, self.normals(pts)
  def intersect_mask(self, r_o, r_d):
    with torch.no_grad():
      return sphere_march(
        self.underlying, r_o, r_d, near=self.near, far=self.far,
        # since this is just for intersection, better to use fewer steps
        iters=16 if self.training else 64,
      )[1]
  def forward(self, rays, with_throughput=True):
    r_o, r_d = rays.split([3,3], dim=-1)
    pts, hit, t = sphere_march(
      self.underlying, r_o, r_d, near=self.near, far=self.far,
      iters=32 if self.training else 64,
    )
    rgb = self.refl(
      x=pts, view=r_d,
      normal=self.normals(pts) if self.refl.can_use_normal else None,
    )
    out = torch.where(hit, rgb, torch.zeros_like(rgb))
    if with_throughput and self.training:
      tput, _best_pos = throughput(self.underlying, r_o, r_d, self.far, self.near)
      out = torch.cat([out, -self.alpha*tput.unsqueeze(-1)], dim=-1)
    return out

#@torch.jit.script
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

  @property
  def latent_size(self): return 0

  def forward(self, p):
    q = self.transform(p.reshape(-1, 3).unsqueeze(0)) - self.centers.unsqueeze(1)
    sd = q.norm(p=2, dim=-1) - self.radii.unsqueeze(-1)
    out = smooth_min(sd, k=32.).reshape(p.shape[:-1] + (1,))
    return out

class MLP(SDFModel):
  def __init__(
    self,
    latent_size:int=32,
  ):
    super().__init__()
    self.mlp = SkipConnMLP(
      in_size=3, out=1+latent_size,
      enc=FourierEncoder(input_dims=3),
      num_layers=6, hidden_size=256,
      xavier_init=True,
    )
    self.latent_size = latent_size
  def forward(self, x): return self.mlp(x)

#def siren_act(v): return (30*v).sin()
class SIREN(SDFModel):
  def __init__(
    self,
    latent_size:int=32,
  ):
    super().__init__()
    self.siren = SkipConnMLP(
      in_size=3, out=1+latent_size, enc=None,
      hidden_size=96,
      activation=torch.sin,
      # Do not have skip conns
      skip=1000,
    )
    self.latent_size = latent_size
  def forward(self, x):
    out = self.siren((30*x).sin())
    assert(out.isfinite().all())
    return out

class Local(SDFModel):
  def __init__(
    self,
    partition_sz: int = 0.5,
    latent_sz:int = 32,
  ):
    super().__init__()
    self.part_sz = partition_sz
    self.latent = SkipConnMLP(in_size=3,out=latent_sz,skip=4)
    self.tform = SkipConnMLP(in_size=3, out=1, latent_size=latent_sz)
  def forward(self, x):
    local = x % self.part_sz
    latent = self.latent(x//self.part_sz)
    out = self.tform(local, latent)
    return out

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
      dist = self(curr)[...,0].reshape_as(curr_dist)
      hits = hits | ((dist < eps) & (curr_dist >= near) & (curr_dist <= far))
      curr_dist = torch.where(~hits, curr_dist + dist, curr_dist)

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
    sd = self(r_o)[...,0]
    curr_min = sd
    idxs = torch.zeros_like(sd, dtype=torch.long, device=r_d.device)
    for i in range(batch_size):
      t = step * (i+1)
      sd = self(r_o + t * r_d)[..., 0]
      idxs = torch.where(sd < curr_min, i+1, idxs)
      curr_min = torch.minimum(curr_min, sd)
  idxs = idxs.unsqueeze(-1)
  best_pos = r_o  + idxs * step * r_d
  return self(best_pos)[...,0], best_pos


#@torch.jit.script
def masked_loss(img_loss=F.mse_loss):
  # masked loss takes some image loss, such as l2, l1 or ssim, and then applies it with an
  # additional loss on the alpha channel.
  def aux(
    # got and exp have 4 channels, where the last are got_mask and exp_mask
    got, exp,
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
