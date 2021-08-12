import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random

from .nerf import ( CommonNeRF, compute_pts_ts )
from .neural_blocks import ( SkipConnMLP, FourierEncoder, NNEncoder )
from .utils import ( autograd, eikonal_loss )
import src.refl as refl
import src.renderers as renderers
from tqdm import trange

def load(args):
  if args.sdf_kind == "spheres": cons = SmoothedSpheres
  elif args.sdf_kind == "siren": cons = SIREN
  elif args.sdf_kind == "local": cons = Local
  elif args.sdf_kind == "mlp": cons = MLP
  elif args.sdf_kind == "triangles": cons = Triangles
  else: raise NotImplementedError(f"Unknown SDF kind: {args.sdf_kind}")

  model = cons(latent_size=args.latent_size)

  if args.sphere_init: model.set_to_sphere()

  if args.bound_sphere_rad > 0: model = UnitSphere(inner=model,rad=args.bound_sphere_rad)
  # refl inst may also have a nested light
  refl_inst = refl.load(args, model.latent_size)

  sdf = SDF(
    model, refl_inst,
    t_near=args.near,
    t_far=args.far,
  )
  if args.integrator_kind is not None:
    return renderers.load(args, sdf, refl_inst)

  return sdf

class SDFModel(nn.Module):
  def __init__(
    self,
    latent_size: int = 32,
  ):
    super().__init__()
    self.latent_size = latent_size
  def forward(self, _pts): raise NotImplementedError()

  def normals(self, pts, values = None):
    with torch.enable_grad():
      if not pts.requires_grad: autograd_pts = pts.requires_grad_()
      else: autograd_pts = pts

      if values is None: values = self(autograd_pts)
      normals = autograd(autograd_pts, values)
    return normals
  # will optimize this SDF to be a sphere at the start
  def set_to_sphere(self, rad:float = 0.5, iters:int=1000):
    opt = optim.Adam(self.parameters(), lr=5e-5, weight_decay=0)
    t = trange(iters)
    for i in t:
      opt.zero_grad()
      v = 4*torch.randn(5000, 3)
      got = self(v)[...,0]
      exp = v.norm(dim=-1) - rad
      loss = F.mse_loss(got, exp)
      t.set_postfix(l=loss.item())
      loss.backward()
      opt.step()

# Wraps another SDF as the intersection of a sphere centered at the origin.
class UnitSphere(SDFModel):
  def __init__(
    self,
    inner: SDFModel,
    rad:float=3,
  ):
    super().__init__(latent_size=inner.latent_size)
    self.inner = inner
    self.rad = rad

  def forward(self, pts):
    sph = torch.linalg.norm(pts, dim=-1, ord=2) - self.rad
    inner = self.inner(pts)
    return torch.cat([
      torch.maximum(inner[..., 0], sph).unsqueeze(-1),
      inner[..., 1:],
    ], dim=-1)

class SDF(nn.Module):
  def __init__(
    self,
    underlying: SDFModel,
    reflectance: refl.Reflectance,
    t_near: float,
    t_far: float,
    alpha:int = 1000,
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

  @property
  def latent_size(self): return self.underlying.latent_size

  def normals(self, pts, values = None): return self.underlying.normals(pts, values)
  def from_pts(self, pts):
    raw = self.underlying(pts)
    latent = raw[..., 1:]
    return raw[..., 0], latent if latent.shape[-1] != 0 else None

  def intersect_w_n(self, r_o, r_d):
    pts, hit, t = sphere_march(
      self.underlying, r_o, r_d, near=self.near, far=self.far,
      iters=128 if self.training else 256,
    )
    return pts, hit, t, self.normals(pts)
  def intersect_mask(self, r_o, r_d, near=None, far=None, eps=1e-3):
    with torch.no_grad():
      return ~sphere_march(
        self.underlying, r_o, r_d, eps=eps,
        near=self.near if near is None else near,
        far=self.far if far is None else far,
        # since this is just for intersection, alright to use fewer steps
        iters=64 if self.training else 128,
      )[1]
  def forward(self, rays, with_throughput=True):
    r_o, r_d = rays.split([3,3], dim=-1)
    pts, hit, t = sphere_march(
      self.underlying, r_o, r_d, near=self.near, far=self.far,
      iters=128 if self.training else 192,
    )
    latent = None if self.latent_size == 0 else self.underlying(pts[hit])[..., 1:]
    out = torch.zeros_like(r_d)
    n = None
    if self.refl.can_use_normal:
      self.n = torch.zeros_like(out)
      n = self.normals(pts[hit])
      self.n[hit] = n
    # use masking in order to speed up efficiency
    out[hit] = self.refl(
      x=pts[hit], view=r_d[hit], normal=n,
      latent=latent, mask=hit,
    )
    if with_throughput and self.training:
      out = torch.cat([out, self.throughput(r_o, r_d)], dim=-1)
    return out
  def debug_normals(self, rays):
    r_o, r_d = rays.split([3,3], dim=-1)
    pts, hit, t = sphere_march(
      self.underlying, r_o, r_d, near=self.near, far=self.far,
      iters=128 if self.training else 192,
    )
    latent = None if self.underlying.latent_size == 0 else self.underlying(pts[hit])[..., 1:]
    out = torch.zeros_like(r_d)
    out[hit] = self.normals(pts[hit])
    return out
  def throughput(self, r_o, r_d):
    tput, _best_pos = throughput(self.underlying, r_o, r_d, self.near, self.far)
    return -self.alpha*tput.unsqueeze(-1)

#@torch.jit.script
def smooth_min(v, k:float=32, dim:int=0):
  return -torch.exp(-k * v).sum(dim).clamp(min=1e-6).log()/k

class SmoothedSpheres(SDFModel):
  def __init__(
    self,
    n:int=128,

    with_mlp=True,
    **kwargs,
  ):
    super().__init__(**kwargs)
    # has no latent size
    self.latent_size = 0

    self.centers = nn.Parameter(0.3 * torch.rand(n,3, requires_grad=True) - 0.15)
    self.radii = nn.Parameter(0.2 * torch.rand(n, requires_grad=True) - 0.1)

    self.tfs = nn.Parameter(torch.zeros(n, 3, 3, requires_grad=True))
    if with_mlp:
      self.mlp = SkipConnMLP(
        in_size=3, out=1,
        num_layers=5, hidden_size=128,
        enc=FourierEncoder(input_dims=3),
        xavier_init=True,
      )

  @torch.jit.export
  def transform(self, p):
    tfs = self.tfs + torch.eye(3, device=p.device).unsqueeze(0)
    return torch.einsum("ijk,ibk->ibj", tfs, p.expand(tfs.shape[0], -1, -1))

  def forward(self, p):
    q = self.transform(p.reshape(-1, 3).unsqueeze(0)) - self.centers.unsqueeze(1)
    sd = q.norm(p=2, dim=-1) - self.radii.unsqueeze(-1)
    out = smooth_min(sd, k=32.).reshape(p.shape[:-1] + (1,))
    if hasattr(self, "mlp"): out = out + self.mlp(p).tanh() * (1-out.sigmoid())
    return out


def dot(a,b, dim:int=-1, keepdim:bool=False): return (a * b).sum(dim=dim,keepdim=keepdim)
# dot2(a) = dot(a,a)
def dot2(a, dim:int=-1, keepdim:bool=False): return dot(a,a,dim=dim,keepdim=keepdim)

# triangle is similar to spheres but contains triangles instead
class Triangles(SDFModel):
  def __init__(
    self,
    n:int=32,

    **kwargs,
  ):
    super().__init__(**kwargs)
    # has no latent size
    self.latent_size = 0

    # each triangle requires 3 points
    self.points = nn.Parameter(0.3 * torch.rand(n,3,3, requires_grad=True) - 0.15)
    #self.thickness = nn.Parameter(0.3 * torch.rand(n, requires_grad=True) - 0.15)

  def forward(self, p):
    pa,pb,pc = (p.reshape(-1, 1, 1, 3) - self.points).split([1,1,1],dim=-2)
    ac,ba,cb = (self.points - self.points.roll(1, dims=-2)).split([1,1,1], dim=-2)
    nor = torch.cross(ba, ac, dim=-1)

    sidedness = \
      dot(torch.cross(ba, nor), pa, dim=-1).sign() + \
      dot(torch.cross(cb, nor), pb, dim=-1).sign() + \
      dot(torch.cross(ac, nor), pc, dim=-1).sign()


    same_sided = dot2(ba*(dot(ba, pa,keepdim=True)/dot2(ba,keepdim=True)).clamp(min=0,max=1)-pa)\
        .minimum(dot2(cb*(dot(cb, pb,keepdim=True)/dot2(cb,keepdim=True)).clamp(min=0,max=1)-pb))\
        .minimum(dot2(ac*(dot(ac, pc,keepdim=True)/dot2(ac,keepdim=True)).clamp(min=0,max=1)-pc))

    opp_sided = dot(nor, pa,dim=-1).square()/dot(nor,nor, dim=-1)
    out = torch.where(sidedness < 2, same_sided, opp_sided).clamp(min=1e-8).sqrt()
    # need to add a smooth min across all the triangles as well as an extrusion
    # apply thickness to each triangle to allow certain ones to take up more space.
    out = out.squeeze(-1) - 4e-2
    # smooth min or just normal union?
    out = smooth_min(out,dim=-1).reshape(p.shape[:-1] + (1,))
    #if out.numel() == 0: return out.reshape(p.shape[:-1] + (1,))
    #out = out.min(dim=-1)[0].reshape(p.shape[:-1] + (1,))
    return out

class MLP(SDFModel):
  def __init__(
    self,
    **kwargs,
  ):
    super().__init__(**kwargs)
    self.mlp = SkipConnMLP(
      in_size=3, out=1+self.latent_size,
      enc=FourierEncoder(input_dims=3),
      num_layers=6, hidden_size=256,
      xavier_init=True,
    )
  def forward(self, x): return self.mlp(x)

#def siren_act(v): return (30*v).sin()
class SIREN(SDFModel):
  def __init__(
    self,
    **kwargs,
  ):
    super().__init__(**kwargs)
    self.siren = SkipConnMLP(
      in_size=3, out=1+self.latent_size, enc=None,
      num_layers=8, hidden_size=256,
      activation=torch.sin,
      # Do not have skip conns
      skip=1000,
    )
  def forward(self, x):
    out = self.siren((30*x).sin())
    assert(out.isfinite().all())
    return out

# TODO better verify this works? Haven't tried it out a ton on a lot of methods
class Local(SDFModel):
  def __init__(
    self,
    partition_sz: int = 0.5,
    **kwargs,
  ):
    super().__init__(**kwargs)
    self.part_sz = partition_sz
    self.latent = SkipConnMLP(in_size=3,out=self.latent_size,skip=4)
    self.tform = SkipConnMLP(
      in_size=3, out=1+self.latent_size, latent_size=self.latent_size,
      enc=NNEncoder(input_dims=3),
    )
  def forward(self, x):
    local = x % self.part_sz
    latent = self.latent(x/self.part_sz)
    return self.tform(local, latent)

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
    raise NotImplementedError()
    return torch.where(hits, vals, torch.zeros_like(vals))

# sphere_march is a traditional sphere marching algorithm on the SDF.
# It returns the (pts: R^3s, mask: bools, t: step along rays)
#
# note that this implementation is efficient in that it only will compute distance
# for pts that are still candidates.
def sphere_march(
  self,
  r_o, r_d,
  iters: int = 32,
  eps: float = 1e-3,
  near: float = 0, far: float = 1,
):
  device = r_o.device
  with torch.no_grad():
    hits = torch.zeros(r_o.shape[:-1] + (1,), dtype=torch.bool, device=device)
    rem = torch.ones_like(hits).squeeze(-1)
    curr_dist = torch.full_like(hits, near, dtype=torch.float)
    for i in range(iters):
      curr = r_o[rem] + r_d[rem] * curr_dist[rem]
      dist = self(curr)[...,0].reshape_as(curr_dist[rem])
      hits[rem] |= ((dist < eps) & (curr_dist[rem] <= far))
      # anything that was hit or is past range no longer need to compute
      curr_dist[rem] += dist
      rem[hits.squeeze(-1) | (curr_dist > far).squeeze(-1)] = False
    curr = r_o + r_d * curr_dist
  ...
  return curr, hits.squeeze(-1), curr_dist

def throughput(
  self,
  r_o, r_d,
  near: float, far: float,
  batch_size:int = 128,
):
  assert(far > near)
  # some random jitter I guess?
  max_t = far-near+random.random()*(2/batch_size)
  step = max_t/batch_size
  with torch.no_grad():
    sd = self(r_o + near * r_d)[...,0]
    curr_min = sd
    idxs = torch.zeros_like(sd, dtype=torch.long, device=r_d.device)
    for i in range(batch_size):
      t = near + step * (i+1)
      sd = self(r_o + t * r_d)[..., 0]
      idxs = torch.where(sd < curr_min, i+1, idxs)
      curr_min = torch.minimum(curr_min, sd)
    idxs = idxs.unsqueeze(-1)
    best_pos = r_o  + (near + idxs * step) * r_d
  return self(best_pos)[...,0], best_pos


#@torch.jit.script
def masked_loss(img_loss=F.mse_loss):
  # masked loss takes some image loss, such as l2, l1 or ssim, and then applies it with an
  # additional loss on the alpha channel.
  def aux(
    # got and exp have 4 channels, where the last are got_mask and exp_mask
    got, exp,
    mask_weight:int=1
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
