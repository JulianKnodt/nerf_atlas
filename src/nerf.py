import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import math

from .neural_blocks import (
  SkipConnMLP, UpdateOperator, FourierEncoder, PositionalEncoder, NNEncoder, EncodedGRU,
)
from .utils import (
  dir_to_elev_azim, autograd, laplace_cdf, load_sigmoid,
  sample_random_hemisphere, sample_random_sphere, upshifted_sigmoid,
  load_mip, to_spherical
)
import src.refl as refl
from .renderers import ( load_occlusion_kind, direct )
import src.march as march

@torch.jit.script
def cumuprod_exclusive(t):
  cp = torch.cumprod(t, dim=0)
  cp = torch.roll(cp, 1, dims=0)
  cp[0, ...] = 1.0
  return cp

#@torch.jit.script # cannot jit script cause of tensordot :)
def compute_pts_ts(
  rays, near, far, steps, lindisp=False, perturb: float = 0,
):
  r_o, r_d = rays.split([3,3], dim=-1)
  device = r_o.device
  if lindisp:
    t_vals = torch.linspace(0, 1, steps, device=device, dtype=r_o.dtype)
    ts = 1/(1/max(near, 1e-10) * (1-t_vals) + 1/far * (t_vals))
  else:
    ts = torch.linspace(near, far, steps=steps, device=device, dtype=r_o.dtype)

  if perturb > 0:
    mids = 0.5 * (ts[:-1] + ts[1:])
    lower = torch.cat([mids, ts[-1:]])
    upper = torch.cat([ts[:1], mids])
    rand = torch.rand_like(lower) * perturb
    ts = lower + (upper - lower) * rand
  pts = r_o.unsqueeze(0) + torch.tensordot(ts, r_d, dims = 0)
  return pts, ts, r_o, r_d

# given a set of densities, and distances between the densities,
# compute alphas from them.
#@torch.jit.script
def alpha_from_density(
  density, ts, r_d,
  softplus: bool = True,
):
  if softplus: sigma_a = F.softplus(density-1)
  else: sigma_a = F.relu(density)

  end_val = torch.full_like(ts[..., :1], 1e10)
  dists = torch.cat([ts[..., 1:] - ts[..., :-1], end_val], dim=-1)
  while len(dists.shape) < 4: dists = dists[..., None]
  dists = dists * torch.linalg.norm(r_d, dim=-1)
  alpha = 1 - torch.exp(-sigma_a * dists)
  # TODO is this equivalent to -expm1? Does it matter?
  #alpha = -torch.expm1(-sigma_a * dists)
  weights = alpha * cumuprod_exclusive(1.0 - alpha + 1e-10)
  return alpha, weights

# perform volumetric integration of density with some other quantity
# returns the integrated 2nd value over density at timesteps.
@torch.jit.script
def volumetric_integrate(weights, other): return torch.sum(weights[..., None] * other, dim=0)

# perform volumetric integration but only using some of other's values where the weights
# are big enough.
#
# TODO the computation of `other` itself should be sparse, so that it doesn't need to be
# computed in the first place.
@torch.jit.script
def sparse_volumetric_integrate(weights, other, eps:float=1e-3):
  vals = torch.full_like(other, 1e-3)
  mask = weights > 1e-3
  vals[mask] = other[mask]
  return torch.sum(weights[..., None] * vals, dim=0)


# bg functions, need to be here for pickling
def black(_elaz_r_d, _weights): return 0
def white(_, weights): 1-weights.sum(dim=0).unsqueeze(-1)
# having a random color will probably help prevent any background
def random_color(_elaz_r_d, weights):
  # TODO need to think this through more
  # This will make it so that there never is a background.
  summed = (1-weights.sum(dim=0).unsqueeze(-1))
  return torch.rand_like(summed) * summed
sky_kinds = {
  "black": black,
  "white": white,
  "mlp": "MLP_MARKER",
  "random": random_color,
}
def load_nerf(args):
  if args.model != "ae": args.latent_l2_weight = 0
  mip = load_mip(args)
  per_pixel_latent_size = 64 if args.data_kind == "pixel-single" else 0
  kwargs = {
    "mip": mip,
    "out_features": args.feature_space,
    "steps": args.steps,
    "t_near": args.near,
    "t_far": args.far,
    "per_pixel_latent_size": per_pixel_latent_size,
    "per_point_latent_size": 0,
    "instance_latent_size": 0,
    "sigmoid_kind": args.sigmoid_kind,
    "bg": args.bg,
  }
  cons = model_kinds.get(args.model, None)
  if cons is None: raise NotImplementedError(args.model)
  elif args.model == "ae":
    kwargs["normalize_latent"] = args.normalize_latent
    kwargs["encoding_size"] = args.encoding_size
  elif args.model == "volsdf":
    kwargs["sdf"] = sdf.load(args, with_integrator=False)
    kwargs["occ_kind"] = args.occ_kind
    kwargs["integrator_kind"] = args.integrator_kind or "direct"
  model = cons(**kwargs)
  if args.model == "ae" and args.latent_l2_weight > 0: model.set_regularize_latent()

  return model

class CommonNeRF(nn.Module):
  def __init__(
    self,

    # constructor for the reflectan
    r = None,

    steps: int = 64,

    #out_features: int = 3, # 3 is for RGB
    t_near: float = 0,
    t_far: float = 1,
    density_std: float = 0.01,
    noise_std: int = 1e-2,
    mip = None,
    instance_latent_size: int = 0,
    per_pixel_latent_size: int = 0,
    per_point_latent_size: int = 0,

    intermediate_size: int = 0,

    sigmoid_kind: str = "thin",
    bg: str = "black",
  ):
    super().__init__()
    self.empty_latent = nn.Parameter(
      torch.zeros(1,1,1,1,0, dtype=torch.float, requires_grad=False), requires_grad=False,
    )

    self.t_near = t_near
    self.t_far = t_far
    self.steps = steps
    self.mip = mip

    self.per_pixel_latent_size = per_pixel_latent_size
    self.per_pixel_latent = None

    self.instance_latent_size = instance_latent_size
    self.instance_latent = None

    self.per_pt_latent_size = per_point_latent_size
    self.per_pt_latent = None

    try: self.intermediate_size = intermediate_size
    except: ...

    self.alpha = None
    self.noise_std = 0.2

    self.set_bg(bg)
    if r is not None: self.refl = r(self.total_latent_size())
    self.set_sigmoid(sigmoid_kind)

  def forward(self, _x): raise NotImplementedError()
  def set_bg(self, bg="black"):
    sky_color_fn = sky_kinds.get(bg, None)
    if sky_color_fn is None: raise NotImplementedError(bg)
    self.sky_color = sky_color_fn

    if bg == "mlp":
      self.sky_mlp = SkipConnMLP(
        in_size=2, out=3, enc=FourierEncoder(input_dims=2),
        num_layers=3, hidden_size=64, init="xavier",
      )
      self.sky_color_fn = self.sky_from_mlp

  def set_sigmoid(self, kind="thin"):
    act = load_sigmoid(kind)
    self.feat_act = act
    if isinstance(self.refl, refl.LightAndRefl): self.refl.refl.act = act
    else: self.refl.act = act
  def sky_from_mlp(self, elaz_r_d, weights):
    return (1-weights.sum(dim=0)).unsqueeze(-1) * fat_sigmoid(self.sky_mlp(elaz_r_d))
  def total_latent_size(self) -> int:
    return self.mip_size() + \
      self.per_pixel_latent_size + \
      self.instance_latent_size + \
      self.per_pt_latent_size
  def set_per_pt_latent(self, latent):
    assert(latent.shape[-1] == self.per_pt_latent_size), \
      f"expected latent in [T, B, H, W, L={self.per_pixel_latent_size}], got {latent.shape}"
    assert(len(latent.shape) == 5), \
      f"expected latent in [T, B, H, W, L], got {latent.shape}"
    self.per_pt_latent = latent
  def set_per_pixel_latent(self, latent):
    assert(latent.shape[-1] == self.per_pixel_latent_size), \
      f"expected latent in [B, H, W, L={self.per_pixel_latent_size}], got {latent.shape}"
    assert(len(latent.shape) == 4), \
      f"expected latent in [B, H, W, L], got {latent.shape}"
    self.per_pixel_latent = latent
  def set_instance_latent(self, latent):
    assert(latent.shape[-1] == self.instance_latent_size), "expected latent in [B, L]"
    assert(len(latent.shape) == 2), "expected latent in [B, L]"
    self.instance_latent = latent

  def set_refl(self, refl):
    if hasattr(self, "refl"): self.refl = refl
    # TODO probably want to warn here

  def depths(self, depths):
    with torch.no_grad():
      return volumetric_integrate(self.alpha, depths[..., None, None, None])

  @property
  def nerf(self): return self

  def mip_size(self): return 0 if self.mip is None else self.mip.size() * 6
  def mip_encoding(self, r_o, r_d, ts):
    if self.mip is None: return None
    end_val = torch.tensor([1e10], device=ts.device, dtype=ts.dtype)
    ts = torch.cat([ts, end_val], dim=-1)
    return self.mip(r_o, r_d, ts[..., :-1], ts[..., 1:])

  # gets the current latent vector for this NeRF instance
  def curr_latent(self, pts_shape) -> ["T", "B", "H", "W", "L_pp + L_inst"]:
    curr = self.empty_latent.expand(pts_shape[:-1] + (0,)) if self.per_pt_latent is None \
      else self.per_pt_latent

    if self.per_pixel_latent is not None:
      ppl = self.per_pixel_latent[None, ...].expand(pts_shape[:-1] + (-1,))
      curr = torch.cat([curr, ppl], dim=-1)

    if self.instance_latent is not None:
      il = self.instance_latent[None, :, None, None, :].expand(pts_shape[:-1] + (-1,))
      curr = torch.cat([curr, il], dim=-1)

    return curr

class TinyNeRF(CommonNeRF):
  # No frills, single MLP NeRF
  def __init__(
    self,
    out_features: int = 3,
    **kwargs,
  ):
    super().__init__(**kwargs)
    self.estim = SkipConnMLP(
      in_size=3, out=1 + out_features,
      latent_size = self.total_latent_size(),
      num_layers=6, hidden_size=256, init="xavier",
    )

  def forward(self, rays):
    pts, self.ts, r_o, r_d = compute_pts_ts(
      rays, self.t_near, self.t_far, self.steps, perturb = 1 if self.training else 0,
    )
    return self.from_pts(pts, self.ts, r_o, r_d)

  def from_pts(self, pts, ts, r_o, r_d):
    latent = self.curr_latent(pts.shape)
    mip_enc = self.mip_encoding(r_o, r_d, ts)
    if mip_enc is not None: latent = torch.cat([latent, mip_enc], dim=-1)

    density, feats = self.estim(pts, latent).split([1, 3], dim=-1)

    self.alpha, self.weights = alpha_from_density(density, ts, r_d)
    return volumetric_integrate(self.weights, self.feat_act(feats)) + \
      self.sky_color(None, self.weights)

# A plain old nerf
class PlainNeRF(CommonNeRF):
  def __init__(
    self,
    out_features: int = 3,
    **kwargs,
  ):
    super().__init__(
      r = lambda ls: refl.View(
        out_features=out_features,
        latent_size=ls+self.intermediate_size,
      ),
      **kwargs,
    )

    self.first = SkipConnMLP(
      in_size=3, out=1 + self.intermediate_size, latent_size=self.total_latent_size(),
      enc=FourierEncoder(input_dims=3),
      num_layers = 6, hidden_size = 128, init="xavier",
    )

  def forward(self, rays):
    pts, self.ts, r_o, r_d = compute_pts_ts(
      rays, self.t_near, self.t_far, self.steps, perturb = 1 if self.training else 0,
    )
    return self.from_pts(pts, self.ts, r_o, r_d)

  def from_pts(self, pts, ts, r_o, r_d):
    latent = self.curr_latent(pts.shape)

    # If there is a mip encoding, stack it with the latent encoding.
    mip_enc = self.mip_encoding(r_o, r_d, ts)
    if mip_enc is not None: latent = torch.cat([latent, mip_enc], dim=-1)

    first_out = self.first(pts, latent)

    density = first_out[..., 0]
    if self.training and self.noise_std > 0:
      density = density + torch.randn_like(density) * self.noise_std

    intermediate = first_out[..., 1:]

    view = r_d[None, ...].expand_as(pts)
    rgb = self.refl(x=pts, view=view, latent=torch.cat([latent, intermediate], dim=-1))

    self.alpha, self.weights = alpha_from_density(density, ts, r_d)
    return volumetric_integrate(self.weights, rgb) + self.sky_color(view, self.weights)

def histogram_pts_ts(
  rays, near, far, rq,
):
  r_o, r_d = rays.split([3,3], dim=-1)
  device = r_o.device
  elaz = dir_to_elev_azim(r_d)
  hist = F.softplus(rq(torch.cat([r_o, elaz], dim=-1))).add(1e-2).cumsum(dim=-1)
  ts = (near + (far - near) * hist/hist.max(keepdim=True, dim=-1)).reshape(-1, *r_d.shape[:-1])
  pts = r_o.unsqueeze(0) + ts * r_d
  return pts, ts, r_o, r_d

# NeRF which uses a spline to compute ray query points
class HistogramNeRF(CommonNeRF):
  def __init__(
    self,
    out_features: int = 3,
    **kwargs,
  ):
    super().__init__(
      r = lambda ls: refl.View(
        out_features=out_features,
        latent_size=ls+self.intermediate_size,
      ),
      **kwargs,
    )

    self.ray_query = SkipConnMLP(
      in_size=5, out=self.step_size, enc=FourierEncoder(input_dims=3),
      num_layers = 6, hidden_size = 128, init="xavier",
    )

    self.first = SkipConnMLP(
      in_size=3, out=1 + self.intermediate_size, latent_size=self.total_latent_size(),
      enc=FourierEncoder(input_dims=3),
      num_layers = 6, hidden_size = 128, init="xavier",
    )

  def forward(self, rays):
    pts, self.ts, r_o, r_d = histogram_pts_ts(rays, self.t_near, self.t_far, self.ray_query)
    return self.from_pts(pts, self.ts, r_o, r_d)

  def from_pts(self, pts, ts, r_o, r_d):
    latent = self.curr_latent(pts.shape)
    mip_enc = self.mip_encoding(r_o, r_d, ts)
    if mip_enc is not None: latent = torch.cat([latent, mip_enc], dim=-1)

    first_out = self.first(pts, latent)
    density = first_out[..., 0]
    if self.training and self.noise_std > 0:
      density = density + torch.randn_like(density) * self.noise_std
    intermediate = first_out[..., 1:]

    view = r_d[None, ...].expand_as(pts)
    rgb = self.refl(x=pts, view=view, latent=torch.cat([latent, intermediate], dim=-1))

    self.alpha, self.weights = alpha_from_density(density, ts, r_d)
    return volumetric_integrate(self.weights, rgb) + self.sky_color(view, self.weights)

class SplineNeRF(CommonNeRF):
  def __init__(self, out_features: int = 3, **kwargs):
    super().__init__(
      r = lambda ls: refl.View(
        out_features=out_features,
        latent_size=ls+self.intermediate_size,
      ),
      **kwargs,
    )

    # Assume 4x4 in elevation and azimuth
    self.N = N = 8
    self.learned = nn.Parameter(torch.rand(N * N * 32, requires_grad=True), requires_grad=True)

    self.first = SkipConnMLP(
      in_size=1, out=1 + self.intermediate_size, latent_size=32,
      enc=FourierEncoder(input_dims=1),
      num_layers = 5, hidden_size = 256, init="xavier",
    )
  def forward(self, rays):
    pts, self.ts, r_o, r_d = compute_pts_ts(
      rays, self.t_near, self.t_far, self.steps, perturb = 1 if self.training else 0,
    )
    return self.from_pts(pts, self.ts, r_o, r_d)
  def compute_density_intermediate(self, x):
    el, az, rad = to_spherical(x).split([1,1,1], dim=-1)
    el = el/math.pi
    assert((el <= 1).all()), el.max().item()
    assert((el >= 0).all()), el.min().item()
    az = (az/math.pi + 1)/2
    assert((az <= 1).all()), az.max().item()
    assert((az >= 0).all())
    N = self.N
    ps = torch.stack(self.learned.chunk(N), dim=0)
    ps = ps[:, None, None, None,None, :].expand(-1, *el.shape[:-1], -1)
    # TODO need to reshape this to be same size as el
    grid_az = torch.stack(de_casteljau(ps, el, N).chunk(N, dim=-1), dim=0)
    return self.first(rad, de_casteljau(grid_az, az, N))
  def from_pts(self, pts, ts, r_o, r_d):
    first_out = self.compute_density_intermediate(pts)
    density = first_out[..., 0]
    if self.training and self.noise_std > 0:
      density = density + torch.randn_like(density) * self.noise_std
    intermediate = first_out[..., 1:]

    view = r_d[None, ...].expand_as(pts)
    rgb = self.refl(x=pts, view=view, latent=intermediate)

    self.alpha, self.weights = alpha_from_density(density, ts, r_d)
    return volumetric_integrate(self.weights, rgb) + self.sky_color(view, self.weights)

# NeRF with a thin middle layer, for encoding information
class NeRFAE(CommonNeRF):
  def __init__(
    self,
    out_features: int = 3,
    encoding_size: int = 32,
    normalize_latent: bool = False,
    **kwargs,
  ):
    super().__init__(
      r = lambda _: refl.View(
        out_features=out_features,
        latent_size=encoding_size+self.intermediate_size,
      ),
      **kwargs,
    )

    self.latent_size = self.total_latent_size()

    self.encode = SkipConnMLP(
      in_size=3, out=encoding_size, latent_size=self.latent_size,
      num_layers=5, hidden_size=128, enc=FourierEncoder(input_dims=3),
      init="xavier",
    )

    self.density_tform = SkipConnMLP(
      in_size=encoding_size, out=1+self.intermediate_size, latent_size=0,
      num_layers=5, hidden_size=64, init="xavier",
    )

    self.encoding_size = encoding_size
    self.regularize_latent = False
    self.normalize_latent = normalize_latent

  def set_regularize_latent(self):
    self.regularize_latent = True
    self.latent_l2_loss = 0
  def forward(self, rays):
    pts, self.ts, r_o, r_d = compute_pts_ts(
      rays, self.t_near, self.t_far, self.steps, perturb = 1 if self.training else 0,
    )
    return self.from_pts(pts, self.ts, r_o, r_d)

  def from_pts(self, pts, ts, r_o, r_d):
    encoded = self.compute_encoded(pts, ts, r_o, r_d)
    if self.regularize_latent:
      self.latent_l2_loss = torch.linalg.norm(encoded, dim=-1).square().mean()
    return self.from_encoded(encoded, ts, r_d, pts)

  def compute_encoded(self, pts, ts, r_o, r_d):
    latent = self.curr_latent(pts.shape)

    mip_enc = self.mip_encoding(r_o, r_d, ts)

    # If there is a mip encoding, stack it with the latent encoding.
    if mip_enc is not None: latent = torch.cat([latent, mip_enc], dim=-1)

    return self.encode(pts, latent)
  def from_encoded(self, encoded, ts, r_d, pts):
    if self.normalize_latent: encoded = F.normalize(encoded, dim=-1)

    first_out = self.density_tform(encoded)
    density, intermediate = first_out[..., 0], first_out[..., 1:]

    if self.training and self.noise_std > 0:
      density = density + torch.randn_like(density) * self.noise_std

    rgb = self.refl(
      x=pts, view=r_d[None,...].expand_as(pts), latent=torch.cat([encoded, intermediate],dim=-1),
    )

    self.alpha, self.weights = alpha_from_density(density, ts, r_d)

    return volumetric_integrate(self.weights, rgb) + self.sky_color(None, self.weights)

def identity(x): return x

# https://arxiv.org/pdf/2106.12052.pdf
class VolSDF(CommonNeRF):
  def __init__(
    self, sdf,
    # how many features to pass from density to RGB
    out_features: int = 3,
    occ_kind=None,
    integrator_kind="direct",
    scale_softplus=False,
    **kwargs,
  ):
    super().__init__(**kwargs)
    self.sdf = sdf
    # the reflectance model is in the SDF, so don't encode it here.
    self.scale = nn.Parameter(torch.tensor(0.1, requires_grad=True))
    self.secondary = None
    self.out_features = out_features
    self.scale_act = identity if not scale_softplus else nn.Softplus()
    if occ_kind is not None:
      assert(isinstance(self.sdf.refl, refl.LightAndRefl)), \
        f"Must have light w/ volsdf integration {type(self.sdf.refl)}"
      self.occ = load_occlusion_kind({}, occ_kind, self.sdf.latent_size)

      if integrator_kind == "direct": self.secondary = self.direct
      elif integrator_kind == "path": self.convert_to_path()
      else: raise NotImplementedError(f"unknown integrator kind {integrator_kind}")
  def convert_to_path(self):
    if self.secondary == self.path: return False
    self.secondary = self.path
    self.path_n = N = 3
    missing_cmpts = 3 * (N + 1) + 6

    # transfer_fn := G(x1, x2) -> [0,1]
    self.transfer_fn = SkipConnMLP(
      in_size=6, out=1, enc=FourierEncoder(input_dims=6),
      # multiply by two here ince it's the pair of latent values at sets of point
      latent_size = self.sdf.latent_size * 2,
      hidden_size=512,
    )
    return True
  def direct(self, r_o, weights, pts, view, n, latent):
    out = torch.zeros_like(pts)
    for light in self.sdf.refl.light.iter():
      light_dir, light_val = self.occ(pts, light, self.sdf.intersect_mask, latent=latent)
      bsdf_val = self.sdf.refl(x=pts, view=view, normal=n, light=light_dir, latent=latent)
      out = out + bsdf_val * light_val
    return out
  # Single bounce path tracing. In theory could be extended to an arbitrary # of bounces.
  def path(self, r_o, weights, pts, view, n, latent):
    out = torch.zeros_like(pts)

    # number of samples for 1st order bounces
    N = self.path_n if self.training else max(10, self.path_n*2)

    # for each point sample some number of directions
    dirs = sample_random_sphere(n, num_samples=N)
    # compute intersection of random directions with surface
    ext_pts, ext_hits, dists, _ = march.bisect(
      self.sdf.underlying, pts[None,...].expand_as(dirs), dirs, iters=64, near=5e-3, far=6,
    )

    ext_sdf_vals, ext_latent = self.sdf.from_pts(ext_pts)

    ext_view = F.normalize(ext_pts - r_o[None,None,...], dim=-1)
    ext_n = F.normalize(self.sdf.normals(ext_pts), dim=-1).detach()

    fit = lambda x: x.unsqueeze(0).expand(N,-1,-1,-1,-1,-1)
    # reflection at the intersection points from light incoming from the random directions
    first_step_bsdf = self.sdf.refl(
      x=fit(pts), view=ext_view, normal=fit(n), light=-dirs, latent=fit(latent),
    )
    # compute transfer function (G) between ext_pts and pts (which is a proxy for the density).
    tf = self.transfer_fn(
      torch.cat([ext_pts, pts.unsqueeze(0).expand_as(ext_pts)],dim=-1),
      torch.cat([ext_latent, latent.unsqueeze(0).expand_as(ext_latent)], dim=-1),
    ).sigmoid()
    first_step_bsdf = first_step_bsdf * tf

    for light in self.sdf.refl.light.iter():
      # compute direct lighting at each point (identical to direct)
      light_dir, light_val = self.occ(pts, light, self.sdf.intersect_mask, latent=latent)
      bsdf_val = self.sdf.refl(x=pts, view=view, normal=n, light=light_dir, latent=latent)
      out = out + bsdf_val * light_val
      # compute light contribution and bsdf at 2ndary points from this light
      ext_light_dir, ext_light_val = \
        self.occ(ext_pts, light, self.sdf.intersect_mask, latent=ext_latent)
      path_bsdf = self.sdf.refl(
        x=ext_pts, view=dirs, normal=ext_n, light=ext_light_dir, latent=ext_latent,
      )
      second_step = ext_light_val * path_bsdf
      # sum over the contributions at each point adding with each secondary contribution
      # Take the mean, so that adding in more samples does not cause an infinite increase in
      # light.
      secondary = (first_step_bsdf * second_step).mean(dim=0)
      out = out + secondary

    return out
  def forward(self, rays):
    pts, self.ts, r_o, r_d = compute_pts_ts(
      rays, self.t_near, self.t_far, self.steps, perturb = 1 if self.training else 0,
    )
    return self.from_pts(pts, self.ts, r_o, r_d)

  @property
  def intermediate_size(self): return self.sdf.latent_size

  def set_refl(self, refl): self.sdf.refl = refl

  @property
  def refl(self): return self.sdf.refl

  def from_pts(self, pts, ts, r_o, r_d):
    latent = self.curr_latent(pts.shape)
    mip_enc = self.mip_encoding(r_o, r_d, ts)
    if mip_enc is not None: latent = torch.cat([latent, mip_enc], dim=-1)

    sdf_vals, latent = self.sdf.from_pts(pts)
    scale = self.scale_act(self.scale)
    self.scale_post_act = scale
    density = 1/scale * laplace_cdf(-sdf_vals, scale)
    self.alpha, self.weights = alpha_from_density(density, ts, r_d, softplus=False)

    n = None
    if self.sdf.refl.can_use_normal or self.secondary is not None:
      self.n = n = F.normalize(self.sdf.normals(pts), dim=-1)

    view = r_d.unsqueeze(0).expand_as(pts)
    if self.secondary is None: rgb = self.sdf.refl(x=pts, view=view, normal=n, latent=latent)
    else: rgb = self.secondary(r_o, self.weights, pts, view, n, latent)

    return volumetric_integrate(self.weights, rgb)
  def set_sigmoid(self, kind="thin"):
    if not hasattr(self, "sdf"): return
    act = load_sigmoid(kind)
    if isinstance(self.refl, refl.LightAndRefl): self.refl.refl.act = act
    else: self.refl.act = act

class RecurrentNeRF(CommonNeRF):
  def __init__(
    self,
    out_features: int = 3,

    **kwargs,
  ):
    super().__init__(
      r = lambda latent_size: refl.View(
        out_features=out_features,
        latent_size=latent_size+self.intermediate_size,
      ),
      **kwargs,
    )

    self.first = EncodedGRU(
      in_size=3, out=1,
      encs=[
        FourierEncoder(input_dims=3, sigma=1<<1),
        FourierEncoder(input_dims=3, sigma=1<<2),
        FourierEncoder(input_dims=3, sigma=1<<3),
        FourierEncoder(input_dims=3, sigma=1<<3),
        FourierEncoder(input_dims=3, sigma=1<<4),
        FourierEncoder(input_dims=3, sigma=1<<4),
        FourierEncoder(input_dims=3, sigma=1<<5),
      ],
      state_size=256,
      latent_out=self.intermediate_size,
    )

  def forward(self, rays):
    pts, self.ts, r_o, r_d = compute_pts_ts(
      rays, self.t_near, self.t_far, self.steps, perturb = 1 if self.training else 0,
    )
    return self.from_pts(pts, self.ts, r_o, r_d)

  def from_pts(self, pts, ts, r_o, r_d):
    latent = self.curr_latent(pts.shape)

    densities, intermediate = self.first(pts, latent)
    acc_density = (torch.cumsum(densities, dim=-1) - densities).detach() + densities
    if self.training and self.noise_std > 0:
      acc_density = acc_density + torch.randn_like(acc_density) * self.noise_std

    view = r_d[None, ...].expand_as(pts)
    rgb = self.refl(x=pts, view=view, latent=torch.cat([latent, intermediate], dim=-1))
    images = []
    for i in range(acc_density.shape[-1]):
      density = acc_density[..., i]
      alpha, weights = alpha_from_density(density, ts, r_d)
      img = volumetric_integrate(weights, rgb)
      images.append(img)
    # return many images and regularize on all of them. # TODO how to make this fit in current
    # framework?
    return images

def alternating_volsdf_loss(model, nerf_loss, sdf_loss):
  def aux(x, ref): return nerf_loss(x, ref[..., :3]) if model.vol_render else sdf_loss(x, ref)
  return aux

# An odd module which alternates between volume rendering and SDF rendering
class AlternatingVolSDF(nn.Module):
  def __init__(
    self,
    volsdf: VolSDF,
    # run_len is how many iterations of volume/SDF rendering it will perform.
    # it performs run_len/2 volume, and run_len/2 SDF
    run_len:int = 4096,
  ):
    super().__init__()
    assert(isinstance(volsdf, VolSDF))
    self.volsdf = volsdf
    self.i = 0
    self.force_volume = False
    self.force_sdf = False
    self.run_len = run_len
    # TODO add some count for doing only sdfs first?

  # forward a few properties to sdf
  @property
  def sdf(self): return self.volsdf.sdf
  @property
  def nerf(self): return self.volsdf
  @property
  def n(self): return self.volsdf.n
  @property
  def total_latent_size(self): return self.volsdf.total_latent_size
  @property
  def refl(self): return self.volsdf.refl
  def set_refl(self, refl): return self.volsdf.set_refl(refl)

  def forward(self, rays):
    if not self.training: return self.volsdf(rays)
    self.i = (self.i + 1) % self.run_len
    self.vol_render = (self.i < self.run_len//2 or self.force_volume) and not self.force_sdf
    if self.vol_render:
      return self.volsdf(rays)
    else:
      return direct(self.volsdf.sdf, self.volsdf.refl, self.volsdf.occ, rays, self.training)

def one(*args, **kwargs): return 1
# de_casteljau's algorithm for evaluating bezier splines without numerical instability..
def de_casteljau(coeffs, t, N: int):
  betas = coeffs
  m1t = 1 - t
  # TODO some way to vectorize this?
  for i in range(1, N): betas = betas[:-1] * m1t + betas[1:] * t
  return betas.squeeze(0)

# de_moor's algorithm for evaluating bezier splines with a given knot vector
def de_moors(coeffs, t, knots, N: int):
  raise NotImplementedError("TODO implement")

def cubic_bezier(coeffs, t, N: int):
  assert(N == 4), f"Must be cubic, got {N}"
  m1t = 1 - t
  m1t_sq, t_sq = m1t * m1t, t * t
  k = torch.stack([m1t_sq * m1t, m1t_sq * t, t_sq * m1t, t_sq * t], dim=0)
  return (k * coeffs).sum(dim=0)

# Dynamic NeRF for multiple frams
class DynamicNeRF(nn.Module):
  def __init__(self, canonical: CommonNeRF, spline:int=0):
    super().__init__()
    self.canonical = canonical
    self.spline = spline

    if spline > 0: self.set_spline_estim(spline)
    else: self.set_delta_estim()
  def set_delta_estim(self):
    self.delta_estim = SkipConnMLP(
      # x,y,z,t -> dx, dy, dz, rigidity
      in_size=4, out=3+1,
      num_layers = 6, hidden_size = 324,
      init="xavier",
    )
    self.time_estim = self.direct_predict
  def set_spline_estim(self, spline_points):
    assert(spline_points > 1), "Must pass N > 1 spline"
    # x,y,z -> n control points, rigidity
    self.delta_estim = SkipConnMLP(
      in_size=3, out=(spline_points-1)*3+1, num_layers=6,
      hidden_size=324, init="xavier",
    )
    self.spline_fn = cubic_bezier if spline_points == 4 else de_casteljau
    self.spline_n = spline_points
    self.time_estim = self.spline_interpolate

  def direct_predict(self, x, t):
    dp, rigidity = self.delta_estim(torch.cat([x, t], dim=-1)).split([3, 1], dim=-1)
    self.rigidity = rigidity = upshifted_sigmoid(rigidity/2)
    self.dp = dp = torch.where(t.abs() < 1e-6, torch.zeros_like(x), dp)
    return dp * rigidity
  def spline_interpolate(self, x, t):
    # t is mostly expected to be between 0 and 1, but can be outside for fun.
    rigidity, ps = self.delta_estim(x).split([1, 3 * (self.spline_n-1)], dim=-1)
    self.rigidity = rigidity = upshifted_sigmoid(rigidity/2)
    ps = torch.stack(ps.split([3] * self.spline_n-1, dim=-1), dim=0)
    init_ps = ps[None, 0]
    self.dp = dp = self.spline_fn(ps - init_ps, t, self.spline_n)
    return dp * rigidity + init_ps.squeeze(0)

  @property
  def nerf(self): return self.canonical
  @property
  def refl(self): return self.canonical.refl
  @property
  def sdf(self): return getattr(self.canonical, "sdf", None)
  @property
  def intermediate_size(self): return self.canonical.intermediate_size
  def total_latent_size(self): return self.canonical.total_latent_size()
  def set_refl(self, refl): self.canonical.set_refl(refl)
  def forward(self, rays_t):
    rays, t = rays_t
    pts, self.ts, r_o, r_d = compute_pts_ts(
      rays, self.canonical.t_near, self.canonical.t_far, self.canonical.steps,
      perturb = 1 if self.training else 0,
    )
    self.canonical.ts = self.ts
    t = t[None, :, None, None, None].expand(*pts.shape[:-1], 1)
    dp = self.time_estim(pts, t)
    return self.canonical.from_pts(pts + dp, self.ts, r_o, r_d)

# Long Dynamic NeRF for computing arbitrary continuous sequences.
class LongDynamicNeRF(nn.Module):
  def __init__(
    self,
    canonical: CommonNeRF,
    segments: int = 32,
    spline:int=4,
    segment_embedding_size:int=128,
    # intermediate size from the anchors to the poinst estimator
    anchor_interim:int=128,
  ):
    super().__init__()

    self.canonical = canonical
    self.spline_n = spline
    self.ses = ses = segment_embedding_size
    self.segments = segments

    n_points = segments * spline
    # keep an embedding for each segment # TODO is there an off by 1 error
    self.seg_emb = nn.Embedding(segments+1, ses)
    self.interim_size = interim_size
    assert(spline > 2), "Must pass N > 2 spline"
    # the anchor points of each spline
    self.anchors = SkipConnMLP(
      in_size=3, out=3+anchor_interim,
      num_layers=5, hidden_size=512, latent_size=ses, init="xavier",
    )
    # the interior of each spline, will get passed latent values from the anchors.
    self.point_estim = SkipConnMLP(
      in_size=3, out=(spline-2) * 3,
      num_layers=6, hidden_size=512,
      latent_size=ses+anchor_interim, init="xavier",
    )
    self.spline_n = spline
    self.global_rigidity = SkipConnMLP(
      in_size=3, out=1, init="xavier", num_layers=5, hidden_size=256,
    )
    # Parameters in SE3, except 0 which is always 0
    self.global_spline = nn.Parameter(torch.randn(n_points, 6))
    self.spline_fn = de_casteljeu if self.spline_n != 4 else cubic_bezier

  def set_refl(self, refl): self.canonical.set_refl(refl)
  def total_latent_size(self): return self.canonical.total_latent_size()
  @property
  def nerf(self): return self.canonical
  @property
  def refl(self): return self.canonical.refl
  @property
  def sdf(self): return getattr(self.canonical, "sdf", None)
  @property
  def intermediate_size(self): return self.canonical.intermediate_size
  def forward(self, rays_t):
    rays, t = rays_t
    rays = rays.expand(t.shape[0], *rays.shape[1:])
    pts, self.ts, r_o, r_d = compute_pts_ts(
      rays, self.canonical.t_near, self.canonical.t_far, self.canonical.steps,
      perturb = 1 if self.training else 0,
    )
    self.pts = pts
    self.canonical.ts = self.ts
    t = t[None, :, None, None, None].expand(*pts.shape[:-1], 1)
    seg = t.floor().int().clamp(min=0)
    assert((seg >= 0).all()), "must have all positive segments"

    embs = self.seg_emb(torch.cat([seg, seg+1], dim=-1))
    anchors, anchor_latent = self.anchors(
      pts[..., None, :].expand(*pts.shape[:-1], 2, pts.shape[-1]), embs,
    ).split([3, self.interim_size], dim=-1)
    print(anchors.shape)
    exit()
    starting, end = [a.squeeze(-2).unsqueeze(0) for a in anchors.split([1,1], dim=-2)]
    seg_emb = embs[..., 0, :]

    point_estim_latent = torch.cat([seg_emb, anchor_latent.flatten(start_dim=-2)], dim=-1)
    inner_points = torch.stack(self.point_estim(pts, point_estim_latent).split(3, dim=-1), dim=0)

    control_pts = torch.cat([starting, inner_points, end], dim=0)
    self.rigidity = rigidity = upshifted_sigmoid(self.global_rigidity(pts)/2)
    self.dp = dp = self.spline_fn(control_pts, t.frac(), self.spline_n+1) * rigidity
    return self.canonical.from_pts(pts + dp, self.ts, r_o, r_d)

# Dynamic NeRFAE for multiple frames with changing materials
class DynamicNeRFAE(DynamicNeRF):
  def __init__(self, canonical: NeRFAE, spline:int=0):
    assert(isinstance(canonical, NeRFAE)), "Must use NeRFAE for DynamicNeRFAE"
    super().__init__(self, canonical, spline)
    # TODO how to make the spline return more values
  def forward(self, rays_t):
    rays, t = rays_t
    pts, ts, r_o, r_d = compute_pts_ts(
      rays, self.canon.t_near, self.canon.t_far, self.canon.steps,
    )
    self.ts = ts

    t = t[None, :, None, None, None].expand(*pts.shape[:-1], 1)
    dp, d_enc = self.time_estim(pts, t).split([3, self.canon.encoding_size], dim=-1)
    encoded = self.canon.compute_encoded(pts + dp, ts, r_o, r_d)
    return self.canon.from_encoded(encoded + d_enc, ts, r_d, pts)

# TODO fix this
class SinglePixelNeRF(nn.Module):
  def __init__(
    self,
    canon: CommonNeRF,
    encoder,
    img,

    device: torch.device = "cuda",
  ):
    super().__init__()
    self.canon = canon
    self.encoder = encoder
    # encode image
    self.encoder(img)

    self.device = device

  @property
  def nerf(self): return self.canon
  def forward(self, rays_uvs):
    rays, uvs = rays_uvs
    latent = self.encoder.sample(uvs)
    self.canon.set_per_pixel_latent(latent)
    return self.canon(rays)

class MPI(nn.Module):
  # Multi Plane Imaging.
  def __init__(
    self,
    canonical: CommonNeRF,

    position = [0,0,0],
    normal = [0,0,-1],
    delta=0.1,

    n_planes: int = 6,

    device="cuda",
  ):
    super().__init__()

    self.n_planes = torch.linspace(canon.t_near, canon.t_far, steps=n_planes, device=device)
    self.position = torch.tensor(position, device=device, dtype=torch.float)
    self.normal = torch.tensor(normal, device=device, dtype=torch.float)
    self.delta = delta

    self.canonical = canonical.to(device)
  def forward(self, rays):
    r_o, r_d = rays.split([3,3], dim=-1)
    device = r_o.device

    n = self.normal.expand_as(r_d)
    denom = (n * r_d).sum(dim=-1, keepdim=True)
    centers = self.position.unsqueeze(0) + torch.tensordot(
      self.delta * torch.arange(self.n_planes, device=device, dtype=torch.float),
      -self.normal, dims=0,
    )
    ts = ((centers - r_o) * n).sum(dim=-1, keepdim=True)/denom
    # if denom is too small it will have numerical instability because it's near parallel.
    hits = torch.where(denom.abs() > 1e-3, ts, torch.zeros_like(denom))
    pts = r_o.unsqueeze(0) + r_d.unsqueeze(0) * hits

    return self.canonical.from_pts(pts, ts, r_o, r_d)

  @property
  def nerf(self): return self.canon
  def from_pts(self, pts, ts, r_o, r_d):
    density, feats = self.estim(pts).split([1, 3], dim=-1)

    alpha, weights = alpha_from_density(density, ts, r_d)
    return volumetric_integrate(weights, self.feat_act(feats))

def load_dyn(args, model, device):
  dyn_cons = dyn_model_kinds.get(args.dyn_model, None)
  if dyn_cons is None: raise NotImplementedError(f"Unknown dyn kind: {args.dyn_model}")

  if args.with_canon is not None:
    model = torch.load(args.with_canon, map_location=device)
    assert(isinstance(model, CommonNeRF)), f"Can only use NeRF subtype, got {type(model)}"
    # TODO if dynae need to check that model is NeRFAE
  kwargs = { "canonical": model, "spline": args.spline }
  if dyn_cons == LongDynamicNeRF: kwargs["segments"] = args.segments
  return dyn_cons(**kwargs)

dyn_model_kinds = {
  "plain": DynamicNeRF,
  "ae": DynamicNeRFAE,
  "long": LongDynamicNeRF,
}

model_kinds = {
  "tiny": TinyNeRF,
  "plain": PlainNeRF,
  "ae": NeRFAE,
  "volsdf": VolSDF,
  # experimental:
  "hist": HistogramNeRF,
  "spline": SplineNeRF,
}

# TODO test this and actually make it correct.
def metropolis_sampling(
  density_estimator,
  ts_init, r_o, r_d,
  iters: int = 6,
):
  # need to make this the shape of r_d exit with last dim 1
  curr = ts_init
  print(r_o.shape)
  exit()
  with torch.no_grad():
    candidates = torch.rand_like(curr) + curr
    curr_density = density_estimator(candidates)
    for i in range(iters):
      candidates = torch.randn_like(curr) + curr
      density = density_estimator(candidates)
      acceptance = density/curr_density
      alphas = torch.rand_like(density)
      mask = acceptance <= alphas
      curr = torch.where(mask, candidates, curr)
      curr_density = torch.where(mask, density, curr_density)
  return curr, r_o + curr * r_d

# TODO need to test this more, doesn't seem to work that well
def inverse_sample(
  density_estimator,
  pts, ts, r_o, r_d,
):
  with torch.no_grad():
    _, weights = alpha_from_density(density_estimator(pts.squeeze(-1)), ts, r_d)
    weights = weights.clamp(min=1e-10)
    pdf = weights/weights.sum(dim=0,keepdim=True)
    cdf = torch.cumsum(pdf, dim=0)
    N = ts.shape[0]
    # XXX this only works because we assume that the number of samples (N) is the same.
    #u = torch.rand_like(cdf)
    u = torch.linspace(0, 1, N, device=cdf.device)\
      [..., None, None, None].expand_as(cdf)
    # XXX this operates on innermost dimension, so need to do this transpose
    inds = torch.searchsorted(
      cdf.transpose(0, -1).contiguous(), u.transpose(0, -1).contiguous(), right=True
    ).transpose(0, -1)
    below = (inds - 1).clamp(min=0)
    above = inds.clamp(max=N-1)
    inds_g = torch.stack([below, above], dim=-1)

    # TODO what is the right dimension to add here?
    cdf_g = torch.gather(cdf.unsqueeze(1).expand_as(inds_g), 0, inds_g)
    bins_g = torch.gather(ts[:, None, None, None, None].expand_as(inds_g), 0, inds_g)

    denom = cdf_g[..., 1] - cdf_g[..., 0]
    denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
    t = (u - cdf_g[..., 0]) / denom
    samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])
    new_pts = r_o + samples.unsqueeze(-1) * r_d
  return samples, new_pts
