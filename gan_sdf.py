import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import argparse
from src.utils import ( autograd, eikonal_loss, save_image )
from src.neural_blocks import ( SkipConnMLP, FourierEncoder, PointNet )
from src.cameras import ( OrthogonalCamera )
from src.march import ( bisect )
from tqdm import trange

import random
import math

def arguments():
  a = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  a.add_argument(
    # TODO add more here to learn between
    "--target", choices=["sphere"], default="sphere", help="What kind of SDF to learn",
  )
  a.add_argument(
    "--epochs", type=int, default=5000, help="Number of epochs to train for",
  )
  a.add_argument(
    "--bounds", type=float, default=1.5, help="Bounded region to train SDF in",
  )
  a.add_argument(
    "--batch-size", type=int, default=16, help="Number of batches to train at the same time",
  )
  a.add_argument(
    "--sample-size", type=int, default=1<<12, help="Number of points to train per batch",
  )
  a.add_argument(
    "--G-step", type=int, default=1, help="Number of steps to take before optimizing G",
  )
  a.add_argument(
    "--eikonal-weight", type=float, default=1e-2, help="Weight of eikonal loss",
  )
  a.add_argument(
    "--save-freq", type=int, default=1000, help="How often to save the model"
  )
  a.add_argument(
    "--num-test-samples", type=int, default=16, help="How tests to run",
  )
  a.add_argument(
    "--load", action="store_true", help="Load old generator and discriminator"
  )
  a.add_argument(
    "--render-size", type=int, default=200, help="Size to render result images",
  )
  return a.parse_args()


device="cpu"
if torch.cuda.is_available():
  device = torch.device("cuda:0")
  torch.cuda.set_device(device)

# Computes samples within some bounds, returning [..., N, 3] samples.
def random_samples_within(bounds: [..., 6], samples:int = 1024):
  # lower-left, upper right
  ll, ur = bounds.split([3,3], dim=-1)
  rand = torch.rand(*ll.shape, device=bounds.device, requires_grad=True)
  samples = ll + rand * (ur - ll)
  return samples

# Picks a random cube bounding box inside of an existing bound.
# Please pass half the size of the desired bounding box, i.e. for a bounding box of size 1 pass
# 0.5.
def subbound(bounds: [..., 6], half_size: float):
  assert(half_size > 0), "Must pass positive size"
  ll, ur = bounds.split([3,3], dim=-1)
  center = torch.rand_like(ll)
  ll = ll + half_size
  ur = ur - half_size
  center = (center + ll) * (ur - ll)
  return torch.cat([center-half_size, center+half_size], dim=-1)

# rescales the points inside of the bound to a canonical [-1, 1] space.
# The half_size of the bounding box is necessary to compute the scaling factor.
def rescale_pts_in_bound(ll, pts, sdf_values, half_size: float, end_size:float):
  rescaled_pts = (pts - ll)/half_size - 1
  # https://iquilezles.org/www/articles/distfunctions/distfunctions.htm
  rescaled_sdf_values = sdf_values * (half_size * 2)

  return rescaled_pts, rescaled_sdf_values

def pts_from_bd_to(ll1, ll2, pts, sdf_values, size_ratio: float):
  new_pts = (pts - ll1) * size_ratio + ll2
  new_sdf_values = sdf_values * size_ratio
  return new_pts, new_sdf_values


# [WIP] returns semi-equispaced samples within some bounds, sampling a fixed amt of times per
# dimesnion.
def stratified_rand_samples(bounds: [..., 6], samples_per_dim:int=8):
  ll, ur = bounds.split([3,3], dim=-1)
  dim = torch.linspace(0,1,samples_per_dim, device=ll.device, dtype=torch.float)
  samples = torch.stack(torch.meshgrid(dim, dim, dim), dim=-1).reshape(-1, 3)
  samples = samples + torch.randn_like(samples)*0.01
  print(samples.shape, ll.shape)
  samples = ll.unsqueeze(-2) + samples * (ur-ll).unsqueeze(-2)
  exit()
  return samples.requires_grad_()

# computes boundes for a bunch of samples.
def bounds_for(samples: [..., "Batch", 3]):
  ll = samples.min(dim=-2)[0]
  ur = samples.max(dim=-2)[0]
  return torch.cat([ll, ur], dim=-1)

def scaled_training_step(
  target, bounds, max_bound,
  latent_noise,
  G, opt_G,
  D, opt_D,

  args,
):
  D.zero_grad()
  exp_sample_size = 0.75 #0.25 + random.random()/2
  subbd_exp = subbound(bounds, exp_sample_size)
  exp_pts = random_samples_within(subbd_exp, args.sample_size)
  exp_sub_vals = target(exp_pts)
  exp_pts, exp_sub_vals = rescale_pts_in_bound(
    subbd_exp[..., :3], exp_pts, exp_sub_vals, exp_sample_size, max_bound,
  )
  s2_real = D(exp_pts.detach(), exp_sub_vals.detach())

  got_sample_size = 0.75 #0.25 + random.random()/2
  subbd_got = subbound(bounds, got_sample_size)
  #got_tgt_bd_sz = 0.25 + random.random()/2
  #subbd_got_tgt = subbound(bounds, got_tgt_bd_sz)
  got_pts = random_samples_within(subbd_got, args.sample_size)
  got_sub_vals = G(got_pts, latent_noise)
  new_got_pts, got_sub_vals = rescale_pts_in_bound(
    subbd_got[..., :3], got_pts, got_sub_vals, got_sample_size, max_bound,
  )
  s2_fake = D(new_got_pts.detach(), got_sub_vals.detach())

  real_loss = F.binary_cross_entropy_with_logits(s2_real, torch.ones_like(s2_real))
  fake_loss = F.binary_cross_entropy_with_logits(s2_fake, torch.zeros_like(s2_fake))

  D_loss = real_loss + fake_loss
  D_loss.backward()
  opt_D.step()

  # partial SDF training
  G.zero_grad()

  got_pts = random_samples_within(subbd_got, args.sample_size)
  got_sub_vals, got_n = G.vals_normal(got_pts, latent_noise)
  new_got_pts, got_sub_vals = rescale_pts_in_bound(
    subbd_got[..., :3], got_pts, got_sub_vals, got_sample_size, max_bound,
  )
  # have to rescale normals due to rescaling? Actually maybe not because
  # we care about the normals in the original space.
  # https://pbr-book.org/3ed-2018/Geometry_and_Transformations/Applying_Transformations
  # got_n = got_n * max_bound/got_sample_size

  s2_fool = D(got_pts, got_sub_vals)
  fooling_loss = F.binary_cross_entropy_with_logits(s2_fool, torch.ones_like(s2_fool))

  G_loss = fooling_loss
  (G_loss + args.eikonal_weight*eikonal_loss(got_n)).backward()
  opt_G.step()

  return

# trains a GAN with a target SDF, as well as a discriminator point net.
def train(
  targets,
  model, opt_G,
  discriminator, opt_D,

  args,
):
  D = discriminator
  max_b = abs(args.bounds)
  min_b = -max_b
  bounds = torch.tensor([
    min_b, min_b, min_b, max_b, max_b, max_b,
  ], device=device, dtype=torch.float)
  sample_size = args.sample_size
  batch_size = args.batch_size
  bounds = bounds[None,None,:].expand(batch_size, sample_size, 6)
  t = trange(args.epochs)
  G_loss = 0
  D_loss = 1e5
  for i in t:
    target = random.choice(targets)
    pt_samples1 = random_samples_within(bounds, sample_size)
    # whole SDF discriminator step
    D.zero_grad()
    exp = target(pt_samples1)

    latent_noise = torch.randn(batch_size, 1, model.latent_size, device=device)\
      .mul(3)\
      .expand(batch_size, sample_size, model.latent_size)
    got = model(pt_samples1, latent_noise)

    s1_real = D(pt_samples1.detach(), exp.detach())
    s1_fake = D(pt_samples1.detach(), got.detach())

    real_loss = F.binary_cross_entropy_with_logits(s1_real, torch.ones_like(s1_real))
    fake_loss = F.binary_cross_entropy_with_logits(s1_fake, torch.zeros_like(s1_fake))

    D_loss = real_loss + fake_loss
    D_loss.backward()
    opt_D.step()

    # sampled SDF discriminator step

    if i % args.G_step == 0:
      # whole model training
      model.zero_grad()
      pt_samples1 = random_samples_within(bounds, sample_size)
      got, got_n = model.vals_normal(pt_samples1, latent_noise)

      s1_fool = D(pt_samples1, got)
      fooling_loss = F.binary_cross_entropy_with_logits(s1_fool, torch.ones_like(s1_fool))

      G_loss = fooling_loss
      (G_loss + args.eikonal_weight*eikonal_loss(got_n)).backward()
      opt_G.step()

    scaled_training_step(
      target, bounds, max_b,
      latent_noise,
      model, opt_G,
      D, opt_D,
      args
    )
    #if i != 0 and i % args.save_freq == 0: save(model, disc, args)
    t.set_postfix(D=f"{D_loss:.03f}", G=f"{G_loss:.03f}")

sphere = lambda x: torch.linalg.norm(x, dim=-1, keepdim=True) - 1
rand_sphere = lambda x: torch.linalg.norm(x, dim=-1, keepdim=True) - (random.random() + 0.1)
def rand_torus(p):
  x,y,z = p.split([1,1,1], dim=-1)
  q = torch.stack([torch.cat([x,z], dim=-1).norm(dim=-1) - random.random(), y.squeeze(-1)], dim=-1)
  out = q.norm(dim=-1,keepdim=True) - random.random()
  return out

def origin_aabb(size:float):
  def aux(x):
    v = x.abs() - size
    return v.clamp(min=1e-6).norm(keepdim=True, dim=-1) + \
      v.max(dim=-1, keepdim=True)[0].clamp(max=-1e-6)
  return aux

def intersection(a, b):
  return lambda x: a(x).maximum(b(x))

def values_normals(sdf, pts):
  assert(pts.requires_grad)
  values = sdf(pts)
  if values.shape[-1] != 1: values = values.unsqueeze(-1)
  assert(values.shape[-1] == 1), f"unsqueeze a dimension from {values.shape}"
  normals = autograd(pts, values)
  return values, normals

class MLP(nn.Module):
  def __init__(
    self,
    latent_size:int=32,
    bounds:float=1.5,
  ):
    super().__init__()
    self.latent_size = latent_size
    self.assigned_latent = None
    self.mlp = SkipConnMLP(
      in_size=3, out=1,
      latent_size=latent_size,
      enc=FourierEncoder(input_dims=3),
      activation=torch.sin,
      num_layers=5, hidden_size=256,
      skip=3,
      xavier_init=True,
    )
    self.bounds = bounds
  def forward(self, x, latent=None):
    l = latent if latent is not None else self.assigned_latent.expand(*x.shape[:-1], -1)
    predicted = self.mlp(x, l)
    if self.training: return predicted

    v = x.abs() - self.bounds
    bounds = v.clamp(min=1e-6).norm(keepdim=True, dim=-1) + \
      v.max(dim=-1, keepdim=True)[0].clamp(max=-1e-6)
    return predicted.maximum(bounds)
  def vals_normal(self, x, latent=None):
    with torch.enable_grad():
      pts = x if x.requires_grad else x.requires_grad_()
      values = self(pts, latent)
      normals = autograd(pts, values[..., 0])
      return values, normals
  def set_to(self, sdf):
    opt = optim.Adam(self.parameters(), lr=1e-3)
    t = trange(1024)
    for i in t:
      opt.zero_grad()
      pts = torch.randn(1024, 3, device=device)
      # TODO move this out of loop? Then only assigns one value to sphere.
      latent_noise = torch.randn(1024, self.latent_size, device=device)
      loss = F.mse_loss(self(pts, latent_noise), sdf(pts))
      loss.backward()
      opt.step()
      t.set_postfix(l2=f"{loss:.03f}")

# renders an SDF using the given camera and crop=(top, left, width, height), of size of the
# image.
def render(
  sdf, cam, crop,
  size,
):
  ii, jj = torch.meshgrid(
    torch.arange(size, device=device, dtype=torch.float),
    torch.arange(size, device=device, dtype=torch.float),
  )
  positions = torch.stack([ii.transpose(-1, -2), jj.transpose(-1, -2)], dim=-1)
  t,l,h,w = crop
  positions = positions[t:t+h,l:l+w,:]
  rays = cam.sample_positions(positions, size=size, with_noise=0)

  r_o, r_d = rays.split([3,3], dim=-1)

  pts, hits, _, _ = bisect(sdf, r_o, r_d, eps=0, near=2, far=6)
  pts = pts.requires_grad_()

  with torch.enable_grad():
    vals, normals = sdf.vals_normal(pts) if isinstance(sdf, MLP) else values_normals(sdf, pts)
  normals = F.normalize(normals, dim=-1)
  normals = (normals+1)/2
  normals[~hits] = 0
  return normals

def save(model, disc, args):
  torch.save(model, "models/G_sdf.pt")
  torch.save(disc, "models/D_sdf.pt")

def load_model(args):
  # need to subtract a little bit from the bounds otherwise they will never be properly trained.
  model = MLP(bounds=args.bounds)
  # binary classification
  discrim = PointNet(
    feature_size=4, classes=1,
    # This overfits to the data?
    #enc=FourierEncoder(input_dims=4),
  )
  return model.to(device), discrim.to(device)

#torch.autograd.set_detect_anomaly(True); print("HAS DEBUG")
def main():
  args = arguments()
  if not args.load:
    model, discrim = load_model(args)
  else:
    model = torch.load("models/G_sdf.pt")
    discrim = torch.load("models/D_sdf.pt")
  opt_G = optim.Adam(model.parameters(), lr=3e-4)
  opt_D = optim.Adam(discrim.parameters(), lr=1e-4, weight_decay=1e-5)
  #model.set_to(sphere)
  #targets = [rand_sphere, rand_torus]
  targets = [sphere]
  train(
    targets, model, opt_G,
    discrim, opt_D,

    args,
  )
  save(model, discrim, args)

  sz = args.render_size
  # TODO render many to see outputs
  with torch.no_grad():
    start_l = torch.randn(model.latent_size, device=device)
    end_l = torch.randn(model.latent_size, device=device)
    nts = args.num_test_samples
    for i in trange(nts):

      cam = OrthogonalCamera(
        pos = torch.tensor(
          [[2*math.cos(i/6),2*math.sin(i/6), 3]],
          device=device, dtype=torch.float,
        ),
        at = torch.tensor([[0,0,0]], device=device, dtype=torch.float),
        up = torch.tensor([[0,1,0]], device=device, dtype=torch.float),
        view_width=1.75,
      )

      model.assigned_latent = start_l * (1 - i/nts) + end_l * i/nts
      # will create a random latent noise each time
      normals = render(model, cam, (0,0,sz,sz), sz)
      save_image(f"outputs/normals_{i:03}.png", normals.squeeze(0))

    normals_exp = render(sphere, cam, (0,0,sz,sz), sz)
    save_image("outputs/normals_expected.png", normals_exp.squeeze(0))
  return

if __name__ == "__main__": main()
