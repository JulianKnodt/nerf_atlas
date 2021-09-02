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

def arguments():
  a = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  a.add_argument(
    # TODO add more here to learn between
    "--target", choices=["sphere"], default="sphere", help="What kind of SDF to learn",
  )
  a.add_argument(
    "--epochs", type=int, default=2000, help="Number of epochs to train for",
  )
  a.add_argument(
    "--bounds", type=float, default=1.5, help="Bounded region to train SDF in",
  )
  a.add_argument(
    "--batch-size", type=int, default=6, help="Number of batches to train at the same time",
  )
  a.add_argument(
    "--sample-size", type=int, default=1<<13, help="Number of points to train per batch",
  )
  a.add_argument(
    "--G-step", type=int, default=1, help="Number of steps to take before optimizing G",
  )
  a.add_argument(
    "--eikonal-weight", type=float, default=10, help="Weight of eikonal loss",
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

def stratified_rand_samples(bounds: [..., 6], samples_per_dim:int=32):
  ll, ur = bounds.split([3,3], dim=-1)
  dim = torch.arange(0,1,samples_per_dim, device=ll.device, dtype=torch.float)
  samples = torch.stack(torch.meshgrid(dim, dim, dim), dim=-1)
  samples = samples - (torch.rand_like(samples) - 0.5)
  samples = ll.unsqueeze(-2) + samples * (ur-ll).unsqueeze(-2)
  return samples

# computes boundes for a bunch of samples.
def bounds_for(samples: [..., "Batch", 3]):
  ll = samples.min(dim=-2)[0]
  ur = samples.max(dim=-2)[0]
  return torch.cat([ll, ur], dim=-1)

# trains a GAN with a target SDF, as well as a discriminator point net.
def train(
  target, model, opt_G,

  discriminator, opt_D,

  args,
):
  D = discriminator
  min_b = -abs(args.bounds)
  max_b = abs(args.bounds)
  bounds = torch.tensor([
    min_b, min_b, min_b,
    max_b, max_b, max_b,
  ], device=device, dtype=torch.float)
  sample_size = args.sample_size
  batch_size = args.batch_size
  bounds = bounds[None,None,:].expand(batch_size, sample_size, 6)
  t = trange(args.epochs)
  G_loss = 0
  for i in t:
    pt_samples1 = random_samples_within(bounds, sample_size)
    D.zero_grad()
    exp, exp_n = values_normals(target, pt_samples1)
    latent_noise = torch.randn(batch_size, 1, model.latent_size, device=device)\
      .expand(batch_size, sample_size, model.latent_size)
    got, got_n = model.vals_normal(pt_samples1, latent_noise)

    s1_real = D(pt_samples1, torch.cat([exp, exp_n], dim=-1))
    s1_fake = D(pt_samples1, torch.cat([got, got_n], dim=-1).detach())

    real_loss = F.binary_cross_entropy_with_logits(s1_real, torch.ones_like(s1_real))
    fake_loss = F.binary_cross_entropy_with_logits(s1_fake, torch.zeros_like(s1_fake))

    D_loss = real_loss + fake_loss
    D_loss.backward()
    opt_D.step()

    if i % args.G_step == 0:
      model.zero_grad()
      pt_samples1 = random_samples_within(bounds, sample_size)
      got, got_n = model.vals_normal(pt_samples1, latent_noise)

      s1_fool = D(pt_samples1, torch.cat([got, got_n], dim=-1))
      fooling_loss = F.binary_cross_entropy_with_logits(s1_fool, torch.ones_like(s1_fool))

      G_loss = fooling_loss + args.eikonal_weight*eikonal_loss(got_n)
      G_loss.backward()
      opt_G.step()
    t.set_postfix(D=f"{D_loss:.03f}", G=f"{G_loss:.03f}")

sphere = lambda x: torch.linalg.norm(x, dim=-1, keepdim=True) - 1
def origin_aabb(size:float):
  def aux(x):
    v = x.abs() - size
    return v.clamp(min=0).norm(keepdim=True, dim=-1) + v.max(dim=-1, keepdim=True)[0].clamp(max=0)
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
      num_layers=6, hidden_size=256,
      xavier_init=True,
    )
    self.bounds = origin_aabb(bounds)
  def forward(self, x, latent=None):
    l = latent if latent is not None else self.assigned_latent.expand(*x.shape[:-1], -1)
    return self.mlp(x, l).maximum(self.bounds(x))
  def vals_normal(self, x, latent=None):
    with torch.enable_grad():
      pts = x if x.requires_grad else x.requires_grad_()
      values = self(pts, latent)
      normals = autograd(pts, values[..., 0])
      return values, normals

# renders a learned SDF
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

  vals, normals = sdf.vals_normal(pts) if isinstance(sdf, MLP) else values_normals(sdf, pts)
  normals = (normals+1)/2
  normals[~hits] = 0
  #return hits.float().unsqueeze(-1).expand(*hits.shape, 3)
  return normals


def load_model(args):
  model = MLP(bounds=args.bounds).to(device)
  # binary classification
  discrim = PointNet(feature_size=7, classes=1).to(device)
  return model, discrim

def main():
  args = arguments()
  model, discrim = load_model(args)
  opt_G = optim.Adam(model.parameters(), lr=1e-4)
  opt_D = optim.Adam(discrim.parameters(), lr=1e-4)
  #target = intersection(sphere, origin_aabb(args.bounds))
  target = sphere
  train(
    target, model, opt_G,
    discrim, opt_D,

    args,
  )
  model.assigned_latent = torch.randn(model.latent_size, device=device)
  cam = OrthogonalCamera(
    pos = torch.tensor([[0,0,-3]], device=device, dtype=torch.float),
    at = torch.tensor([[0,0,0]], device=device, dtype=torch.float),
    up = torch.tensor([[0,1,0]], device=device, dtype=torch.float),
    view_width=2,
  )

  sz = 256
  # TODO render many to see outputs
  normals = render(model, cam, (0,0,sz,sz), sz)
  normals_exp = render(sphere, cam, (0,0,sz,sz), sz)
  save_image("outputs/normals.png", normals.squeeze(0))
  save_image("outputs/normals_expected.png", normals_exp.squeeze(0))
  return

if __name__ == "__main__": main()
