# Global runner for all NeRF methods.
# For convenience, we want all methods using NeRF to use this one file.

import argparse
from tqdm import trange
import torch
import torch.optim as optim
import torch.nn.functional as F
import random

import src.loaders
from src.cameras import ( generate_positions, generate_continuous )
import src.nerf as nerf
from src.utils import save_image

def arguments():
  a = argparse.ArgumentParser()
  a.add_argument("-d", "--data", help="path to data")
  a.add_argument(
    "--data-kind", help="kind of data to load", choices=["original"], default="original"
  )
  a.add_argument("--size", help="size to train or render at", type=int, default=16)
  a.add_argument("--epochs", help="number of epochs to train for", type=int, default=100_000)
  a.add_argument("--batch-size", help="size of each training batch", type=int, default=8)
  a.add_argument("--sample-size", help="size of each training batch", type=int, default=16)
  # TODO add more arguments here
  return a.parse_args()

# TODO automatic device discovery here

device = "cpu"
if torch.cuda.is_available():
  device = torch.device("cuda:0")
  torch.cuda.set_device(device)

def render(
  model,
  cam,
  uv,

  # how big should the image be
  size,
  # what is the total size of the visible region
  crop_size,

  device="cuda",
  with_noise=1e-2,
):
  batch_dims = len(cam)

  u = max(min(uv[0], size-crop_size), 0)
  v = max(min(uv[1], size-crop_size), 0)

  sub_g_x, sub_g_y = torch.meshgrid(
    torch.arange(u, u+crop_size, device=device, dtype=torch.float),
    torch.arange(v, v+crop_size, device=device, dtype=torch.float),
  )

  positions = torch.stack([sub_g_y, sub_g_x], dim=-1)

  rays = cam.sample_positions(
    positions, size=size, with_noise=with_noise,
  )
  return model(rays)

# loads the dataset
def load(args, training=True):
  kind = args.data_kind
  if kind == "original":
    assert(args.data is not None)
    return src.loaders.load_original(args.data, training=training, size=args.size, device=device)
  else:
    raise NotImplementedError(kind)

def train(model, cam, labels, opt, args):
  t = trange(args.epochs)
  batch_size = args.batch_size
  for i in t:
    opt.zero_grad()

    idxs = random.sample(range(len(cam)), batch_size)

    u = random.randint(0, args.size-args.sample_size)
    v = random.randint(0, args.size-args.sample_size)
    out = render(
      model, cam[idxs, ...], (u,v), size=args.size, crop_size=args.sample_size, with_noise=False
    )
    ref = labels[idxs][:, u:u+args.sample_size, v:v+args.sample_size, :]
    loss = F.mse_loss(out, ref)
    loss.backward()
    loss = loss.item()
    opt.step()
    t.set_postfix(loss=f"{loss:03f}", refresh=False)

def test(model, cam, labels, args):
  with torch.no_grad():
    for i in range(labels.shape[0]):
      out = render(
        model, cam[None, i], (0,0), size=args.size, crop_size=args.size, with_noise=False
      ).squeeze(0)
      loss = F.mse_loss(out, labels[i])
      save_image(f"outputs/out_{i:03}.png", out)


def main():
  args = arguments()
  labels, cam = load(args)
  model = nerf.PlainNeRF(latent_size=0, device=device).to(device)

  opt = optim.AdamW(model.parameters(), lr=8e-5, weight_decay=0)
  train(model, cam, labels, opt, args)

  test_labels, test_cam = load(args, training=False)
  test(model, test_cam, test_labels, args)

if __name__ == "__main__": main()

