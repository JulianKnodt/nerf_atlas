# Global runner for all NeRF methods.
# For convenience, we want all methods using NeRF to use this one file.

import argparse
from tqdm import trange
import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import random

import src.loaders
import src.nerf as nerf
from src.utils import save_image
from src.neural_blocks import Upsampler

def arguments():
  a = argparse.ArgumentParser()
  a.add_argument("-d", "--data", help="path to data")
  a.add_argument(
    "--data-kind", help="kind of data to load", choices=["original"], default="original"
  )
  # various size arguments
  a.add_argument("--size", help="size to train at w/ upsampling", type=int, default=32)
  a.add_argument(
    "--render-size", help="size to render images at w/o upsampling", type=int, default=16
  )

  a.add_argument("--epochs", help="number of epochs to train for", type=int, default=30000)
  a.add_argument("--batch-size", help="size of each training batch", type=int, default=8)
  a.add_argument(
    "--neural-upsample", help="add neural upsampling", default=False, action="store_true"
  )
  a. add_argument(
    "--feature-space",
    help="when using neural upsampling, what is the feature space size",
    type=int, default=16,
  )
  a.add_argument(
    "--model", help="which model do we want to use", type=str,
    choices=["tiny", "plain", "ae"], default="plain",
  )
  a.add_argument(
    "-lr", "--learning-rate", help="learning rate", type=float, default=5e-3,
  )
  a.add_argument(
    "--valid-freq", help="how often validation images are generated", type=int, default=500,
  )
  # TODO add more arguments here
  return a.parse_args()

# TODO better automatic device discovery here

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

  device="cuda",
  with_noise=3e-2,
):
  batch_dims = len(cam)

  ii, jj = torch.meshgrid(
    torch.arange(size, device=device, dtype=torch.float),
    torch.arange(size, device=device, dtype=torch.float),
  )

  positions = torch.stack([ii.transpose(-1, -2), jj.transpose(-1, -2)], dim=-1)

  rays = cam.sample_positions(
    positions, size=size, with_noise=with_noise,
  )
  return model(rays)

# loads the dataset
def load(args, training=True):
  kind = args.data_kind
  if kind == "original":
    assert(args.data is not None)
    return src.loaders.load_original(
      args.data, training=training, normalize=False, size=args.size, device=device
    )
  else:
    raise NotImplementedError(kind)

def train(model, cam, labels, opt, args, sched=None):
  t = trange(args.epochs)
  batch_size = args.batch_size
  for i in t:
    opt.zero_grad()

    idxs = random.sample(range(len(cam)), batch_size)

    out = render(
      model, cam[idxs], (0,0), size=args.render_size,
    )
    ref = labels[idxs]
    loss = F.mse_loss(out, ref)
    loss.backward()
    loss = loss.item()
    opt.step()
    t.set_postfix(loss=f"{loss:03f}", lr=f"{sched.get_last_lr()[0]:.1e}", refresh=False)
    if sched is not None: sched.step()
    if (i % args.valid_freq == 0):
      save_image(f"outputs/valid_{i:03}.png", out[0])

def test(model, cam, labels, args):
  with torch.no_grad():
    for i in range(labels.shape[0]):
      out = render(
        model, cam[None, i], (0,0), size=args.render_size, with_noise=False,
      ).squeeze(0)
      loss = F.mse_loss(out, labels[i])
      print(loss.item())
      save_image(f"outputs/out_{i:03}.png", out)

def load_model(args):
  if not args.neural_upsample: args.feature_space = 3
  if args.model == "tiny":
    model = nerf.TinyNeRF(out_features=args.feature_space, device=device).to(device)
  elif args.model == "plain":
    model = nerf.PlainNeRF(out_features=args.feature_space, device=device).to(device)
  elif args.model == "ae":
    model = nerf.NeRFAE(out_features=args.feature_space, device=device).to(device)
  else:
    raise NotImplementedError(args.model)

  # tack on neural upsampling if specified
  if args.neural_upsample:
    upsampler =  Upsampler(
      in_size=args.render_size,
      out=args.size,

      in_features=args.feature_space,
      out_features=3,
    ).to(device)
    # stick a neural upsampling block afterwards
    model = nn.Sequential(model, upsampler, nn.Sigmoid())
  else:
    model = nn.Sequential(model, nn.Sigmoid())
  return model

# in theory this is an interesting loss function but it runs into numerical stability issues
# probably need to converd the prod().log() into a sum of logs.
#def all_mse_loss(ref, img, eps=1e-2):
#  return ((ref - img).square() + eps).reciprocal().prod().log().neg()

def main():
  args = arguments()

  model = load_model(args)
  # for some reason AdamW doesn't seem to work here
  opt = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=0)

  labels, cam = load(args)

  sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=1e-8)
  train(model, cam, labels, opt, args, sched=sched)

  test_labels, test_cam = load(args, training=False)
  test(model, test_cam, test_labels, args)

if __name__ == "__main__": main()

