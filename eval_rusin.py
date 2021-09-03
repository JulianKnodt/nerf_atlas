import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import argparse
from src.utils import ( autograd, eikonal_loss, save_image )
from src.neural_blocks import ( SkipConnMLP, FourierEncoder, PointNet )
from src.cameras import ( OrthogonalCamera )
import src.refl as refl
from tqdm import trange
import random

def arguments():
  a = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  a.add_argument(
    "--refl-model", required=True, type=str,
  )
  return a.parse_args()


device="cpu"
if torch.cuda.is_available():
  device = torch.device("cuda:0")
  torch.cuda.set_device(device)

def main():
  args = arguments()
  with torch.no_grad():
    model = torch.load(args.refl_model)
    assert(hasattr(model, "refl")), "The provided model must have a refl"
    r = model.refl
    print("TODO figure out what to with latent", r.latent_size)
    assert(isinstance(r, refl.Rusin)), "must provide a rusin refl"
    degs = torch.cat(torch.meshgrid(
      torch.arange(90, device=device),
      torch.arange(90, device=device),
      torch.arange(180, device=device),
    ))
    rads = torch.deg2rad(rads)
    latent = None
    out = r.raw(rads, latent)
    print(out.shape)

  return

if __name__ == "__main__": main()
