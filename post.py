# This file exists purely for post-processing of models.
# For doing such thing as extracting geometry, running tests, etc.

import argparse
import torch
import torch.nn as nn
import random
import numpy as np
import matplotlib.pyplot as plt
from src.processing import marching_cubes
import src.nerf as nerf

from tqdm import trange

def arguments():
  a = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  a.add_argument("--load", help="Which model to process", type=str, required=True)
  a.add_argument("--seed", help="Random seed", type=int, default=42)
  a.add_argument("--size", help="Render size", type=int, default=512)

  subparser = a.add_subparsers()
  mc = subparser.add_parser("marching-cubes", help="Run marching-cubes on given model")
  #mc.add_argument(
  #  "--marching-cube", help="Extract mesh with marching cubes", action="store_true",
  #)
  mc.add_argument(
    "-o", "--out", help="Where to save mesh from marching cubes", type=str,
    default="outputs/model.stl",
  )
  mc.set_defaults(func=march)

  sm = subparser.add_parser("--sphere-march", help="Run sphere-marching on SDF")
  sm.add_argument("--steps", type=int, default=128, help="# of sphere marching steps")
  sm.add_argument("--radius", type=float, default=1, help="Distance from origin for camera")
  sm.set_defaults(func=sphere_march)
  return a.parse_args()

# TODO better automatic device discovery here
device = "cpu"
if torch.cuda.is_available():
  device = torch.device("cuda:0")
  torch.cuda.set_device(device)

def seed(s):
  torch.manual_seed(s)
  random.seed(s)
  np.random.seed(s)

def march(model, args):
  # TODO find SDF of model and only run on that.
  out = marching_cubes(model)
  raise NotImplementedError()
  exit()

def sphere_march(model, args):
  assert(isinstance(model, nerf.VolSDF)), "Cannot sphere march non-sdf"
  # TODO get camera views
  elev = torch.linspace(0, math.pi/2, 12, device=device)
  azim = torch.linspace(0, 2*math.pi, 12, device=device)
  for el in elev:
    for az in azim:
      c2w = utils.spherical_pose(el, az, args.radius)
      cam = NeRFCamera(cam_to_world=c2w, focal=0.5)
      ii, jj = torch.meshgrid(
        torch.arange(size, device=device, dtype=torch.float),
        torch.arange(size, device=device, dtype=torch.float),
      )
      positions = torch.stack([ii.transpose(-1, -2), jj.transpose(-1, -2)], dim=-1)
      rays = cam.sample_positions(positions, size=size)
      out = model.sdf.rays(rays)
      print(out.shape)
      exit()
  raise NotImplementedError()


def main():
  args = arguments()
  seed(args.seed)
  exit()
  model = torch.load(args.load, map_location=device)
  args.func(model, args)

  raise NotImplementedError()

if __name__ == "__main__": main()

