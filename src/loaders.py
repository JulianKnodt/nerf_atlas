# A bunch of loaders for various NeRF datasets.
# Each loader returns the dataset, as well a camera model which can be constructed
# from the returned type
# Loader(...) -> Labels, Camera

import json
import os

import torch
import torch.nn.functional as F
import torchvision
import numpy as np
import imageio

from . import cameras
from .utils import load_image

def original(dir=".", normalize=True, training=True, size=256, white_bg=False, device="cuda"):
  kind = "train" if training else "test"
  tfs = json.load(open(dir + f"transforms_{kind}.json"))

  exp_imgs = []
  cam_to_worlds = []
  focal = 0.5 * size / np.tan(0.5 * float(tfs['camera_angle_x']))
  for frame in tfs["frames"]:
    img = load_image(os.path.join(dir, frame['file_path'] + '.png'), resize=(size, size))
    if white_bg: img = img[..., :3]*img[..., -1:] + (1-img[..., -1:])
    exp_imgs.append(img[..., :3])
    tf_mat = torch.tensor(frame['transform_matrix'], dtype=torch.float, device=device)[:3, :4]
    if normalize:
      tf_mat[:3, 3] = F.normalize(tf_mat[:3, 3], dim=-1)
    cam_to_worlds.append(tf_mat)

  cam_to_worlds = torch.stack(cam_to_worlds, dim=0).to(device)
  exp_imgs = torch.stack(exp_imgs, dim=0).to(device)

  return exp_imgs, cameras.NeRFCamera(cam_to_worlds, focal)

def dnerf(
  dir=".", normalize=False, training=True,
  size=256, time_gamma=True, white_bg=False,
  device="cuda"
):
  kind = "train" if training else "test"
  tfs = json.load(open(dir + f"transforms_{kind}.json"))
  exp_imgs = []
  cam_to_worlds = []
  times = []

  focal = 0.5 * size / np.tan(0.5 * float(tfs['camera_angle_x']))
  n_frames = len(tfs["frames"])
  for t, frame in enumerate(tfs["frames"]):
    img = load_image(os.path.join(dir, frame['file_path'] + '.png'), resize=(size, size))
    if white_bg: img = img[..., :3] * img[..., -1:] + (1-img[..., -1:])
    exp_imgs.append(img[..., :3])
    tf_mat = torch.tensor(frame['transform_matrix'], dtype=torch.float, device=device)[:3, :4]
    if normalize:
      tf_mat[:3, 3] = F.normalize(tf_mat[:3, 3], dim=-1)
    cam_to_worlds.append(tf_mat)
    time = getattr(frame, 'time', float(t) / (n_frames-1))
    times.append(time)

  assert(sorted(times) == times), "Internal: assume times are sorted"
  # TODO sort by time if not already sorted.

  cam_to_worlds = torch.stack(cam_to_worlds, dim=0).to(device)
  exp_imgs = torch.stack(exp_imgs, dim=0).to(device)
  times = torch.tensor(times, device=device)

  # This is for testing out DNeRFAE, apply a gamma transform based on the time.
  if time_gamma:
    exp_imgs = exp_imgs.pow((2 * times[:, None, None, None] - 1).exp())

  return (exp_imgs, times), cameras.NeRFCamera(cam_to_worlds, focal)

# taken from https://github.com/nex-mpi/nex-code/blob/main/utils/load_llff.py#L59
def shiny(path, training=True, size=256, device="cuda"):
  #tfs = open(os.path.join(path, "poses_bounds.npy"))
  poses_file = os.path.join(path, "poses_bounds.npy")
  assert(os.path.exists(poses_file))
  poses_arr = np.load(poses_file)
  shape = 5
  if os.path.isfile(os.path.join(path, 'hwf_cxcy.npy')):
    shape = 4
    [h,w,fx,fy,cx,cy] = np.load(os.path.join(path, 'hwf_cxcy.npy'))
    assert(fx == fy), "Internal: assumed that focal x and focal y equal"
  else: raise NotImplementedError()
  poses = poses_arr[:, :-2].reshape([-1, 3, shape])
  # bds is near, far
  bds = poses_arr[:, -2:]

  img_dir = os.path.join(path, 'images')
  assert(os.path.exists(img_dir))
  img_files = [
    os.path.join(img_dir, f) for f in sorted(os.listdir(img_dir)) \
    if f.endswith('JPG') or f.endswith('jpg') or f.endswith('png')
  ]
  imgs = torch.stack([load_image(f, (size, size)) for f in img_files], dim=0).to(device)
  raise NotImplementedError("TODO get camera from poses, bds")
  return imgs, cameras.NeRFCamera(poses, focal=fx)

def single_video(path, training=True, size=256, device="cuda"):
  frames, _, _ = torchvision.io.read_video(path, pts_unit='sec')
  frames = (frames[:100]/255).to(device)
  return frames, cameras.NeRFMMCamera.identity(len(frames), device=device)

def single_image(path, training=True, size=256, device="cuda"):
  img = torchvision.io.read_image(path).to(device)
  img = torchvision.transforms.functional.resize(img, size=(size, size))
  img = img.permute(1,2,0)/255
  return img.unsqueeze(0), cameras.NeRFCamera.identity(1, device=device)

def monocular_video(path=".", training=True, size=256, device="cuda"):
  raise NotImplementedError()
  return NeRFCamera.empty(len(vid))
