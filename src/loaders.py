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

def original(dir=".", normalize=True, training=True, size=256, device="cuda"):
  kind = "train" if training else "test"
  tfs = json.load(open(dir + f"transforms_{kind}.json"))

  exp_imgs = []
  cam_to_worlds = []
  focal = 0.5 * size / np.tan(0.5 * float(tfs['camera_angle_x']))
  for frame in tfs["frames"]:
    img = load_image(os.path.join(dir, frame['file_path'] + '.png'), resize=(size, size))
    exp_imgs.append(img[..., :3])
    tf_mat = torch.tensor(frame['transform_matrix'], dtype=torch.float, device=device)[:3, :4]
    if normalize:
      tf_mat[:3, 3] = F.normalize(tf_mat[:3, 3], dim=-1)
    cam_to_worlds.append(tf_mat)

  cam_to_worlds = torch.stack(cam_to_worlds, dim=0).to(device)
  exp_imgs = torch.stack(exp_imgs, dim=0).to(device)

  return exp_imgs, cameras.NeRFCamera(cam_to_worlds, focal)

def dnerf(dir=".", normalize=False, training=True, size=256, device="cuda"):
  kind = "train" if training else "test"
  tfs = json.load(open(dir + f"transforms_{kind}.json"))
  exp_imgs = []
  cam_to_worlds = []
  times = []

  focal = 0.5 * size / np.tan(0.5 * float(tfs['camera_angle_x']))
  n_frames = len(tfs["frames"])
  for t, frame in enumerate(tfs["frames"]):
    img = load_image(os.path.join(dir, frame['file_path'] + '.png'), resize=(size, size))
    exp_imgs.append(img[..., :3])
    tf_mat = torch.tensor(frame['transform_matrix'], dtype=torch.float, device=device)[:3, :4]
    if normalize:
      tf_mat[:3, 3] = F.normalize(tf_mat[:3, 3], dim=-1)
    cam_to_worlds.append(tf_mat)
    time = getattr(frame, 'time', float(t) / (n_frames-1))
    times.append(time)

  cam_to_worlds = torch.stack(cam_to_worlds, dim=0).to(device)
  exp_imgs = torch.stack(exp_imgs, dim=0).to(device)
  times = torch.tensor(times, device=device)

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
    #h, w, fx, fy, cx, cy
    intrinsic_arr = np.load(os.path.join(path, 'hwf_cxcy.npy'))
  else: raise NotImplementedError()
  poses = poses_arr[:, :-2].reshape([-1, 3, shape]).transpose([1,2,0])
  bds = poses_arr[:, -2:].transpose([1,0])

  img_dir = os.path.join(path, 'images')
  assert(os.path.exists(img_dir))
  img_files = [
    os.path.join(img_dir, f) for f in sorted(os.listdir(img_dir)) \
    if f.endswith('JPG') or f.endswith('jpg') or f.endswith('png')
  ]
  def imread(f): return imageio.imread(f, ignoregamma=f.endswith('png'))
  exit()
  imgs = np.stack([imread(f)[...,:3]/255. for f in img_files], axis=-1)
  poses, bds, intrinsic_arr, imgs
  raise NotImplementedError("TODO get camera from poses, bds")
  return

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
