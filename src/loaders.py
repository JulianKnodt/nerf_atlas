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

from .cameras import NeRFCamera
from .utils import load_image

def original(dir=".", normalize=True, training=True, size=256, device="cuda"):
  kind = "train" if training else "test"
  tfs = json.load(open(dir + f"transforms_{kind}.json"))

  exp_imgs = []
  cam_to_worlds = []
  focal = 0.5 * size / np.tan(0.5 * float(tfs['camera_angle_x']))
  focal = tfs['camera_angle_x']
  for frame in tfs["frames"]:
    img = load_image(os.path.join(dir, frame['file_path'] + '.png'), resize=(size, size))
    exp_imgs.append(img[..., :3])
    tf_mat = torch.tensor(frame['transform_matrix'], dtype=torch.float, device=device)[:3, :4]
    if normalize:
      tf_mat[:3, 3] = F.normalize(tf_mat[:3, 3], dim=-1)
    cam_to_worlds.append(tf_mat)

  cam_to_worlds = torch.stack(cam_to_worlds, dim=0).to(device)
  exp_imgs = torch.stack(exp_imgs, dim=0).to(device)

  return exp_imgs, NeRFCamera(cam_to_worlds, focal)

def dnerf(dir=".", normalize=True, training=True, size=256, device="cuda"):
  kind = "train" if training else "test"
  tfs = json.load(open(dir + f"transforms_{kind}.json"))
  exp_imgs = []
  cam_to_worlds = []
  times = []

  focal = tfs['camera_angle_x']
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

  return (exp_imgs, times), NeRFCamera(cam_to_worlds, focal)

def single_video(path, training=True, size=256, device="cuda"):
  frames, _, _ = torchvision.io.read_video(path, pts_unit='sec')
  frames = frames[:100]
  return frames, NeRFMMCamera.identity(len(frames), device=device)

def single_image(path, training=True, size=256, device="cuda"):
  img = torchvision.io.read_image(path).to(device)
  img = torchvision.transforms.functional.resize(img, size=(size, size))
  img = img.permute(1,2,0)/255
  return img.unsqueeze(0), NeRFCamera.identity(1, device=device)

def monocular_video(path=".", training=True, size=256, device="cuda"):
  return NeRFCamera.empty(len(vid))

def load_cityscapes(dir=".", size=256, training=True, device="cuda"):
  kind = "train" if training else "test"
  cs = torchvision.datasets.Cityscapes(root=dir, split=kind, mode="fine")
  # TODO download city scape to test.
  return cs, NeRFCamera.identity(len(cs))
