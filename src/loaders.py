# A bunch of loaders for various NeRF datasets.
# Each loader returns the dataset, as well a camera model which can be constructed
# from the returned type
# Loader(...) -> Labels, Camera

import json
import os
import torchvision
import torch
import torch.nn.functional as F
import numpy as np

from .cameras import NeRFCamera
from .utils import load_image

def load_original(dir=".", normalize=True, training=True, size=256, device="cuda"):
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

