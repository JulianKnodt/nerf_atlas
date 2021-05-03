# A bunch of loaders for various NeRF datasets.
# Each loader returns the dataset, as well a camera model which can be constructed
# from the returned type
# Loader(...) -> Labels, data, Camera Class

import json
import os
import torchvision
import torch
import numpy as np

from .cameras import NeRFCamera

def load_original(dir=".", normalize=True, training=True, size=256):
  kind = "train" if training else "test"
  tfs = json.load(open(dir + f"transforms_{kind}.json"))

  loader = torchvision.transforms.resize(size)
  exp_imgs = []
  cam_to_worlds = []
  focal = 0.5 * size / np.tan(0.5 * float(tfs['camera_angle_x']))
  focal = tfs['camera_angle_x']
  for frame in tfs["frames"]:
    img = loader(torchvision.io.read_image(os.path.join(dir, frame['file_path'] + '.png')))
    exp_imgs.append(img[..., :3])
    tf_mat = torch.tensor(frame['transform_matrix'], dtype=torch.float, device=device)[:3, :4]
    if normalize: tf_mat[:3, 3] = F.normalize(tf_mat[:3, 3], dim=-1)
    cam_to_worlds.append(tf_mat)

  cam_to_worlds = torch.stack(cam_to_worlds, dim=0)
  exp_imgs = torch.stack(exp_imgs, dim=0)

  return exp_imgs, (cam_to_worlds, focal), NeRFCamera

