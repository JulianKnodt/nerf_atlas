# A bunch of loaders for various NeRF datasets.
# Each loader returns the dataset, as well a camera model which can be constructed
# from the returned type
# Loader(...) -> Labels, Camera, Optional<Lights>

import json
import os

import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms.functional as TVF
import numpy as np
import imageio

from . import cameras
from .utils import load_image
import src.lights as lights

# loads the dataset
def load(args, training=True, device="cuda"):
  assert(args.data is not None)
  kind = args.data_kind
  if args.derive_kind:
    if args.data.endswith(".mp4"): kind = "single_video"
    elif args.data.endswith(".jpg"): kind = "pixel-single"

  with_mask = (args.model == "sdf" or args.volsdf_alternate) and training
  size = args.size
  if kind == "original":
    return original(
      args.data, training=training, normalize=False, size=size,
      white_bg=args.bg=="white",
      with_mask = with_mask,
      device=device,
    )
  elif kind == "nerv_point":
    return nerv_point(
      args.data, training=training, size=size,
      with_mask = with_mask,
      multi_point = args.nerv_multi_point,
      device=device,
    )
  elif kind == "dtu":
    return dtu(
      args.data, training=training, size=size,
      with_mask = with_mask,
      device=device,
    )
  elif kind == "dnerf":
    return dnerf(
      args.data, training=training, size=size, time_gamma=args.time_gamma,
      white_bg=args.bg=="white", device=device,
    )
  elif kind == "single_video":
    return single_video(args.data)
  elif kind == "pixel-single":
    img, cam = single_image(args.data)
    setattr(args, "img", img)
    args.batch_size = 1
    return img, cam
  elif kind == "shiny":
    shiny(args.data)
    raise NotImplementedError()
  else: raise NotImplementedError(f"load data: {kind}")


def original(
  dir=".", normalize=True, training=True, size=256, white_bg=False, with_mask=False,
  device="cuda",
):
  kind = "train" if training else "test"
  tfs = json.load(open(dir + f"transforms_{kind}.json"))
  channels = 3 + with_mask

  exp_imgs = []
  cam_to_worlds = []
  focal = 0.5 * size / np.tan(0.5 * float(tfs['camera_angle_x']))
  for i, frame in enumerate(tfs["frames"]):
    fp = frame['file_path']
    if fp == "":
      # have to special case empty since nerfactor didn't fill in their blanks
      fp = f"test_{i:03}/nn"
    img = load_image(os.path.join(dir, fp + '.png'), resize=(size, size))
    if white_bg: img = img[..., :3]*img[..., -1:] + (1-img[..., -1:])
    exp_imgs.append(img[..., :channels])
    tf_mat = torch.tensor(frame['transform_matrix'], dtype=torch.float, device=device)[:3, :4]
    if normalize: tf_mat[:3, 3] = F.normalize(tf_mat[:3, 3], dim=-1)
    cam_to_worlds.append(tf_mat)

  cam_to_worlds = torch.stack(cam_to_worlds, dim=0).to(device)
  exp_imgs = torch.stack(exp_imgs, dim=0).to(device)
  if with_mask:
    exp_imgs[..., -1] = (exp_imgs[..., -1] - 1e-5).ceil()

  return exp_imgs, cameras.NeRFCamera(cam_to_worlds, focal), None

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

  return (exp_imgs, times), cameras.NeRFCamera(cam_to_worlds, focal), None

def dtu(path=".", training=True, size=256, with_mask=False, device="cuda"):
  import cv2

  num_imgs = 0
  exp_imgs = []
  image_dir = os.path.join(path, "image")
  for f in sorted(os.listdir(image_dir)):
    if f.startswith("._"): continue
    num_imgs += 1
    img = load_image(os.path.join(image_dir, f), resize=(size, size)).to(device)
    exp_imgs.append(img)

  exp_imgs = torch.stack(exp_imgs, dim=0).to(device)

  if with_mask:
    exp_masks = []
    mask_dir = os.path.join(path, "mask")
    for f in sorted(os.listdir(mask_dir)):
      if f.startswith("._"): continue
      mask = load_image(os.path.join(mask_dir, f), resize=(size, size)).to(device)
      exp_masks.append(mask.max(dim=-1)[0].ceil())
    exp_masks = torch.stack(exp_masks, dim=0).to(device)
    exp_imgs = torch.cat([exp_imgs, exp_masks], dim=-1)

  tfs = np.load(os.path.join(path, "cameras.npz"))
  Ps = [tfs[f"world_mat_{i}"] @ tfs[f"scale_mat_{i}"]  for i in range(num_imgs)]
  def KRt_from_P(P):
    K, R, t, _, _, _, _ = cv2.decomposeProjectionMatrix(P)
    K = K/K[2,2]
    intrinsics = np.eye(4)
    intrinsics[:3, :3] = K

    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = R.transpose()
    pose[:3,3] = (t[:3] / t[3])[:,0]
    return torch.from_numpy(intrinsics).float().to(device),\
      torch.from_numpy(pose).float().to(device)
  intrinsics, poses = list(zip(*[KRt_from_P(p[:3, :4]) for p in Ps]))
  poses = torch.stack(poses, dim=0)
  # normalize distance to be at most 1 for convenience
  max_dist = torch.linalg.norm(poses[:, :3, 3], dim=-1).max()
  poses[:, :3, 3] /= max_dist
  intrinsics = torch.stack(intrinsics, dim=0)

  return exp_imgs, cameras.DTUCamera(pose=poses, intrinsic=intrinsics), None

# https://docs.google.com/document/d/1KI7YtWl3nAuS6xH2WFWug87o-1G6PP4GHrnNzZ0LeUk/edit
def nerv_point(path=".", training=True, size=200, multi_point=False, with_mask=False, device="cuda"):
  import imageio
  def load_exr(path): return torch.from_numpy(imageio.imread(path))

  if training: path = path + f"train_point/"
  kind = "train" if training else "test"
  tfs = json.load(open(path + f"transforms_{kind}.json"))
  exp_imgs = []
  exp_masks = []
  light_locs = []
  light_weights = []
  focal = 0.5 * size / np.tan(0.5 * float(tfs['camera_angle_x']))
  cam_to_worlds=[]

  frames = tfs["frames"]
  frames = frames[:100] if multi_point else frames[100:]
  for frame in frames:
    img = load_exr(os.path.join(path, frame['file_path'] + '.exr')).permute(2,0,1)
    img = TVF.resize(img, size=(size, size))
    img[:3,...] = TVF.adjust_gamma(img[:3,...].clamp(min=1e-10), 1/2.2)
    img = img.permute(1,2,0)
    exp_imgs.append(img[..., :3])
    exp_masks.append((img[..., 3] - 1e-5).ceil())
    tf_mat = torch.tensor(frame['transform_matrix'], dtype=torch.float, device=device)[:3, :4]

    cam_to_worlds.append(tf_mat)
    # also have to update light positions since normalizing to unit sphere
    ll = torch.tensor(frame['light_loc'], dtype=torch.float, device=device)
    if len(ll.shape) == 1: ll = ll[None, ...]
    light_locs.append(ll)
    w = torch.tensor(frame.get('light_weights', [[1,1,1]]),dtype=torch.float,device=device)
    w = w[..., :3]
    # TODO figure out the weight of other lights
    weights = 100 if w.shape[0] == 1 else 66
    light_weights.append(w * weights)

  exp_imgs = torch.stack(exp_imgs, dim=0).to(device).clamp(min=0, max=1)
  if with_mask:
    exp_masks = torch.stack(exp_masks, dim=0).to(device)
    exp_imgs = torch.cat([exp_imgs, exp_masks.unsqueeze(-1)], dim=-1)

  light_locs = torch.stack(light_locs, dim=0).to(device)
  cam_to_worlds = torch.stack(cam_to_worlds, dim=0).to(device)

  light_weights = torch.stack(light_weights, dim=0)
  light = lights.Point(center=light_locs, intensity=light_weights)
  assert(exp_imgs.isfinite().all())
  return exp_imgs, cameras.NeRFCamera(cam_to_worlds, focal), light.to(device)


# taken from https://github.com/nex-mpi/nex-code/blob/main/utils/load_llff.py#L59
# holy balls their code is illegible, I don't know how to reproduce it.
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
  return imgs, cameras.NeRFCamera(poses, focal=fx), None

def single_video(path, training=True, size=256, device="cuda"):
  frames, _, _ = torchvision.io.read_video(path, pts_unit='sec')
  frames = (frames[:100]/255).to(device)
  return frames, cameras.NeRFMMCamera.identity(len(frames), device=device), None

def single_image(path, training=True, size=256, device="cuda"):
  img = torchvision.io.read_image(path).to(device)
  img = torchvision.transforms.functional.resize(img, size=(size, size))
  img = img.permute(1,2,0)/255
  return img.unsqueeze(0), cameras.NeRFCamera.identity(1, device=device), None
