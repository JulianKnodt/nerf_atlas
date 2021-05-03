import torch
import torch.nn.functional as F
from dataclasses import dataclass
from ..utils import rotate_vector

# General Camera interface
@dataclass
class Camera():
  camera_to_world = None
  world_to_camera = None
  # samples from positions in [0,1] screen space to global
  def sample_positions(self, positions, sampler, bundle_size):
    raise NotImplementedError()

# A camera made specifically for generating rays from NeRF models
@dataclass
class NeRFCamera(Camera):
  cam_to_world:torch.tensor = None
  focal: float=None
  device:str ="cuda"
  def __len__(self): return self.cam_to_world.shape[0]

  def sample_positions(
    self,
    position_samples,
    sampler, bundle_size=4,
    size=512,
    with_noise=False,
    N=1,
  ):
    device=self.device
    u,v = position_samples.split(1, dim=-1)
    # u,v each in range [0, size]
    if with_noise:
      u = u + (torch.rand_like(u)-0.5)*with_noise
      v = v + (torch.rand_like(v)-0.5)*with_noise

    d = torch.stack(
      [
        # W
        (u - size * 0.5) / self.focal,
        # H
        -(v - size * 0.5) / self.focal,
        -torch.ones_like(u),
      ],
      dim=-1,
    )
    r_d = torch.sum(
      d[..., None, :] * self.cam_to_world[..., :3, :3], dim=-1
    )
    # normalize direction and exchange [W,H,B,3] -> [B,W,H,1,3]
    r_d = F.normalize(r_d, dim=-1).permute(2,0,1,3).unsqueeze(-2)
    r_o = self.cam_to_world[..., :3, -1][:, None, None, None, :].expand_as(r_d)
    return torch.cat([r_o, r_d], dim=-1)

# The camera described in the NeRF-- paper
@dataclass
class NeRFMMCamera(Camera):
  # position
  t: torch.tensor = None
  # angle of rotation about axis
  angle: torch.tensor = None
  axis:torch.tensor = None
  # intrinsic focal positions
  focals: torch.tensor = None
  device:str ="cuda"
  def __len__(self): return self.t.shape[0]
  def parameters(self): return [angle, axis, t, focals]

  def sample_positions(
    self,
    position_samples,
    sampler, bundle_size=4,
    size=512,
    with_noise=False,
    N=1,
  ):
    device=self.device
    u,v = position_samples.split(1, dim=-1)
    # u,v each in range [0, size]
    if with_noise:
      u = u + (torch.rand_like(u)-0.5)*with_noise
      v = v + (torch.rand_like(v)-0.5)*with_noise

    d = torch.stack(
      [
        # W
        (u - size * 0.5) / self.focals[..., 0],
        # H
        -(v - size * 0.5) / self.focals[..., 1],
        -torch.ones_like(u),
      ],
      dim=-1,
    )
    r_d = rotate_vector(d, self.axis, self.angle.cos(), self.angle.sin())
    # normalize direction and exchange [W,H,B,3] -> [B,W,H,1,3]
    r_d = F.normalize(r_d, dim=-1).permute(2,0,1,3).unsqueeze(-2)
    r_o = self.t[:, None, None, None, :].expand_as(r_d)
    return torch.cat([r_o, r_d], dim=-1)

# A camera made specifically for generating rays from NeRV models
@dataclass
class NeRVCamera(Camera):
  world_to_cam: torch.tensor = None
  loc: torch.tensor = None
  focal: float=None
  device:str ="cuda"
  def __len__(self): return self.world_to_cam.shape[0]

  def sample_positions(
    self,
    position_samples,
    sampler, bundle_size=4,
    size=512,
    with_noise=False,
    N=1,
  ):
    device=self.device
    u,v = position_samples.split(1, dim=-1)
    # u,v each in range [0, size]
    d = torch.stack([
        (u - size * 0.5) / self.focal, (v - size * 0.5) / self.focal, torch.ones_like(u),
    ], dim=-1)
    r_d = torch.sum(
      d[..., None] * self.world_to_cam[..., :3, :3], dim=-2
    )
    # normalize direction and exchange [W,H,B,3] -> [B,W,H,1,3]
    r_d = F.normalize(r_d, dim=-1).permute(2,0,1,3).unsqueeze(-2)
    # TODO figure out r_o?
    return torch.cat([r_o, r_d], dim=-1)

def lift(x,y,z,intrinsics, size):
    total_shape = x.shape
    fx = intrinsics[..., 0, 0, None].expand(total_shape)
    fy = intrinsics[..., 1, 1, None].expand(total_shape)
    cx = intrinsics[..., 0, 2, None].expand(total_shape) # size
    cy = intrinsics[..., 1, 2, None].expand(total_shape) # size
    sk = intrinsics[..., 0, 1, None].expand(total_shape)
    x = x.expand(total_shape)
    y = y.expand(total_shape)
    z = z.expand(total_shape)

    x_lift = (x - cx + cy*sk/fy - sk*y/fy) / fx * z
    y_lift = (y - cy) / fy * z

    # homogeneous
    return torch.stack([x_lift, y_lift, z, torch.ones_like(z)], dim=-1)

# A camera specifically for rendering DTU scenes as described in IDR.
@dataclass
class DTUCamera(Camera):
  pose: torch.Tensor = None
  intrinsic: torch.Tensor = None
  device: str = "cuda"
  def __len__(self): return self.pose.shape[0]
  def sample_positions(
    self,
    position_samples,
    sampler, bundle_size=4,
    size=512,
    with_noise=False,
    N=1,
  ):
    device = self.device
    pose = self.pose
    intrinsic = self.intrinsic
    # copied directly from https://github.com/lioryariv/idr/blob/44959e7aac267775e63552d8aac6c2e9f2918cca/code/utils/rend_util.py#L48
    if pose.shape[1] == 7: #In case of quaternion vector representation
      assert(False)
    else: # In case of pose matrix representation
      r_o = pose[:, :3, 3]
    W, H, _ = position_samples.shape
    N = len(self)

    # 1600 is magical because it's the size of the original images given to us
    # In theory it would need to be changed if training on a different image set.
    normalize= torch.tensor([1600, 1200], device=device, dtype=torch.float)/size
    u,v = (position_samples * normalize)\
      .reshape(-1, 2).split(1, dim=-1)
    u = u.reshape(1, -1).expand(N, -1)
    v = v.reshape(1, -1).expand(N, -1)

    points = lift(u, v, torch.ones_like(u), intrinsics=intrinsic, size=size)

    world_coords = torch.bmm(pose, points.permute(0,2,1)).permute(0,2,1)[..., :3]

    r_o = r_o[:, None, :].expand_as(world_coords)
    r_d = F.normalize(world_coords - r_o, dim=-1)

    return torch.cat([r_o, r_d], dim=-1)\
      .reshape(N, W, H, 1, 6)\
      .expand(N, W, H, bundle_size, 6)
