import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import math

from .neural_blocks import ( SkipConnMLP, FourierEncoder )
from .utils import ( autograd, elev_azim_to_dir )

def load(args):
  cons = light_kinds.get(args.light_kind, None)
  if cons is None: raise NotImplementedError(f"light kind: {args.light_kind}")

  kwargs = {}
  if cons == "field":
    kwargs["num_embeddings"] = args.num_labels
  return cons(**kwargs)

class Light(nn.Module):
  def __init__(
    self
  ):
    super().__init__()
  def __getitem__(self, _v): return self
  @property
  def supports_idx(self): raise NotImplementedError()
  def forward(self, x): raise NotImplementedError()

class Field(Light):
  def __init__(
      self,
      num_embeddings:int=100,
      embedding_size:int=32,
      monochrome=True,
    ):
    super().__init__()
    self.color_dims = color_dims = 1 if monochrome else 3
    self.mlp = SkipConnMLP(
      in_size=3, out=color_dims+2, hidden_size=256, xavier_init=True,
      enc=FourierEncoder(input_dims=3),
      latent_size=0 if num_embeddings == 1 else embedding_size,
    )
    # since this is a field it doesn't have a specific distance and thus is treated like ambient
    # light by having a far distance.
    self.far_dist = 20

    assert(embedding_size >= 1), "Must have embedding size of at least 1"
    self.num_embeddings = num_embeddings
    if num_embeddings == 1:
      self.embedding = None
      return
    self.embedding = nn.Embedding(num_embeddings, embedding_size)
    self.curr_idx = 0

  @property
  def supports_idx(self): return self.num_embeddings > 1
  def set_idx(self, v): self.curr_idx = v
  def iter(self): yield self
  def forward(self, x, mask=None):
    if mask is not None: raise NotImplementedError()
    own_latent = None if self.embedding is None else \
      self.embedding(self.curr_idx)[None, :, None, None].expand(*x.shape[:-1], -1)
    intensity, elaz = self.mlp(x, own_latent).split([self.color_dims, 2], dim=-1)
    r_d = elev_azim_to_dir(elaz)
    return r_d, self.far_dist, F.relu(intensity).expand_as(x)

class Point(Light):
  def __init__(
    self,
    center = [0,0,0],
    train_center=False,
    intensity=[1],
    train_intensity=False,
    distance_decay=False,
  ):
    super().__init__()
    if type(center) == torch.Tensor: self.center = center
    else:
      center = torch.tensor(center, requires_grad=train_center, dtype=torch.float)
      self.center = nn.Parameter(center)
    self.train_center = train_center

    if type(intensity) == torch.Tensor: self.intensity = intensity
    else:
      if len(intensity) == 1: intensity = intensity * 3
      intensity = torch.tensor(intensity, requires_grad=train_intensity, dtype=torch.float)\
        .expand_as(self.center)
      self.intensity = nn.Parameter(intensity)
    self.train_intensity = train_intensity
    self.distance_decay = distance_decay

  # returns some subset of the training batch
  def __getitem__(self, v):
    return Point(
      center=self.center[v],
      train_center=self.train_center,
      intensity=self.intensity[v],
      train_intensity=self.train_intensity,
      distance_decay=self.distance_decay,
    )
  @property
  def supports_idx(self): return True
  # return a singular light from a batch, altho it may be more efficient to use batching
  # this is conceptually nicer, and allows for previous code to work.
  def iter(self):
    for i in range(self.center.shape[1]):
      yield Point(
        center=self.center[:, i, :],
        train_center=self.train_center,
        intensity=self.intensity[:, i, :],
        train_intensity=self.train_intensity,
        distance_decay=self.distance_decay,
      )
  def forward(self, x, mask=None):
    loc = self.center[:, None, None, :]
    if mask is not None: loc = loc.expand((*mask.shape, 3))[mask]
    # direction from pts to the light
    d = loc - x
    dist = torch.linalg.norm(d, ord=2, dim=-1)
    d = F.normalize(d, eps=1e-6, dim=-1)
    intn = self.intensity[:, None, None, :]
    if mask is not None: intn = intn.expand((*mask.shape, 3,))[mask]
    if self.distance_decay:
      decay = dist.square()
      spectrum = intn/decay.clamp(min=1e-5).unsqueeze(-1)
    else: spectrum=intn

    return d, dist, spectrum

light_kinds = {
  "field": Field,
  "point": Point,
  "dataset": lambda **kwargs: None,
  None: None,
}
