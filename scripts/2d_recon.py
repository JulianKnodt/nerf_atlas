import sys
sys.path[0] = sys.path[0][:-len("scripts/")] # hacky way to treat it as root directory

import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision as tv
from src.neural_blocks import ( SkipConnMLP, FourierEncoder )
from src.utils import ( fat_sigmoid )
from tqdm import trange

ts = 128

class LearnedImage(nn.Module):
  def __init__(self):
    super().__init__()
    self.query = SkipConnMLP(
      in_size=2, out=1, latent_size=0,
      activation=torch.sin, num_layers=5, hidden_size=256, init="siren",
    )
  def forward(self, x):
    return fat_sigmoid(self.query(x))

class PixelImage(nn.Module):
  def __init__(self):
    super().__init__()
    self.data = nn.Parameter(torch.randn(1, 1, 256, 256))
  def forward(self, x):
    B = x.shape[0]
    vals = F.grid_sample(
      self.data.expand(B,-1,-1,-1),
      x,
      padding_mode="border",
      align_corners=False,
    ).permute(0,2,3,1)
    return vals

class LIIF(nn.Module):
  def __init__(
    self,
    reso:int=16,
    emb_size:int=128
  ):
    super().__init__()
    self.grid = nn.Parameter(torch.randn(1, emb_size, reso, reso))
    self.query = SkipConnMLP(
      in_size=emb_size, out=1, latent_size=0,
      activation=torch.sin, num_layers=5, hidden_size=256, init="siren",
    )
  def forward(self, x):
    B = x.shape[0]
    latent = F.grid_sample(
      self.grid.expand(B, -1,-1,-1), x,
      padding_mode="border",
      align_corners=False,
    ).permute(0,2,3,1)
    return fat_sigmoid(self.query(latent))


#torch.autograd.set_detect_anomaly(True); print("DEBUG")

class LongAnimator(nn.Module):
  def __init__(
    self, img,
    segments:int,
    spline:int=4,
    seg_emb_size:int=128,
    anchor_interim:int=128,
  ):
    super().__init__()
    self.img = img
    self.spline_n = spline
    self.ses = ses = seg_emb_size
    segments = int(segments)
    self.segments = segments
    self.midsize = anchor_interim

    self.seg_emb = nn.Embedding(segments+2, ses)
    self.anchors = SkipConnMLP(
      in_size=2, out=2+anchor_interim, latent_size=ses,
      num_layers=5, hidden_size=128, init="xavier",
    )
    self.point_estim=SkipConnMLP(
      in_size=2, out=(spline-2)*2,
      num_layers=5, hidden_size=128,
      latent_size=ses+2*anchor_interim, init="xavier",
    )
  def forward(self, x, t):
    B = t.shape[0]
    t = t[:, None, None, None]
    seg = t.floor().int().clamp(min=0)
    emb = self.seg_emb(torch.cat([seg,seg+1], dim=-1)).expand(-1, *x.shape[1:3], -1, -1)
    anchors, anchor_latent = self.anchors(
      x[..., None, :].expand(B,-1,-1,2,-1),
      emb.expand(-1, *x.shape[1:3], -1, -1),
    ).split([2, self.midsize], dim=-1)
    start, end = [a.squeeze(-2) for a in anchors.split([1,1], dim=-2)]
    point_estim_latent = torch.cat([emb[..., 0, :], anchor_latent.flatten(-2)], dim=-1)
    midpts = torch.stack(self.point_estim(x.expand(B,-1,-1,-1), point_estim_latent).split(2, dim=-1), dim=0)
    ctrl_pts = torch.cat([start[None], midpts, end[None]], dim=0)
    dx = de_casteljau(ctrl_pts, t.frac(), self.spline_n).fmod(1)
    return self.img(x+dx)

# A single Skip Connected MLP
class SimpleAnimator(nn.Module):
  def __init__(self, img, *args, **kwargs):
    self.img = img
    self.pred = SkipConnMLP(
      in_size=1, out=2,
      num_layers=7, hidden_size=512,
      init="xavier",
    )
  def forward(self, x, t):
    B = t.shape[0]
    dx = self.pred(t)[:, None, None, None]
    return self.img(x + dx)

def de_casteljau(coeffs, t, N: int):
  betas = coeffs
  m1t = 1 - t
  for i in range(1, N): betas = betas[:-1] * m1t + betas[1:] * t
  return betas.squeeze(0)

def train(model, ref, times):
  t = trange(25_000)
  batch_size=min(3, times.shape[0])
  grid = torch.stack(torch.meshgrid(
    torch.linspace(-1, 1, ts),
    torch.linspace(-1, 1, ts),
    indexing="ij",
  ),dim=-1).unsqueeze(0).to(device)
  opt = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=0)
  for i in t:
    opt.zero_grad()
    idxs = random.sample(range(times.shape[0]), batch_size)
    got = model(grid, times[idxs])
    exp = ref[idxs]
    loss = F.mse_loss(got, exp)
    #loss = F.l1_loss(got, exp)
    loss.backward()
    opt.step()
    t.set_postfix(L2=f"{loss.item():.02e}")
    if i % 250 == 0:
      with torch.no_grad():
        pred_img = tv.utils.make_grid(got.permute(0,3,1,2))
        exp_img = tv.utils.make_grid(ref[idxs].permute(0,3,1,2))
        result = torch.cat([pred_img, exp_img], dim=1)
        tv.utils.save_image(result, f"outputs/animate_{i:05}.png")
    if i % 1000 == 0 and i != 0:
      torch.save(model, "models/animate_long.pt")
  torch.save(model, "models/animate_long.pt")


def test(ref, model, num_secs, n:int=1800):
  model = model.eval()
  times = torch.linspace(0,num_secs,n,device=device)
  grid = torch.stack(torch.meshgrid(
    torch.linspace(-1, 1, ts),
    torch.linspace(-1, 1, ts),
    indexing="ij",
  ),dim=-1).unsqueeze(0).to(device)

  batch_size = 12
  out = []
  for batch in times.split(batch_size, dim=0):
    out.append(model(grid, batch))
  out = torch.cat(out, dim=0).cpu()
  loss = F.mse_loss(ref, out)
  print("Final Loss", f"{loss.item():.03e}")
  pred_img = tv.utils.make_grid(out.permute(0,3,1,2), num_secs)
  tv.utils.save_image(pred_img, f"outputs/final.png")
  tv.io.write_video("outputs/animation.mp4", out.expand(-1, -1, -1, 3)*255, int(n/num_secs))

device="cuda:0"
def main():
  with torch.no_grad():
    frames, _, info = tv.io.read_video("data/heider/animation.mp4", pts_unit="sec")
    og_frames = frames
    fps = info["video_fps"]
    og_num_frames = frames.shape[0]
    num_secs = int(frames.shape[0]//fps)
    frames = frames[::int(fps//5)].to(device)
    num_frames = frames.shape[0]
    frames = (frames/255).mean(dim=-1, keepdim=True)
    frames = tv.transforms.functional.resize(frames.permute(0,3,1,2), (ts, ts)).permute(0,2,3,1)

  times = torch.linspace(0, num_secs, num_frames).to(device)
  model = LongAnimator(LIIF(), segments=num_secs*2).to(device)
  #model = torch.load("models/animate_long.pt")
  train(model, frames[:15], times[:15])
  with torch.no_grad():
    ref = tv.transforms.functional.resize(og_frames.permute(0,3,1,2), (ts, ts)).permute(0,2,3,1)
    ref = (ref/255).mean(dim=-1, keepdim=True)
    test(ref, model, num_secs, og_num_frames)


if __name__ == "__main__": main()
