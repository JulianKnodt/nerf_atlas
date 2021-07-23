import torch
import torch.nn as nn
from skimage import measure
import io

def marching_cubes(
  model,
  bounds=[(-1,-1,-1), (1,1,1)],
  # TODO add way to estimate bounds
  samples=2**22,
  # TODO add marching cube level here
):
  (x0, y0, z0), (x1, y1, z1) = bounds
  volume = (x1 - x0) * (y1 - y0) * (z1 - z0)
  step = (volume / samples) ** (1 / 3)
  X = torch.linspace(x0, x1, step)
  Y = torch.linspace(y0, y1, step)
  Z = torch.linspace(z0, z1, step)
  # TODO This is probably way to big to fit in memory.
  points = torch.stack(torch.meshgrid(X, Y, Z), dim=-1)
  # TODO process it with skimage.measure.marching_cube
  print(points.shape)
  exit()
  np_pts = points.numpy()
  verts, faces, normals, _ = measure.marching_cubes(np_pts, level=0)
  out = write_stl(verts, faces, normals)
  print(out)
  exit()
  raise NotImplementedError()

def write_stl(verts, faces, normals):
  out = io.StringIO()
  out.write("solid model\n")
  for f in faces:
    print(f)
    exit()
    v0, v1, v2 = verts[f]
    n = normals[f]
    facet = f"""
    facet normal {n[0]} {n[1]} {n[2]}
      outer loop
        vertex {v0[0]} {v0[1]} {v0[2]}
        vertex {v1[0]} {v1[1]} {v1[2]}
        vertex {v2[0]} {v2[1]} {v2[2]}
      endloop
    endfacet\n"""
  out.write("endsolid model")
  return out
