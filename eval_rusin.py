import torch
import torch.optim as optim
import argparse
from src.utils import ( save_image )
import src.refl as refl
from tqdm import trange

def arguments():
  a = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  a.add_argument(
    "--refl-model", required=True, type=str,
  )
  return a.parse_args()


device="cpu"
if torch.cuda.is_available():
  device = torch.device("cuda:0")
  torch.cuda.set_device(device)

def main():
  args = arguments()
  with torch.no_grad():
    model = torch.load(args.refl_model)
    assert(hasattr(model, "refl")), "The provided model must have a refl"
    r = model.refl
    if isinstance(r, refl.LightAndRefl): r = r.refl
    # just check the first choice for now, can add a flag for it later
    if isinstance(r, refl.WeightedChoice): r = r.choices[1]
    # check again if it's another lit item
    if isinstance(r, refl.LightAndRefl): r = r.refl
    assert(isinstance(r, refl.Rusin)), f"must provide a rusin refl, got {type(r)}"

    degs = torch.stack(torch.meshgrid(
      # phi_d
      torch.linspace(0, 90, 256, device=device, dtype=torch.float),
      # theta_h
      torch.linspace(0, 90, 256, device=device, dtype=torch.float),
      # theta_d
      torch.arange(180, device=device, dtype=torch.float),
    ), dim=-1)
    rads = torch.deg2rad(degs)
    latent = torch.randn(*rads.shape[:-2], r.latent_size, device=device)
    for i, theta_h in enumerate(rads.split(1, dim=2)):
      theta_h = theta_h.squeeze(2)
      out = r.raw(theta_h, latent)
      save_image(f"outputs/rusin_eval_{i:03}.png", out)
  return

if __name__ == "__main__": main()