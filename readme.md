# NeRF Atlas

A repository which contains NeRF and a bunch of extensions to NeRF.

---

## Usage

```sh
python3 runner.py -h
<All the flags>
```

## Dependencies

PyTorch, NumPy, tqdm, matplotlib.
Install them how you want.

---

## Extensions:

Currently, this repository contains a few extensions on "Plain" NeRF.

Model Level:

- TinyNeRF: One MLP for both density and output spectrum.
- NeRFAE (NeRF Auto Encoder): Our extension, which encodes every point in space as a vector in a
  latent material space, and derives density and RGB from this latent space. In theory this
  should allow for similar points to be learned more effectively.
- [D-NeRF](https://arxiv.org/abs/2011.13961) for dynamic scenes, using an MLP to encode a
  positional change.

Encoding:

- [Fourier Features](https://github.com/tancik/fourier-feature-networks) are applied to all
  low-dimensional inputs. This is always on.
- [MipNeRF](https://arxiv.org/abs/2103.13415) can be turned on with cylinder or conic volumes.

Training/Efficiency:

- DataParallel can be turned on and off.
- Train on cropped regions of the image for smaller GPUs.
- Neural Upsampling with latent spaces inspired by
  [GIRAFFE](https://arxiv.org/pdf/2011.12100.pdf). The results don't look great, but to be fair
  the paper also has some artifacts.

