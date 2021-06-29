PHONY:

clean:
	-@rm outputs/*.png

volsdf: clean
	python3 -O runner.py -d data/nerf_synthetic/lego/ --data-kind original \
	--size 64 --crop --epochs 30_000 --crop-size 32 \
	--near 2 --far 6 --batch-size 4 --model volsdf --sdf-kind siren \
	-lr 5e-4 --no-sched --loss-window 750 --valid-freq 100 \
	--nosave --sdf-eikonal 0.1 --loss-fns l2 --save-freq 2500 --sigmoid-kind fat

food: clean
	python3 runner.py -d data/food/ --data-kind shiny --size 64 \
	--crop --epochs 50_000  --save models/food.pt --model ae --crop --batch-size 4 \
	--crop-size 24 --near 2 --far 6 -lr 5e-4 --no-sched --valid-freq 499 \

# note: l1 loss completely breaks dnerf
dnerf: clean
	python3 runner.py -d data/data/jumpingjacks/ --data-kind dnerf --size 32 \
	--crop --epochs 30_000  --save models/djj_ae.pt --model ae --crop --batch-size 4 \
	--crop-size 20 --near 2 --far 6 -lr 5e-4 --no-sched --valid-freq 499 \
  --dnerf-tf-smooth-weight 1e-4 --load models/djj_ae.pt

dnerf_gru: clean
	python3 runner.py -d data/data/bouncingballs/ --data-kind dnerf --size 64 \
	--crop --epochs 80_000  --save models/djj_gru_ae.pt --model ae --crop --batch-size 2 \
	--crop-size 24 --near 2 --far 6 -lr 5e-4 --no-sched --valid-freq 499 \
  --gru-flow --load models/djj_gru_ae.pt

# testing out dnerfae dataset on dnerf
dnerf_dyn: clean
	python3 runner.py -d data/data/jumpingjacks/ --data-kind dnerf --size 64 \
	--crop --epochs 80_000  --save models/djj_gamma.pt --model ae --crop --batch-size 1 \
	--crop-size 40 --near 2 --far 6 -lr 5e-4 --no-sched --valid-freq 499 \
	--serial-idxs --time-gamma --loss-window 750 #--load models/djj_gamma.pt

dnerfae: clean
	python3 runner.py -d data/data/jumpingjacks/ --data-kind dnerf --size 128 \
	--crop --epochs 40_000  --save models/djj_ae_gamma.pt --model ae --crop --batch-size 2 \
	--crop-size 32 --near 2 --far 6 -lr 2e-4 --no-sched --valid-freq 499 \
	--dnerfae --time-gamma --loss-window 750 --loss-fns rmse \
	--sigmoid-kind thin --load models/djj_ae_gamma.pt  #--omit-bg #--serial-idxs

sdf: clean
	python3 -O runner.py -d data/nerf_synthetic/lego/ --data-kind original \
	--size 128 --crop --epochs 5000 --save models/lego_sdf.pt --crop-size 64 \
	--near 2 --far 6 --batch-size 6 --model sdf --sdf-kind siren \
  -lr 5e-4 --no-sched --loss-window 750 --valid-freq 100 \
  --nosave --sdf-eikonal 0.1 --loss-fns l1 --save-freq 2500

original: clean
	python3 -O runner.py -d data/nerf_synthetic/lego/ --data-kind original \
	--size 64 --crop --epochs 80_000 --save models/lego.pt \
	--near 2 --far 6 --batch-size 5 --crop-size 26 --model plain -lr 5e-4 \
	--l1-loss --valid-freq 499 --no-sched #--load models/lego.pt #--omit-bg

unisurf: clean
	python3 -O runner.py -d data/nerf_synthetic/lego/ --data-kind original \
	--size 64 --crop --epochs 80_000 --save models/lego_us.pt \
	--near 2 --far 6 --batch-size 5 --crop-size 26 --model unisurf -lr 5e-4 \
	--l1-loss --valid-freq 499 --no-sched --load models/lego_us.pt #--omit-bg

test_original: clean
	python3 -O runner.py -d data/nerf_synthetic/lego/ --data-kind original \
	--size 64 --crop --epochs 0 --near 2 --far 6 --batch-size 5 \
  --crop-size 26 --load models/lego.pt

ae: clean
	python3 -O runner.py -d data/nerf_synthetic/lego/ --data-kind original \
	--size 64 --crop --epochs 80_000 --save models/lego_ae.pt \
	--near 2 --far 6 --batch-size 5 --crop-size 26 --model ae -lr 5e-4 \
	--valid-freq 499 --no-sched --l1-loss #--load models/lego_ae.pt #--omit-bg
test_ae: clean
	python3 -O runner.py -d data/nerf_synthetic/lego/ --data-kind original \
	--size 64 --crop --epochs 0 --near 2 --far 6 --batch-size 5 \
  --crop-size 26 --load models/lego_ae.pt

single-video: clean
	python3 runner.py -d data/video/fencing.mp4 \
	--size 128 --crop --epochs 30_000 --save models/fencing.pt \
	--near 2 --far 10 --batch-size 5 --mip cylinder --model ae -lr 1e-3 \
	#--load models/lego_plain.pt --omit-bg

og_upsample: clean
	python3 -O runner.py -d data/nerf_synthetic/lego/ --data-kind original \
	--render-size 16 --size 64 --epochs 80_000 --save models/lego_up.pt \
	--near 2 --far 6 --batch-size 4 --model plain -lr 5e-4 \
	--loss-fns l2 --valid-freq 499 --no-sched --neural-upsample --nosave


# [WIP]
pixel-single: clean
	python3 runner.py -d data/celeba_example.jpg --data-kind pixel-single --render-size 16 \
  --crop --crop-size 16 --save models/celeba_sp.pt --mip cylinder --model ae
