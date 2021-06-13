clean:
	-@rm outputs/*.png

food: clean
	python3 runner.py -d data/food/ --data-kind shiny --render-size 64 \
	--crop --epochs 50_000  --save models/food.pt --model ae --crop --batch-size 4 \
	--crop-size 24 --near 2 --far 6 -lr 5e-4 --no-sched --valid-freq 499 \
  #--load models/food.pt --decay 1e-6

# note: l1 loss completely breaks dnerf
dnerf: clean
	python3 -O runner.py -d data/data/jumpingjacks/ --data-kind dnerf --render-size 32 \
	--crop --epochs 50_000  --save models/djj_ae.pt --model ae --crop --batch-size 4 \
	--crop-size 24 --near 2 --far 6 -lr 5e-4 --no-sched --valid-freq 499 \
  --load models/djj_ae.pt --decay 1e-6

sdf: clean
	python3 runner.py -d data/nerf_synthetic/lego/ --data-kind original --sdf \
	--render-size 128 --crop --epochs 50_000 --save models/lego.pt --crop-size 8 \
	--near 2 --far 6 --batch-size 24  --decay 5e-7 --model ae \
  --load models/lego.pt --n-sparsify-alpha 100 -lr 2e-3 # --mip cylinder --nerf-eikonal

original: clean
	python3 -O runner.py -d data/nerf_synthetic/lego/ --data-kind original \
	--render-size 64 --crop --epochs 80_000 --save models/lego.pt \
	--near 2 --far 6 --batch-size 5 --crop-size 26 --model plain -lr 5e-4 \
	--l1-loss --valid-freq 499 --no-sched --load models/lego.pt #--omit-bg
test_original: clean
	python3 -O runner.py -d data/nerf_synthetic/lego/ --data-kind original \
	--render-size 64 --crop --epochs 0 --near 2 --far 6 --batch-size 5 \
  --crop-size 26 --load models/lego.pt

ae: clean
	python3 -O runner.py -d data/nerf_synthetic/lego/ --data-kind original \
	--render-size 64 --crop --epochs 80_000 --save models/lego_ae.pt \
	--near 2 --far 6 --batch-size 5 --crop-size 26 --model ae -lr 5e-4 \
	--valid-freq 499 --no-sched --l1-loss #--load models/lego_ae.pt #--omit-bg
test_ae: clean
	python3 -O runner.py -d data/nerf_synthetic/lego/ --data-kind original \
	--render-size 64 --crop --epochs 0 --near 2 --far 6 --batch-size 5 \
  --crop-size 26 --load models/lego_ae.pt

single-video: clean
	python3 runner.py -d data/video/fencing.mp4 \
	--render-size 128 --crop --epochs 30_000 --save models/fencing.pt \
	--near 2 --far 10 --batch-size 5 --mip cylinder --model ae -lr 1e-3 \
	#--load models/lego_plain.pt --omit-bg

# [WIP]
pixel-single: clean
	python3 runner.py -d data/celeba_example.jpg --data-kind pixel-single --render-size 16 \
  --crop --crop-size 16 --save models/celeba_sp.pt --mip cylinder --model ae
