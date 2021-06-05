clean:
	-@rm outputs/*.png
dnerf: clean
	python3 runner.py -d data/data/jumpingjacks/ --data-kind dnerf --render-size 32 \
	--crop --epochs 50_000 --mip cylinder --save models/djj_ae.pt --model ae \
  --near 0 --far 1 --load models/djj_ae.pt
sdf: clean
	python3 runner.py -d data/nerf_synthetic/lego/ --data-kind original --sdf \
	--render-size 128 --crop --epochs 50_000 --save models/lego.pt --crop-size 8 \
	--near 2 --far 6 --batch-size 24  --decay 5e-7 --model ae \
  --load models/lego.pt --n-sparsify-alpha 100 -lr 5e-3 # --mip cylinder --nerf-eikonal
original: clean
	python3 runner.py -d data/nerf_synthetic/lego/ --data-kind original \
	--render-size 128 --crop --epochs 50_000 --save models/lego_plain.pt \
	--near 2 --far 6 --batch-size 16 --mip cylinder --model ae -lr 1e-3 \
	--load models/lego_plain.pt --l1-loss #--omit-bg

single-video: clean
	python3 runner.py -d data/video/fencing.mp4 \
	--render-size 128 --crop --epochs 30_000 --save models/fencing.pt \
	--near 2 --far 10 --batch-size 5 --mip cylinder --model ae -lr 5e-3 \
	#--load models/lego_plain.pt --omit-bg

# [WIP]
pixel-single: clean
	python3 runner.py -d data/celeba_example.jpg --data-kind pixel-single --render-size 16 \
  --crop --crop-size 16 --save models/celeba_sp.pt --mip cylinder --model ae
