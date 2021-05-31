clean:
	-@rm outputs/*.png
pixel-single: clean
	python3 runner.py -d data/celeba_example.jpg --data-kind pixel-single --render-size 16 \
  --crop --crop-size 16 --save models/celeba_sp.pt --mip cylinder --model ae
dnerf: clean
	python3 runner.py -d data/data/jumpingjacks/ --data-kind dnerf --render-size 32 \
	--crop --epochs 50_000 --mip cylinder --save models/djj_ae.pt --model ae \
  --near 0 --far 1 --load models/djj_ae.pt
sdf: clean
	python3 runner.py -d data/nerf_synthetic/lego/ --data-kind original --sdf \
	--render-size 32 --crop --epochs 30_000 --mip cylinder --save models/lego.pt \
	--near 2 --far 6 --batch-size 5 #--load models/lego.pt #--omit-bg
