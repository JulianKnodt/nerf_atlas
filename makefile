clean:
	-@rm outputs/*.png
pixel-single: clean
	python3 runner.py -d data/celeba_example.jpg --data-kind pixel-single --render-size 16 \
  --crop --crop-size 16 --save models/celeba_sp.pt --mip cylinder --model ae
dnerf: clean
	python3 runner.py -d data/data/jumpingjacks/ --data-kind dnerf --render-size 32 \
	--crop --epochs 60_000 --mip cylinder --save models/djj_ae.pt --model ae
