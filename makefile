PHONY:

clean:
	-@rm outputs/*.png

volsdf: clean
	python3 runner.py -d data/nerf_synthetic/lego/ --data-kind original \
	--size 192 --crop --epochs 50_000 --crop-size 25 \
	--near 2 --far 6 --batch-size 4 --model volsdf --sdf-kind mlp \
	-lr 1e-3 --loss-window 750 --valid-freq 250 \
	--sdf-eikonal 0.1 --loss-fns l2 --save-freq 5000 --sigmoid-kind fat \
	--save models/lego_volsdf.pt --load models/lego_volsdf.pt

volsdf_with_normal: clean
	python3 runner.py -d data/nerf_synthetic/lego/ --data-kind original \
	--size 192 --crop --epochs 50_000 --crop-size 16 \
	--near 2 --far 6 --batch-size 4 --model volsdf --sdf-kind mlp \
	-lr 1e-3 --loss-window 750 --valid-freq 250 --nosave \
	--sdf-eikonal 0.1 --loss-fns l2 --save-freq 5000 --sigmoid-kind fat \
	--refl basic --normal-kind elaz --light-kind point

rusin: clean
	python3 runner.py -d data/nerf_synthetic/lego/ --data-kind original \
	--size 64 --crop --epochs 50_000 --crop-size 25 \
	--near 2 --far 6 --batch-size 3 --model volsdf --sdf-kind mlp \
	-lr 1e-3 --loss-window 750 --valid-freq 250 \
	--sdf-eikonal 0.1 --loss-fns l2 --save-freq 5000 --sigmoid-kind fat \
	--nosave --light-kind field --refl-kind rusin

nerfactor_ds := pinecone
nerf-sh: clean
	python3 runner.py -d data/nerfactor/${nerfactor_ds}/ \
	--data-kind original --size 128 --crop --epochs 25_000 --crop-size 25 \
	--near 2 --far 6 --batch-size 5 --model plain \
	-lr 1e-3 --loss-window 750 --valid-freq 250 \
	--loss-fns l2 --save-freq 5000 --sigmoid-kind fat \
	--refl sph-har --save models/${nerfactor_ds}-sh.pt \
  --load models/${nerfactor_ds}-sh.pt

# TODO fix this dataset, using it is a complete trash-fire
food: clean
	python3 runner.py -d data/food/ --data-kind shiny --size 64 \
	--crop --epochs 50_000  --save models/food.pt --model ae --crop --batch-size 4 \
	--crop-size 24 --near 2 --far 6 -lr 5e-4 --no-sched --valid-freq 499 \

# note: l1 loss completely breaks dnerf
dnerf: clean
	python3 runner.py -d data/dynamic/jumpingjacks/ --data-kind dnerf --size 32 \
	--crop --epochs 30_000  --save models/djj_ae.pt --model ae --crop --batch-size 3 \
	--crop-size 20 --near 2 --far 6 -lr 1e-3 --no-sched --valid-freq 499 \
	#--load models/djj_ae.pt

vsd_dataset := bouncingballs
volsdf_dnerf: clean
	python3 runner.py -d data/dynamic/$(vsd_dataset)/ --data-kind dnerf --size 32 \
	--crop --epochs 80_000  --save models/dvs_$(vsd_dataset).pt --model volsdf --crop \
  --batch-size 3 --crop-size 22 --near 2 --far 6 -lr 3e-4 --valid-freq 499 \
  --loss-fns l2 --sdf-kind mlp --loss-window 1000 --sdf-eikonal 0.1 \
  #--load models/dvs_$(vsd_dataset).pt

dnerf_gru: clean
	python3 runner.py -d data/dynamic/bouncingballs/ --data-kind dnerf --size 64 \
	--crop --epochs 80_000  --save models/djj_gru_ae.pt --model ae --crop --batch-size 2 \
	--crop-size 24 --near 2 --far 6 -lr 1e-3 --no-sched --valid-freq 499 \
  --gru-flow #--load models/djj_gru_ae.pt

# testing out dnerfae dataset on dnerf
dnerf_dyn: clean
	python3 runner.py -d data/dynamic/jumpingjacks/ --data-kind dnerf --size 64 \
	--crop --epochs 80_000  --save models/djj_gamma.pt --model ae --crop --batch-size 1 \
	--crop-size 40 --near 2 --far 6 -lr 5e-4 --no-sched --valid-freq 499 \
	--serial-idxs --time-gamma --loss-window 750 #--load models/djj_gamma.pt

dnerfae: clean
	python3 runner.py -d data/dynamic/jumpingjacks/ --data-kind dnerf --size 128 \
	--crop --epochs 40_000  --save models/djj_ae_gamma.pt --model ae --crop --batch-size 2 \
	--crop-size 32 --near 2 --far 6 -lr 2e-4 --no-sched --valid-freq 499 \
	--dnerfae --time-gamma --loss-window 750 --loss-fns rmse \
	--sigmoid-kind thin --load models/djj_ae_gamma.pt  #--omit-bg #--serial-idxs

sdf: clean
	python3 -O runner.py -d data/nerf_synthetic/lego/ --data-kind original \
	--size 128 --crop --epochs 5000 --save models/lego_sdf.pt --crop-size 128 \
	--near 2 --far 6 --batch-size 6 --model sdf --sdf-kind mlp \
  -lr 5e-4 --loss-window 750 --valid-freq 100 \
  --nosave --sdf-eikonal 0.1 --loss-fns l2 --save-freq 2500

scan_number := 97
dtu: clean
	python3 runner.py -d data/DTU/scan$(scan_number)/ --data-kind dtu \
	--size 192 --crop --epochs 50000 --save models/dtu$(scan_number).pt --save-freq 5000 \
	--near 0.3 --far 1.8 --batch-size 3 --crop-size 28 --model volsdf -lr 1e-3 \
	--loss-fns l2 --valid-freq 499 --sdf-kind mlp \
	--loss-window 1000 --sdf-eikonal 0.1 --sigmoid-kind fat --load models/dtu$(scan_number).pt

# -- Begin NeRV tests

# hotdogs | armadillo, fun datasets :)
nerv_dataset := hotdogs
nerv_point: clean
	python3 -O runner.py -d data/nerv_public_release/${nerv_dataset}/ \
	--data-kind nerv_point --model volsdf --sdf-kind mlp \
	--save models/nerv_${nerv_dataset}.pt \
	--size 200 --crop --crop-size 12 --epochs 10_000 --loss-window 1500 \
	--near 2 --far 6 --batch-size 4 -lr 2e-4 --refl-kind rusin \
	--sdf-eikonal 0.1 --light-kind dataset --seed -1 \
	--notraintest \
	--loss-fns l2 --valid-freq 499 --occ-kind learned \
  --load models/nerv_${nerv_dataset}.pt

nerv_point_sdf: clean
	python3 -O runner.py -d data/nerv_public_release/${nerv_dataset}/ \
	--data-kind nerv_point --model sdf --sdf-kind siren \
	--save models/nerv_sdf_${nerv_dataset}.pt \
	--size 200 --crop --crop-size 48 --epochs 50_000 --loss-window 250 \
	--near 2 --far 6 --batch-size 3 -lr 3e-4 --refl-kind rusin \
	--sdf-eikonal 0.1 --light-kind dataset \
	--loss-fns l2 --valid-freq 250 --save-freq 1000 --omit-bg --seed -1 \
	--occ-kind learned --notraintest \
	--integrator-kind direct --load models/nerv_sdf_${nerv_dataset}.pt

nerv_point_alternating: clean
	python3 -O runner.py -d data/nerv_public_release/${nerv_dataset}/ \
	--data-kind nerv_point --model volsdf --sdf-kind mlp \
	--save models/nerv_alt_${nerv_dataset}.pt \
	--size 200 --crop --crop-size 12 --epochs 15_000 --loss-window 500 \
	--near 2 --far 6 --batch-size 4 -lr 3e-4 --refl-kind rusin \
	--sdf-eikonal 0.1 --light-kind dataset \
	--loss-fns l2 --valid-freq 251 --save-freq 2500 --seed -1 \
	--occ-kind learned --volsdf-alternate --notraintest \
	#--load models/nerv_alt_${nerv_dataset}.pt

# -- End NeRV tests

original: clean
	python3 -O runner.py -d data/nerf_synthetic/lego/ --data-kind original \
	--size 64 --crop --epochs 80_000 --save models/lego.pt \
	--near 2 --far 6 --batch-size 4 --crop-size 26 --model plain -lr 1e-3 \
	--loss-fns l2 --valid-freq 499 --nosave #--load models/lego.pt #--omit-bg

test_original: clean
	python3 -O runner.py -d data/nerf_synthetic/lego/ --data-kind original \
	--size 64 --crop --epochs 0 --near 2 --far 6 --batch-size 5 \
  --crop-size 26 --load models/lego.pt

ae: clean
	python3 -O runner.py -d data/nerf_synthetic/lego/ --data-kind original \
	--size 64 --crop --epochs 80_000 --save models/lego_ae.pt \
	--near 2 --far 6 --batch-size 5 --crop-size 20 --model ae -lr 1e-3 \
	--valid-freq 499 --no-sched --loss-fns l2 #--load models/lego_ae.pt #--omit-bg

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
