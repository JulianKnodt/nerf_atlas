PHONv:

clean:
	-@rm outputs/*.png

original: clean
	python3 -O runner.py -d data/nerf_synthetic/lego/ --data-kind original \
	--size 64 --crop --epochs 80_000 --save models/lego.pt \
	--near 2 --far 6 --batch-size 4 --crop-size 26 --model plain -lr 1e-3 \
	--loss-fns l2 --valid-freq 499 --refl-kind view #--load models/lego.pt #--omit-bg

volsdf: clean
	python3 runner.py -d data/nerf_synthetic/lego/ --data-kind original \
	--size 128 --crop --epochs 30_000 --crop-size 16 \
	--near 2 --far 6 --batch-size 8 --model volsdf --sdf-kind mlp \
	-lr 5e-4 --loss-window 750 --valid-freq 250 \
	--sdf-eikonal 0.1 --loss-fns l2 --save-freq 2500 --sigmoid-kind thin \
	--depth-images --refl-kind pos --omit-bg \
  --color-space rgb xyz hsv \
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
	--data-kind original --size 128 --crop --epochs 0 --crop-size 25 \
	--near 2 --far 6 --batch-size 5 --model plain \
	-lr 1e-3 --loss-window 750 --valid-freq 250 \
	--loss-fns l2 --save-freq 5000 --sigmoid-kind leaky_relu \
	--refl-kind sph-har --save models/${nerfactor_ds}-sh.pt \
  --notest --depth-images --normals-from-depth \
  --load models/${nerfactor_ds}-sh.pt

nerfactor_volsdf: clean
	python3 runner.py -d data/nerfactor/${nerfactor_ds}/ \
	--data-kind original --size 128 --crop --epochs 25_000 --crop-size 18 \
	--near 2 --far 6 --batch-size 5 --model volsdf --sdf-kind siren \
	-lr 1e-3 --loss-window 750 --valid-freq 250 \
	--loss-fns l2 --color-spaces rgb --save-freq 2500 --sigmoid-kind leaky_relu \
	--refl-kind view --save models/${nerfactor_ds}-volsdf.pt --depth-images \
	--normals-from-depth \
  --notest \
  --sdf-eikonal 1e-2 \
  --load models/${nerfactor_ds}-volsdf.pt \
  #--smooth-normals 1e-2 --smooth-eps-rng \

nerfactor_volsdf_direct: clean
	python3 runner.py -d data/nerfactor/${nerfactor_ds}/ \
	--data-kind original --size 128 --crop --crop-size 14 --epochs 50_000 \
	--near 2 --far 6 --batch-size 4 --model volsdf --sdf-kind siren \
	-lr 1e-3 --loss-window 750 --valid-freq 500 \
	--loss-fns l2 --save-freq 2500 --occ-kind all-learned \
	--refl-kind rusin --save models/${nerfactor_ds}-volsdfd.pt --light-kind field \
  --color-spaces rgb --depth-images --normals-from-depth \
  --sdf-eikonal 1e-2 --smooth-normals 1e-2 --smooth-eps-rng \
  --sigmoid-kind normal --notest \
  --load models/${nerfactor_ds}-volsdfd.pt

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
nerv_dataset := armadillo
nerv_point: clean
	python3 runner.py -d data/nerv_public_release/${nerv_dataset}/ \
	--data-kind nerv_point --model volsdf --sdf-kind mlp \
	--save models/nerv_${nerv_dataset}.pt \
	--size 32 --crop --crop-size 14 --epochs 80_000  --loss-window 1500 \
	--near 2 --far 6 --batch-size 4 -lr 8e-4 --refl-kind rusin \
	--sdf-eikonal 1 --light-kind dataset --seed -1 \
	--loss-fns l2 --valid-freq 500 --save-freq 2500 --occ-kind all-learned \
  --color-spaces rgb --depth-images --depth-query-normal \
  --sigmoid-kind upshifted_softplus --skip-loss 100 \
  --notraintest --draw-colormap \
  --normals-from-depth --msssim-loss --depth-query-normal --display-smoothness \
  #--load models/nerv_${nerv_dataset}.pt \
  # --smooth-normals 1e-5 --smooth-eps 1e-3 --smooth-surface 1e-5 \

nerv_point_diffuse: clean
	python3 runner.py -d data/nerv_public_release/${nerv_dataset}/ \
	--data-kind nerv_point --model volsdf --sdf-kind mlp \
	--save models/nerv_diffuse_${nerv_dataset}.pt \
	--size 200 --crop --crop-size 12 --epochs 0  --loss-window 1500 \
	--near 2 --far 6 --batch-size 4 -lr 8e-4 --refl-kind diffuse \
	--sdf-eikonal 1 --light-kind dataset --seed -1 \
	--loss-fns l2 rmse --valid-freq 500 --save-freq 2500 --occ-kind all-learned \
  --color-spaces rgb hsv xyz --depth-images --depth-query-normal \
  --sigmoid-kind upshifted_softplus --skip-loss 100 \
  --load models/nerv_diffuse_${nerv_dataset}.pt \
  --smooth-normals 1e-5 --smooth-eps 1e-3 --notraintest \
  --normals-from-depth --msssim-loss --depth-query-normal --display-smoothness \
  --smooth-surface 1e-5

nerv_point_sdf: clean
	python3 runner.py -d data/nerv_public_release/${nerv_dataset}/ \
	--data-kind nerv_point --model sdf --sdf-kind mlp \
	--save models/nerv_sdf_${nerv_dataset}.pt \
	--size 200 --crop --crop-size 32 --epochs 20_000 --loss-window 500 \
	--near 2 --far 6 --batch-size 3 -lr 5e-4 --refl-kind multi_rusin \
	--sdf-eikonal 0.1 --light-kind dataset \
	--loss-fns l2 l1 rmse --valid-freq 250 --save-freq 1000 --seed -1 \
	--occ-kind learned --sdf-isect-kind bisect \
  --integrator-kind direct --color-spaces rgb hsv xyz \
	--load models/nerv_sdf_${nerv_dataset}.pt

nerv_point_alternating: clean
	python3 runner.py -d data/nerv_public_release/${nerv_dataset}/ \
	--data-kind nerv_point --model volsdf --sdf-kind mlp \
	--save models/nerv_alt_${nerv_dataset}.pt \
	--size 200 --crop --crop-size 12 --epochs 50_000 --loss-window 500 \
	--near 2 --far 6 --batch-size 4 -lr 5e-4 --refl-kind rusin \
	--sdf-eikonal 0.1 --light-kind dataset \
	--loss-fns l1 l2 --valid-freq 250 --save-freq 2500 --seed -1 \
	--occ-kind all-learned --volsdf-alternate --notraintest \
	--sdf-isect-kind bisect --color-spaces rgb hsv xyz \
	--load models/nerv_alt_${nerv_dataset}.pt

# experimenting with path tracing and nerv
nerv_point_path: clean
	python3 runner.py -d data/nerv_public_release/${nerv_dataset}/ \
	--data-kind nerv_point --model volsdf --sdf-kind mlp \
	--save models/nerv_path_${nerv_dataset}.pt \
	--size 32 --crop --crop-size 6 --epochs 20_000 --loss-window 500 \
	--near 2 --far 6 --batch-size 3 -lr 5e-4 --refl-kind rusin \
	--sdf-eikonal 0.1 --light-kind dataset --seed -1 \
	--loss-fns l2 --valid-freq 500 --occ-kind all-learned \
  --color-spaces rgb --save-freq 1000 \
  --integrator-kind path --depth-images --notraintest --skip-loss 500 \
  --smooth-eps 2e-3 --smooth-occ 1e-1 --sigmoid-kind upshifted_softplus \
  --normals-from-depth --msssim-loss --depth-query-normal --display-smoothness \
  --smooth-normals 1e-1 --normals-from-depth \
  --load models/nerv_path_${nerv_dataset}.pt #--path-learn-missing

nerv_point_subrefl: clean
	python3 runner.py -d data/nerv_public_release/${nerv_dataset}/ \
	--data-kind nerv_point --model volsdf --sdf-kind mlp \
	--save models/nerv_weighted_${nerv_dataset}.pt \
	--size 200 --crop --crop-size 12 --epochs 30_000 --loss-window 1500 \
	--near 2 --far 6 --batch-size 4 -lr 3e-4 --refl-kind weighted \
	--sdf-eikonal 0.1 --light-kind dataset --seed -1 \
	--loss-fns l2 rmse --valid-freq 500 --occ-kind all-learned \
  --color-spaces rgb hsv xyz \
  --notraintest --omit-bg \
  --load models/nerv_weighted_${nerv_dataset}.pt

nerv_point_fourier: clean
	python3 runner.py -d data/nerv_public_release/${nerv_dataset}/ \
	--data-kind nerv_point --model volsdf --sdf-kind mlp \
	--save models/nerv_fourier_${nerv_dataset}.pt \
	--size 200 --crop --crop-size 14 --epochs 50_000 --loss-window 1500 \
	--near 2 --far 6 --batch-size 4 -lr 8e-4 --refl-kind fourier \
	--sdf-eikonal 0.1 --light-kind dataset --seed -1 \
	--loss-fns l2 rmse --valid-freq 500 --occ-kind all-learned \
  --color-spaces rgb hsv xyz \
  --notraintest --depth-images \
  --smooth-normals 1e-3 --smooth-eps 1e-3 --notraintest \
  --normals-from-depth --msssim-loss --depth-query-normal --display-smoothness \
  --smooth-surface 1e-3 --sdf-isect-kind bisect \
  --draw-colormap \
  --load models/nerv_fourier_${nerv_dataset}.pt

# -- End NeRV tests

test_original: clean
	python3 -O runner.py -d data/nerf_synthetic/lego/ --data-kind original \
	--size 64 --crop --epochs 0 --near 2 --far 6 --batch-size 5 \
  --crop-size 26 --load models/lego.pt

ae: clean
	python3 -O runner.py -d data/nerf_synthetic/lego/ --data-kind original \
	--size 64 --crop --epochs 80_000 --save models/lego_ae.pt \
	--near 2 --far 6 --batch-size 5 --crop-size 20 --model ae -lr 1e-3 \
	--valid-freq 499 --no-sched --loss-fns l2 #--load models/lego_ae.pt #--omit-bg

# [WIP]
single_video: clean
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
pixel_single: clean
	python3 runner.py -d data/celeba_example.jpg --data-kind pixel-single --render-size 16 \
  --crop --crop-size 16 --save models/celeba_sp.pt --mip cylinder --model ae


# scripts

gan_sdf:
	python3 gan_sdf.py --epochs 15_000 --num-test-samples 256 --sample-size 1000 \
  --eikonal-weight 1 --nosave --noglobal --render-size 256 --crop-size 128 --load

volsdf_gan:
	python3 gan_sdf.py --epochs 25_000 --num-test-samples 256 --sample-size 900 \
  --eikonal-weight 0 --target volsdf --volsdf-model models/lego_volsdf.pt \
	--refl-kind pos --bounds 2 --noglobal --render-size 256 --crop-size 128 --G-model mlp \
  --load --G-rep 3

volsdf_gan_no_refl:
	python3 gan_sdf.py --epochs 25_000 --num-test-samples 256 --sample-size 1024 \
  --eikonal-weight 1e-2 --target volsdf --volsdf-model models/lego_volsdf.pt \
	--bounds 1.5 --noglobal --render-size 128 --G-model mlp

# evaluates the reflectance of a rusin model
eval_rusin:
	python3 eval_rusin.py --refl-model models/nerv_hotdogs.pt

fieldgan: clean
	python3 fieldgan.py --image data/mondrian.jpg --epochs 2500
	#python3 fieldgan.py --image data/food/images/IMG_1268.png --epochs 2500

rnn_nerf: clean
	python3 -O rnn_runner.py -d data/nerf_synthetic/lego/ --data-kind original \
	--size 256 --crop --epochs 7_500 --save models/rnn_lego.pt \
	--near 2 --far 6 --batch-size 4 --crop-size 12 -lr 1e-3 \
  --save-freq 2500 \
	--loss-fns l2 --valid-freq 499 --load models/rnn_lego.pt
