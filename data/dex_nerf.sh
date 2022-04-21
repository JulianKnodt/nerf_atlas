mkdir dex_nerf
cd dex_nerf

curl -L --output a.zip \
"https://github.com/BerkeleyAutomation/dex-nerf-datasets/releases/download/corl2021/dex_nerf_real_bottle.zip"
unzip a.zip
rm a.zip

curl -L --output a.zip \
"https://github.com/BerkeleyAutomation/dex-nerf-datasets/releases/download/corl2021/dex_nerf_real_flask.zip"
unzip a.zip
rm a.zip

curl -L --output a.zip \
"https://github.com/BerkeleyAutomation/dex-nerf-datasets/releases/download/corl2021/dex_nerf_real_safety_glasses.zip"
unzip a.zip
rm a.zip

curl -L --output a.zip \
"https://github.com/BerkeleyAutomation/dex-nerf-datasets/releases/download/corl2021/dex_nerf_real_wineglass.zip"
unzip a.zip
rm a.zip

curl -L --output a.zip \
"https://github.com/BerkeleyAutomation/dex-nerf-datasets/releases/download/corl2021/dex_nerf_real_dishwasher.zip"
unzip a.zip
rm a.zip

curl -L --output a.zip \
"https://github.com/BerkeleyAutomation/dex-nerf-datasets/releases/download/corl2021/dex_nerf_simulated_clutter_light_array.zip"
unzip a.zip
rm a.zip
