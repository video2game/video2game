mkdir -p pretrained_models
cd pretrained_models
wget -O omnidata_dpt_depth_v2.ckpt 'https://zenodo.org/records/10447888/files/omnidata_dpt_depth_v2.ckpt?download=1'
wget -O omnidata_dpt_normal_v2.ckpt 'https://zenodo.org/records/10447888/files/omnidata_dpt_normal_v2.ckpt?download=1'
cd ..