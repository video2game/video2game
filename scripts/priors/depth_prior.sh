ROOT_DIR="/home/hongchix/main/root/datasets/360_v2/garden"
python priors/depth_prior.py \
    --source_dir ${ROOT_DIR}/images \
    --output_dir ${ROOT_DIR}/depth \
    --vis_dir ${ROOT_DIR}/depth_vis \
    --ckpt_dir ./pretrained_models

