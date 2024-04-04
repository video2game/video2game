ROOT_DIR="/home/hongchix/main/root/datasets/360_v2/garden"
python priors/normal_prior.py \
    --source_dir ${ROOT_DIR}/images \
    --output_dir ${ROOT_DIR}/normals \
    --ckpt_dir ./pretrained_models \
    --slice 3