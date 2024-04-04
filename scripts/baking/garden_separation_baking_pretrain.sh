workspace="example"
mesh_output_label_0="subpart_0"
mesh_output_label_1="subpart_1"
exp_name="gardenvase"
ROOT_DIR="/home/hongchix/main/root/datasets/360_v2/garden"

python baking_pretrain.py --config configs/release/garden.txt \
    --root_dir ${ROOT_DIR} \
    --lr 0.005 --baking_iters 20000 \
    --workspace ${workspace} \
    --load_mesh_paths \
     results/colmap/${exp_name}/${exp_name}_${mesh_output_label_0}.ply \
     results/colmap/${exp_name}/${exp_name}_${mesh_output_label_1}.ply \
    --ray_sampling_strategy single_image \
    --ssaa 1 \
    --exp_name ${exp_name} \
    --batch_size 1 \
    --ngp_gridsize 128 \
    --baking_specular_dim_mlp 32 \
    --baking_specular_dim 3 \
    --tcngp_F 8 \
    --tcngp_log2_T 21 \
    --baking_antialias \
    --baking_per_level_scale 1024 \
    --diffuse_step 1000