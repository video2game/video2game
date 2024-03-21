workspace="example"
mesh_output_label="example"
exp_name="gardenvase"

python baking_pretrain.py --config configs/release/garden.txt \
    --ckpt_load ckpts/colmap/${exp_name}/last.ckpt  \
    --lr 0.005 --baking_iters 20000 \
    --workspace ${workspace} \
    --load_mesh_path results/colmap/${exp_name}/${exp_name}_${mesh_output_label}.ply \
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
#    \
#    --strict_split