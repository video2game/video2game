workspace="example"
mesh_output_label="example"
seq="7606-7665"
exp_name=${seq}

python baking_pretrain.py --config configs/release/kitti.txt \
    --ckpt_load ckpts/kitti/${exp_name}/last.ckpt  \
    --lr 0.005 --baking_iters 10000 \
    --workspace ${workspace} \
    --load_mesh_path results/kitti/${exp_name}/${exp_name}_${mesh_output_label}.ply \
    --ray_sampling_strategy single_image --skydome \
    --center_pos_trans ./pickles/center_trans.pkl \
    --ssaa 1 \
    --exp_name ${exp_name} \
    --batch_size 1 \
    --ngp_gridsize 128 \
    --sphere_scale_x_p 0.75 \
    --sphere_scale_x_n 0.75 \
    --sphere_scale_y_p 0.75 \
    --sphere_scale_y_n 0.75 \
    --sphere_scale_z_p 0.75 \
    --sphere_scale_z_n 0.75 \
    --baking_specular_dim_mlp 64 \
    --baking_specular_dim 3 \
    --tcngp_F 8 \
    --tcngp_log2_T 21 \
    --baking_antialias \
    --baking_per_level_scale 1024 \
    --diffuse_step 1000
#    \
#    --strict_split