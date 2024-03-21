workspace="example"
mesh_output_label="example"
seq="7606-7665"
exp_name=${seq}
output_label="uv4096"

python baking_pretrain_export.py --config configs/release/kitti.txt \
    --exp_name ${exp_name} \
    --tcngp_F 8 \
    --tcngp_log2_T 21 \
    --texture_size 4096 \
    --texture_dilate 32 \
    --texture_erode 3 \
    --xatlas_pad 1 \
    --workspace ${workspace} \
    --sphere_scale_x_p 0.75 \
    --sphere_scale_x_n 0.75 \
    --sphere_scale_y_p 0.75 \
    --sphere_scale_y_n 0.75 \
    --sphere_scale_z_p 0.75 \
    --sphere_scale_z_n 0.75 \
    --baking_specular_dim_mlp 64 \
    --baking_specular_dim 3 \
    --baking_output ${output_label} \
    --baking_per_level_scale 1024 \
    --skydome --center_pos_trans ./pickles/center_trans.pkl