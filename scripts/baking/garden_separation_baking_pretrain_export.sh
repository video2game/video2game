workspace="example"
mesh_output_label="example"
exp_name="gardenvase"
output_label="uv4096"

python baking_pretrain_export.py --config configs/release/garden.txt \
    --exp_name ${exp_name} \
    --tcngp_F 8 \
    --tcngp_log2_T 21 \
    --texture_size 4096 \
    --texture_dilate 32 \
    --texture_erode 3 \
    --xatlas_pad 1 \
    --workspace ${workspace} \
    --baking_specular_dim_mlp 32 \
    --baking_specular_dim 3 \
    --baking_output ${output_label} \
    --baking_per_level_scale 1024