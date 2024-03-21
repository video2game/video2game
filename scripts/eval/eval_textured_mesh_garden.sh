workspace="example"
mesh_output_label="example"
exp_name="gardenvase"
output_label="uv4096"

python eval_textured_mesh.py --config configs/release/garden.txt \
    --exp_name ${exp_name} \
    --tcngp_F 8 \
    --tcngp_log2_T 21 \
    --workspace ${workspace} \
    --baking_specular_dim_mlp 32 \
    --baking_specular_dim 3 \
    --baking_output ${output_label} \
    --baking_per_level_scale 1024