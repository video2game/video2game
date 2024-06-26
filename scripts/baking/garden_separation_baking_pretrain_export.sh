ROOT_DIR="/home/hongchix/main/root/datasets/360_v2/garden"

workspace="example"
mesh_output_label_0="subpart_0"
exp_name="gardenvase"
output_label=${exp_name}_${mesh_output_label_0}

python baking_pretrain_export.py --config configs/release/garden.txt \
    --root_dir ${ROOT_DIR} \
    --exp_name ${exp_name} \
    --texture_export_substitude_mesh \
    --load_mesh_paths results/colmap/${exp_name}/${exp_name}_${mesh_output_label_0}.ply \
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

workspace="example"
mesh_output_label_1="subpart_1"
exp_name="gardenvase"
output_label=${exp_name}_${mesh_output_label_1}

python baking_pretrain_export.py --config configs/release/garden.txt \
    --root_dir ${ROOT_DIR} \
    --exp_name ${exp_name} \
    --texture_export_substitude_mesh \
    --load_mesh_paths results/colmap/${exp_name}/${exp_name}_${mesh_output_label_1}.ply \
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