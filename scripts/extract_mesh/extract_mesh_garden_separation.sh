exp_name="gardenvase"
mesh_output_label="subpart_0"
ROOT_DIR="/home/hongchix/main/root/datasets/360_v2/garden"

python extract_mesh.py --config configs/release/garden.txt \
    --exp_name ${exp_name} \
    --root_dir ${ROOT_DIR} \
    --ngp_gridsize 128 \
    --extract_mesh_scale_x_min 0. \
    --extract_mesh_scale_x_max 1. \
    --extract_mesh_scale_y_min 0. \
    --extract_mesh_scale_y_max 1. \
    --extract_mesh_scale_z_min 0.4 \
    --extract_mesh_scale_z_max 0.7 \
    --extract_mesh_partition_x 3 \
    --extract_mesh_partition_y 3 \
    --extract_mesh_partition_z 3 \
    --ngp_F 8 \
    --ngp_F_ 8 \
    --ngp_log2_T 19 \
    --ngp_log2_T_ 21 \
    --ckpt_load ckpts/colmap/${exp_name}/last.ckpt \
    --num_classes 2 \
    --mesh_output_label ${mesh_output_label} \
    --center_extract_mesh \
    --mesh_post_process \
    --center_decimate \
    --center_decimate_target 5000000 \
    --filter_box 8 \
    --center_level 50 \
    --level 1 \
    --extract_mesh_partition_offset 0.01 \
    --mesh_postprocess_clean_f 1000 \
    --extract_mesh_post_contract \
    --pymesh_processing \
    --pymesh_processing_target_len 0.001 \
    --mesh_semantic_filter \
    --mesh_semantic_filter_semid 0 \
    --extract_mesh_generate_semantic_boundary \
    --extract_mesh_boundary_label ${mesh_output_label}


exp_name="gardenvase"
mesh_output_label="subpart_1"

python extract_mesh.py --config configs/release/garden.txt \
    --exp_name ${exp_name} \
    --root_dir ${ROOT_DIR} \
    --ngp_gridsize 128 \
    --extract_mesh_scale_x_min 0. \
    --extract_mesh_scale_x_max 1. \
    --extract_mesh_scale_y_min 0. \
    --extract_mesh_scale_y_max 1. \
    --extract_mesh_scale_z_min 0.4 \
    --extract_mesh_scale_z_max 0.7 \
    --extract_mesh_partition_x 3 \
    --extract_mesh_partition_y 3 \
    --extract_mesh_partition_z 3 \
    --ngp_F 8 \
    --ngp_F_ 8 \
    --ngp_log2_T 19 \
    --ngp_log2_T_ 21 \
    --ckpt_load ckpts/colmap/${exp_name}/last.ckpt \
    --num_classes 2 \
    --mesh_output_label ${mesh_output_label} \
    --center_extract_mesh \
    --mesh_post_process \
    --center_decimate \
    --center_decimate_target 5000000 \
    --filter_box 8 \
    --center_level 50 \
    --level 1 \
    --extract_mesh_partition_offset 0.01 \
    --mesh_postprocess_clean_f 1000 \
    --extract_mesh_post_contract \
    --pymesh_processing \
    --pymesh_processing_target_len 0.001 \
    --mesh_semantic_filter \
    --mesh_semantic_filter_semid 1 \
    --extract_mesh_generate_semantic_boundary \
    --extract_mesh_boundary_label ${mesh_output_label}
