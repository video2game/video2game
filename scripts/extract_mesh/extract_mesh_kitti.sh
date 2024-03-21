exp_name="7606-7665"
mesh_output_label="example"

python extract_mesh.py --config configs/release/kitti.txt \
    --exp_name ${exp_name} \
    --ngp_gridsize 128 \
    --sphere_scale_x_p 0.75 \
    --sphere_scale_x_n 0.75 \
    --sphere_scale_y_p 0.75 \
    --sphere_scale_y_n 0.75 \
    --sphere_scale_z_p 0.75 \
    --sphere_scale_z_n 0.75 \
    --extract_mesh_scale_x_min 0. \
    --extract_mesh_scale_x_max 1. \
    --extract_mesh_scale_y_min 0. \
    --extract_mesh_scale_y_max 1. \
    --extract_mesh_scale_z_min 0.4 \
    --extract_mesh_scale_z_max 0.7 \
    --extract_mesh_partition_x 3 \
    --extract_mesh_partition_y 3 \
    --extract_mesh_partition_z 3 \
    --ckpt_load ckpts/kitti/${exp_name}/last.ckpt \
    --mesh_output_label ${mesh_output_label} \
    --mesh_semantic_filter \
    --mesh_semantic_filter_semid 4 \
    --extract_mesh_sem_filter_negative \
    --center_extract_mesh \
    --mesh_post_process \
    --center_decimate \
    --center_decimate_target 5000000 \
    --filter_box 4 \
    --level 10 \
    --extract_mesh_partition_offset 0.01 \
    --mesh_postprocess_clean_f 1000 \
    --pymesh_processing \
    --pymesh_processing_target_len 0.006

#    --ngp_F 8 \
#    --ngp_F_ 8 \
#    --ngp_log2_T 19 \
#    --ngp_log2_T_ 21 \