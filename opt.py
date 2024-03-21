import argparse
import configargparse
import numpy as np
def get_opts():
    # parser = argparse.ArgumentParser()
    parser = configargparse.ArgumentParser()
    parser.add_argument('--config', is_config_file=True, help='config file path')
    # common args for all datasets
    parser.add_argument('--root_dir', type=str, required=True,
                        help='root directory of dataset')
    parser.add_argument('--dataset_name', type=str, default='nerf',
                        choices=['kitti', 'colmap', 'vr_nerf', 'scannetpp', 'scannetpp_dslr', 'nerfstudio'],
                        help='which dataset to train/test')
    parser.add_argument('--split', type=str, default='train',
                        choices=['train', 'trainval', 'test'],
                        help='use which split to train')
    parser.add_argument('--downsample', type=float, default=1.0,
                        help='downsample factor (<=1.0) for the images')
    parser.add_argument('--anti_aliasing_factor', type=float, default=1.0,
                        help='Render larger images then downsample')

    # model parameters
    parser.add_argument('--scale', type=float, default=0.5,
                        help='scene scale (whole scene must lie in [-scale, scale]^3')
    parser.add_argument('--embed_a', action='store_true', default=False,
                        help='whether to use appearance embeddings')
    parser.add_argument('--embed_a_len', type=int, default=4,
                        help='the length of the appearance embeddings')
    parser.add_argument('--num_classes', type=int, default=7,
                        help='total number of semantic classes')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--cascade', type=int, default=None)
    
    # contraction
    parser.add_argument('--contraction_type', type=str, default='AABB')
    parser.add_argument('--sphere_scale_x_p', type=float, default=1)
    parser.add_argument('--sphere_scale_x_n', type=float, default=1)
    parser.add_argument('--sphere_scale_y_p', type=float, default=1)
    parser.add_argument('--sphere_scale_y_n', type=float, default=1)
    parser.add_argument('--sphere_scale_z_p', type=float, default=1)
    parser.add_argument('--sphere_scale_z_n', type=float, default=1)
    
    # nerf NGP parameters
    parser.add_argument('--ngp_L', type=int, default=16)
    parser.add_argument('--ngp_F', type=int, default=2)
    parser.add_argument('--ngp_log2_T', type=int, default=17)
    parser.add_argument('--ngp_L_', type=int, default=16)
    parser.add_argument('--ngp_F_', type=int, default=2)
    parser.add_argument('--ngp_log2_T_', type=int, default=19)
    parser.add_argument('--ngp_base_res', type=int, default=16)
    parser.add_argument('--ngp_base_growth', type=int, default=2048)
    parser.add_argument('--ngp_growth_factor', type=float, default=None)
    parser.add_argument('--ngp_gridsize', type=int, default=128)
    
    # nerf MLP parameters
    parser.add_argument('--mlp_sigma_neurons', type=int, default=128)
    parser.add_argument('--mlp_rgb_neurons', type=int, default=128)
    parser.add_argument('--mlp_norm_neurons', type=int, default=32)
    parser.add_argument('--mlp_sem_neurons', type=int, default=32)
    
    
    # for kitti 360 dataset
    parser.add_argument('--kitti_seq', type=int, default=0, 
                        help='scene sequence index')
    parser.add_argument('--kitti_start', type=int, default=1538,
                        help='starting frame index')
    parser.add_argument('--kitti_end', type=int, default=1601,
                        help='ending frame index')
    parser.add_argument('--kitti_test_id', type=int, nargs='+', default=[],
                        help='frames for testing')
    parser.add_argument('--kitti_dual_seq', action='store_true', default=False)
    parser.add_argument('--kitti_dual_start', type=int, default=1538,
                        help='starting frame index')
    parser.add_argument('--kitti_dual_end', type=int, default=1601,
                        help='ending frame index')
    parser.add_argument('--n_imgs', type=int, default=None)
    parser.add_argument('--sky_sem', type=int, default=4)
    
    parser.add_argument('--test_kitti_seq', type=int, default=0)
    parser.add_argument('--test_kitti_start', type=int, default=1538,
                        help='starting frame index')
    parser.add_argument('--test_kitti_end', type=int, default=1601,
                        help='ending frame index')
    parser.add_argument('--test_output_dir', type=str, default=None)
    
    # for kitti loop
    parser.add_argument('--center_pos_trans', type=str, default='')
    parser.add_argument('--center',  action='store_true')
    parser.add_argument('--keep', type=str, default='')
    parser.add_argument('--kitti_reloc', action='store_true')

    # camera pose normalization
    parser.add_argument('--scale_factor', type=float, default=1)
    

    # training options
    parser.add_argument('--batch_size', type=int, default=8192,
                        help='number of rays in a batch')
    parser.add_argument('--ray_sampling_strategy', type=str, default='all_images',
                        choices=['all_images', 'same_image', 'single_image', 'epoch_like'],
                        help='''
                        ray_sampling_strategy:
                            1. NeRF training
                            - all_images: randomly sample from all images
                            - same_image: randomly sample from only one image
                            - epoch_like: randomly sample but iterate over all pixels in one epoch
                            2. NeRF baking
                        ''')
    parser.add_argument('--num_epochs', type=int, default=30,
                        help='number of training epochs')
    parser.add_argument('--num_gpus', type=int, default=1,
                        help='number of gpus')
    parser.add_argument('--lr', type=float, default=1e-2,
                        help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=0,
                        help='learning rate')
    parser.add_argument('--max_samples', type=int, default=None)
    parser.add_argument('--T_threshold', type=float, default=1e-4)
    parser.add_argument('--lr_decay', type=float, default=30)
    
    # apply mask
    parser.add_argument('--use_mask',  action='store_true')
    
    # loss choices
    parser.add_argument('--depth_mono', action='store_true', default=False,
                        help='use 2D depth prediction as supervision')
    parser.add_argument('--normal_mono', action='store_true', default=False,
                        help='use 2D normal prediction as supervision')
    parser.add_argument('--normal_ref', action='store_true', default=False,
                        help='use density gradient as normal supervision (Ref-NeRF)')
    parser.add_argument('--remove_sparsity',  action='store_true')
    parser.add_argument('--kitti_lidar_proxy', action='store_true')
    parser.add_argument('--depth_dp_proxy',  action='store_true')
    parser.add_argument('--nerf_normal_sm',  action='store_true')
    parser.add_argument('--normal_analytic_mono',  action='store_true')
    parser.add_argument('--nerf_xyz_opacity',  action='store_true')
    parser.add_argument('--normal_mono_skipsem', type=int, nargs='+', default=[])
    
    # loss weights
    parser.add_argument('--nerf_lambda_opa', type=float, default=2e-4)
    parser.add_argument('--nerf_opacity_solid',  action='store_true')
    parser.add_argument('--nerf_lambda_xyz_opa', type=float, default=2e-4)
    
    parser.add_argument('--nerf_lambda_distortion', type=float, default=3e-4)
    
    parser.add_argument('--nerf_lambda_depth_mono', type=float, default=1e-3)
    parser.add_argument('--nerf_lambda_depth_dp', type=float, default=1e-3)
    parser.add_argument('--nerf_lambda_lidar_proxy', type=float, default=1e-3)
    
    parser.add_argument('--nerf_lambda_normal_mono', type=float, default=1e-3)
    parser.add_argument('--nerf_lambda_normal_ref', type=float, default=1e-3)
    
    parser.add_argument('--nerf_lambda_sky', type=float, default=1e-1)
    parser.add_argument('--nerf_lambda_semantic', type=float, default=4e-2)
    
    parser.add_argument('--nerf_lambda_sparsity', type=float, default=1e-4)
    
    parser.add_argument('--nerf_lambda_normal_sm', type=float, default=2e-4)
    parser.add_argument('--nerf_normal_sm_eps', type=float, default=1e-3)
    
    
    # experimental training options
    parser.add_argument('--optimize_ext', action='store_true', default=False,
                        help='whether to optimize extrinsics (experimental)')
    parser.add_argument('--random_bg', action='store_true', default=False,
                        help='''whether to train with random bg color (real dataset only)
                        to avoid objects with black color to be predicted as transparent
                        ''')

    # validation options
    parser.add_argument('--eval_lpips', action='store_true', default=False,
                        help='evaluate lpips metric (consumes more VRAM)')
    parser.add_argument('--val_only', action='store_true', default=False,
                        help='run only validation (need to provide ckpt_path)')
    parser.add_argument('--no_save_test', action='store_true', default=False,
                        help='whether to save test image and video')
    parser.add_argument('--strict_split',  action='store_true')
    parser.add_argument('--validate_all',  action='store_true')
    parser.add_argument('--train_log_iterations',  type=int, default=5000)

    # misc
    parser.add_argument('--exp_name', type=str, default='exp',
                        help='experiment name')
    parser.add_argument('--ckpt_load', type=str, default=None,
                        help='pretrained checkpoint to load (including optimizers, etc)')
    parser.add_argument('--ckpt_save', type=str, default='checkpoint.ckpt',
                        help='pretrained checkpoint to save (including optimizers, etc)')
    parser.add_argument("--workspace", type=str, default=None)

    # render
    parser.add_argument('--render_rgb', action='store_true', default=False,
                        help='render rgb series')
    parser.add_argument('--render_depth', action='store_true', default=False,
                        help='render depth series')
    parser.add_argument('--render_normal', action='store_true', default=False,
                        help='render normal series')
    parser.add_argument('--render_semantic', action='store_true', default=False,
                        help='render semantic segmentation series')
    parser.add_argument('--chunk_size', type=int, default=131072, 
                        help='Divide image into chunks for rendering')
    
    
    # camera pose normalization file
    parser.add_argument('--pos_trans', type=str, default=None)
        
    # nerf test output    
    parser.add_argument('--eval_nerf_output', type=str, default='eval_nerf')
    # textured mesh eval output
    parser.add_argument('--eval_textured_mesh_output', type=str, default='eval_textured_mesh')

    
    # mesh extraction parameters
    
    # mesh extraction: marching cube density level
    parser.add_argument('--level', type=float, default=10)
    parser.add_argument('--center_level', type=float, default=-1)
    
    # mesh extraction: resolution and range
    parser.add_argument('--center_extract_mesh',  action='store_true')
    parser.add_argument('--extract_mesh_cropN', type=int, default=256)
    
    parser.add_argument('--mesh_post_process',  action='store_true')
    parser.add_argument('--center_decimate',  action='store_true')
    parser.add_argument('--center_decimate_target', type=int, default=80000)
    parser.add_argument('--extract_mesh_scale_x_min', type=float, default=0.1)
    parser.add_argument('--extract_mesh_scale_x_max', type=float, default=0.9)
    parser.add_argument('--extract_mesh_scale_y_min', type=float, default=0.1)
    parser.add_argument('--extract_mesh_scale_y_max', type=float, default=0.9)
    parser.add_argument('--extract_mesh_scale_z_min', type=float, default=0.4)
    parser.add_argument('--extract_mesh_scale_z_max', type=float, default=0.75)

    parser.add_argument('--center_create_ring',  action='store_true')
    parser.add_argument('--extract_mesh_ring_scale_x_min', type=float, default=0.15)
    parser.add_argument('--extract_mesh_ring_scale_x_max', type=float, default=0.85)
    parser.add_argument('--extract_mesh_ring_scale_y_min', type=float, default=0.15)
    parser.add_argument('--extract_mesh_ring_scale_y_max', type=float, default=0.85)
    parser.add_argument('--extract_mesh_ring_scale_z_min', type=float, default=0.4)
    parser.add_argument('--extract_mesh_ring_scale_z_max', type=float, default=0.75)
    
    # mesh extraction resolution: partitioning
    parser.add_argument('--extract_mesh_partition_x', type=int, default=3)
    parser.add_argument('--extract_mesh_partition_y', type=int, default=3)
    parser.add_argument('--extract_mesh_partition_z', type=int, default=3)
    parser.add_argument('--extract_mesh_partition_offset', type=float, default=0.004)
    
    # mesh extraction: post-processing
    parser.add_argument('--filter_box', type=float, default=-1)
    parser.add_argument('--pymesh_processing',  action='store_true')
    parser.add_argument('--extract_mesh_post_contract',  action='store_true')
    parser.add_argument('--pymesh_processing_target_len', type=float, default=0.003)

    parser.add_argument('--mesh_postprocess_clean_f', type=int, default=5)
    parser.add_argument('--mesh_postprocess_v_pct', type=int, default=0)
    parser.add_argument('--mesh_postprocess_dilation', type=int, default=10)
    parser.add_argument('--extract_mesh_partition_decimate', type=float, default=1.0)

    # mesh extraction: output label
    parser.add_argument('--mesh_output_label', type=str, default="")

    # mesh extraction: mesh seperation parameters
    parser.add_argument('--extract_mesh_sem_filter_negative',  action='store_true') # sem_filter_negative
    parser.add_argument('--extract_mesh_bbox_src', type=str, default=None)
    parser.add_argument('--extract_mesh_generate_semantic_boundary',  action='store_true')
    parser.add_argument('--extract_mesh_boundary_label', type=str, default='') # boundary_output_label
    parser.add_argument('--extract_mesh_kitti_bbox_vehicle', type=str, default=None)
    parser.add_argument('--mesh_semantic_filter',  action='store_true')
    parser.add_argument('--mesh_semantic_filter_semid', type=int, default=0)
    
    
    # baking: pre-training parameters
    parser.add_argument('--baking_iters', type=int, default=10000)
    parser.add_argument('--load_mesh_path', type=str, default=None)
    parser.add_argument('--load_mesh_paths', type=str, default=[], nargs='+')
    parser.add_argument('--color_space', type=str, default='srgb')
    parser.add_argument('--diffuse_step', type=int, default=-1)
    parser.add_argument('--pos_gradient_boost', type=float, default=1, help="nvdiffrast option")
    parser.add_argument('--baking_rgb_loss', type=str, default="sml1") # n2m_rgb_loss
    parser.add_argument('--baking_per_level_scale', type=float, default=None) # n2m_per_level_scale
    parser.add_argument('--baking_antialias',  action='store_true') # rpermute
    
    # baking pre-training model parameters
    parser.add_argument('--baking_specular_dim', type=int, default=3) # n2m_specular_dim
    parser.add_argument('--baking_specular_dim_mlp', type=int, default=64) # specular_dim_mlp
    parser.add_argument('--tcngp_L', type=int, default=16)
    parser.add_argument('--tcngp_F', type=int, default=2)
    parser.add_argument('--tcngp_log2_T', type=int, default=19)
    parser.add_argument('--tcngp_dim_hidden', type=int, default=128)
    parser.add_argument('--tcngp_num_layers', type=int, default=3)
    parser.add_argument('--tcngp_scale', type=float, default=0.5,
                        help='scene scale (whole scene must lie in [-scale, scale]^3')
    parser.add_argument('--tcngp_contraction_type', type=str, default=None)
    
    # baking pre-training misc
    parser.add_argument('--skydome', action='store_true')

    # baking: pre-training export textured mesh parameters
    parser.add_argument('--ssaa', type=float, default=2)
    parser.add_argument('--texture_size', type=int, default=8192, help="exported texture resolution")
    parser.add_argument('--baking_output', type=str, default=None) # n2m_test_output
    parser.add_argument('--xatlas_pad', type=int, default=0)

    parser.add_argument('--texture_dilate', type=int, default=3) # n2m_test_dilate
    parser.add_argument('--texture_compress', type=int, default=3) # n2m_test_compress
    parser.add_argument('--texture_erode', type=int, default=2) # n2m_test_erode
    parser.add_argument('--texture_export_cas', type=int, default=None) # n2m_test_export_cas
    parser.add_argument('--texture_boundary_input_label', type=str, default=None) # n2m_test_boundary_input
    parser.add_argument('--texture_export_substitude_mesh', type=str, default=None) # n2m_substitude_mesh

    # baking: for texture finetuning
    parser.add_argument('--baking_noft_sky',  action='store_true') # n2m_noft_sky
    parser.add_argument('--baking_ft_mlp',  action='store_true') # n2m_ft_mlp
    parser.add_argument('--baking_ft_mlp_lr', type=float, default=1e-3) # n2m_ft_mlp_lr
    parser.add_argument('--baking_ft_mlp_wd', type=float, default=0) # n2m_ft_mlp_wd

    return parser.parse_args()
