seq="7606-7665"
exp_name=${seq}-our

python train.py --config configs/kitti/loop_contract/${seq}.txt \
  --exp_name ${seq}-our \
  --ray_sampling_strategy epoch_like \
  --nerf_lambda_normal_ref 2e-4 \
  --nerf_lambda_normal_mono 1e-3 \
  --ngp_F 8 --ngp_F_ 8 --ngp_log2_T 19 --ngp_log2_T_ 21 \
  --lr 2e-3 \
  --depth_dp_proxy --nerf_lambda_depth_dp 1e-3 \
  --ngp_gridsize 128 \
  --sphere_scale_x_p 0.75 --sphere_scale_x_n 0.75 \
  --sphere_scale_y_p 0.75 --sphere_scale_y_n 0.75 \
  --sphere_scale_z_p 0.75 --sphere_scale_z_n 0.75 \
  --T_threshold 2e-4 \
  --lr_decay 100 \
  --strict_split \
  --normal_mono_skipsem 4 \
  --n_imgs 108