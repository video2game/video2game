exp_name="gardenvase"

# normal_mono activate normal prior
# depth_mono activate depth prior
# render_semantic activate semantic/instance MLP learning

python train.py --config configs/release/garden.txt \
  --exp_name ${exp_name} \
  --ray_sampling_strategy same_image \
  --normal_mono \
  --nerf_lambda_normal_ref 2e-4 \
  --nerf_lambda_normal_mono 1e-3 \
  --normal_analytic_mono \
  --lr 2e-3 \
  --depth_mono \
  --nerf_lambda_depth_mono 1e-2 \
  --render_semantic \
  --ngp_gridsize 128 \
  --scale_factor 1.0 \
  --ngp_F 8 --ngp_F_ 8 --ngp_log2_T 19 --ngp_log2_T_ 21 \
  --lr_decay 100
