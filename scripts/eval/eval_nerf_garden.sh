exp_name="gardenvase"

# normal_mono activate normal prior
# depth_mono activate depth prior
# render_semantic activate semantic/instance MLP learning

python eval_nerf.py --config configs/release/garden.txt \
  --exp_name ${exp_name} \
  --ckpt_load ckpts/colmap/${exp_name}/last.ckpt \
  --ray_sampling_strategy same_image \
  --render_semantic \
  --ngp_gridsize 128 \
  --scale_factor 1.0 \
  --ngp_F 8 --ngp_F_ 8 --ngp_log2_T 19 --ngp_log2_T_ 21 \
  --lr_decay 100 \
  --eval_nerf_output eval_nerf
