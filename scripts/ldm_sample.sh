export CUDA_VISIBLE_DEVICES=4
export PYTHONWARNINGS="ignore"

python ldm_sample.py \
--output_dir run/sample/ldm-celebahq-256 \
--batch_size 16 \
--total_samples 16 \
--ddim_steps 100 \
--pruned_model_ckpt run/finetuned/ldm-celebahq-256/pruned/unet_ema_pruned.pth \
--model_path run/finetuned/ldm-celebahq-256 \
--skip_type uniform \
