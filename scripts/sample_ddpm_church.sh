export CUDA_VISIBLE_DEVICES=4,6
export PYTHONWARNINGS="ignore"
python -m torch.distributed.launch --nproc_per_node=2 --master_port 22222 --use_env \
ddpm_sample.py \
--output_dir run/sample/ddpm_church_256_pruned_c2c_iter1000_step10 \
--batch_size 64 \
--total_samples 30000 \
--model_path run/finetuned/ddpm_church_256_pruned_c2c_iter1000_step10_iter200k \
--pruned_model_ckpt run/finetuned/ddpm_church_256_pruned_c2c_iter1000_step10_iter200k/pruned/unet_ema_pruned.pth \
--skip_type uniform \