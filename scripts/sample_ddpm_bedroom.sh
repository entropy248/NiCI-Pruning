export CUDA_VISIBLE_DEVICES=4
export PYTHONWARNINGS="ignore"
#python -m torch.distributed.launch --nproc_per_node=2 --master_port 22222 --use_env \
python ddpm_sample.py \
--output_dir run/sample/ddpm_bedroom_256_pruned_c2c_nkd \
--batch_size 64 \
--total_samples 30000 \
--model_path run/finetuned/ddpm_bedroom_256_pruned_c2c_nkd \
--pruned_model_ckpt run/finetuned/ddpm_bedroom_256_pruned_c2c_nkd/pruned/unet_ema_pruned-200000.pth \
--skip_type uniform \