export CUDA_VISIBLE_DEVICES=6
export PYTHONWARNINGS="ignore"

python ddpm_sample.py \
--output_dir run/sample/ddpm_cifar10_pretrained_t \
--batch_size 1024 \
--total_samples 50000 \
--pruned_model_ckpt run/finetuned/6.75ddpm_cifar10_pruned_0.5_step_c2c_1epo_pstep/pruned/unet_ema_pruned.pth \
--model_path run/finetuned/6.75ddpm_cifar10_pruned_0.5_step_c2c_1epo_pstep \
--skip_type uniform \

