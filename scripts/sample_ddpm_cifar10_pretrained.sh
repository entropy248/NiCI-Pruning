export CUDA_VISIBLE_DEVICES=3
export PYTHONWARNINGS="ignore"
python ddpm_sample.py \
--output_dir run/sample/ddpm_cifar10_pretrained_quad \
--batch_size 128 \
--model_path pretrained/ddpm_ema_cifar10/ddpm_ema_cifar10 \