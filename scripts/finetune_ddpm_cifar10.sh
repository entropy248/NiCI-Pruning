export CUDA_VISIBLE_DEVICES=4
export PYTHONWARNINGS="ignore"

python ddpm_train.py \
  --dataset="cifar10" \
  --model_path="run/pruned/ddpm_cifar10_pruned_step_c2c" \
  --pruned_model_ckpt="run/pruned/ddpm_cifar10_pruned_step_c2c/pruned/unet_pruned.pth" \
  --resolution=32 \
  --output_dir="run/finetuned/ddpm_cifar10_pruned_step_c2c" \
  --train_batch_size=128 \
  --num_iters=100000 \
  --gradient_accumulation_steps=1 \
  --learning_rate=2e-4 \
  --lr_warmup_steps=0 \
  --save_model_steps 20000 \
  --dataloader_num_workers 32 \
  --adam_weight_decay 0.00 \
  --ema_max_decay 0.9999 \
  --dropout 0.1 \
  --use_ema \

# --teacher_path='pretrained/ddpm_ema_cifar10/ddpm_ema_cifar10' --alpha=5 --beta=0 --tau=1 \
