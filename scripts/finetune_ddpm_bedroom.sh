export CUDA_VISIBLE_DEVICES=4
export PYTHONWARNINGS="ignore"

python ddpm_train.py \
  --dataset="data/bedroom" \
  --model_path="run/pruned/ddpm_ema_bedroom_256_pruned_c2c" \
  --pruned_model_ckpt="run/pruned/ddpm_ema_bedroom_256_pruned_c2c/pruned/unet_pruned.pth" \
  --teacher_path='pretrained/ddpm_bedroom' --alpha=400 --beta=0 --tau=1 \
  --resolution=256 \
  --output_dir="run/finetuned/ddpm_bedroom_256_pruned_c2c_nkd" \
  --train_batch_size=16 \
  --num_iters=200000 \
  --gradient_accumulation_steps=1 \
  --learning_rate=2e-4 \
  --lr_warmup_steps=0 \
  --save_model_steps 40000 \
  --dataloader_num_workers 4 \
  --adam_weight_decay 0.00 \
  --ema_max_decay 0.9999 \
  --dropout 0.1 \
  --use_ema \

#
