export CUDA_VISIBLE_DEVICES=4,6
export PYTHONWARNINGS="ignore"

python -m torch.distributed.launch --nproc_per_node=2 --master_port 22222 --use_env \
ddpm_train.py \
  --dataset="data/church" \
  --model_path="run/finetuned/ddpm_church_256_pruned_c2c_iter1000_step10" \
  --pruned_model_ckpt="run/finetuned/ddpm_church_256_pruned_c2c_iter1000_step10/pruned/unet_pruned.pth" \
  --teacher_path='pretrained/ddpm_church' --alpha=400 --beta=0 --tau=1 \
  --resolution=256 \
  --output_dir="run/finetuned/ddpm_church_256_pruned_c2c_iter1000_step10" \
  --train_batch_size=8 \
  --num_iters=200000 \
  --gradient_accumulation_steps=1 \
  --learning_rate=2e-4 \
  --lr_warmup_steps=0 \
  --save_model_steps 20000 \
  --dataloader_num_workers 4 \
  --adam_weight_decay 0.00 \
  --ema_max_decay 0.9999 \
  --dropout 0.1 \
  --use_ema \

#
