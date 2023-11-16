export CUDA_VISIBLE_DEVICES=4
export PYTHONWARNINGS="ignore"

python ldm_train.py \
  --dataset="data/celeba_hq_images" \
  --model_path="pretrained/ldm_celeba_256" \
  --resolution=64 \
  --output_dir="run/finetuned/ldm-celebahq-256-test" \
  --train_batch_size=16 \
  --num_iters=10000 \
  --gradient_accumulation_steps=1 \
  --learning_rate=2e-4 \
  --lr_warmup_steps=0 \
  --save_model_steps 2000  \
  --dataloader_num_workers 4 \
  --adam_weight_decay 0.00 \
  --ema_max_decay 0.9999 \
  --dropout 0.1 \
  --use_ema \

#  --pruned_model_ckpt="run/pruned/ldm-celebahq-256/pruned/unet_pruned.pth" \

#  --teacher_path='pretrained/ldm_celeba_256' --alpha=0 --beta=0 --tau=1 \