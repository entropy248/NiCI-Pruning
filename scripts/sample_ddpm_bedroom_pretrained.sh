export CUDA_VISIBLE_DEVICES=3
export PYTHONWARNINGS="ignore"
python ddpm_sample.py \
--output_dir run/sample/ddpm_bedroom_pretrained \
--batch_size 64 \
--total_samples 30000 \
--model_path pretrained/ddpm_bedroom \