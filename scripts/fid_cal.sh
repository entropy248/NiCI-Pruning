export CUDA_VISIBLE_DEVICES=4
export PYTHONWARNINGS="ignore"
# get dataset
#python fid_score.py --save-stats data/church run/fid_stats_church.npz \
#--res 256 --batch-size 1024

python fid_score.py run/sample/ddpm_bedroom_256_pruned_c2c_nkd run/fid_stats_bedroom.npz \
--batch-size 256

#python fid_score.py run/sample/ddpm_cifar10_pruned_c2c_bw1 run/fid_stats_cifar10.npz \
#--batch-size 512

#python fid_score.py run/sample/ddpm_church_256_pruned_c2c_nkd run/fid_stats_church.npz \
#--batch-size 256