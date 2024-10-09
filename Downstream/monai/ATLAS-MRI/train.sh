now=$(date +"%Y%m%d_%H%M%S")
name=VoCo
pretrained_root=/pretrained
logdir=runs/logs_swin_base_VoComni
feature_size=48
data_dir=/data/ATLAS-MRI/
cache_dir=/data/cache/ATLAS-MRI
use_ssl_pretrained=False
use_persistent_dataset=True

mkdir -p $logdir

torchrun --master_port=21503 main.py \
    --name $name \
    --pretrained_root $pretrained_root \
    --feature_size $feature_size \
    --data_dir $data_dir \
    --cache_dir $cache_dir \
    --use_ssl_pretrained $use_ssl_pretrained \
    --use_persistent_dataset $use_persistent_dataset \
    --logdir $logdir | tee $logdir/$now.txt