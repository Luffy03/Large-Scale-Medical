now=$(date +"%Y%m%d_%H%M%S")
logdir=runs/logs_omni_swin_base
pretrained_root=/pretrained
noamp=False
max_epochs=30
batch_size=4
feature_size=48
data_dir=/data/VoComni
cache_dir=/data/cache/VoComni
use_ssl_pretrained=False
use_persistent_dataset=True

mkdir -p $logdir

torchrun --master_port=21503 main.py \
    --noamp $noamp \
    --max_epochs $max_epochs \
    --batch_size $batch_size \
    --pretrained_root $pretrained_root \
    --feature_size $feature_size \
    --data_dir $data_dir \
    --cache_dir $cache_dir \
    --use_ssl_pretrained $use_ssl_pretrained \
    --use_persistent_dataset $use_persistent_dataset \
    --logdir $logdir | tee $logdir/$now.txt