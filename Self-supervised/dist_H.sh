now=$(date +"%Y%m%d_%H%M%S")
logdir=runs/logs_swin_H
mkdir -p $logdir
resume=False
feature_size=192
batch_size=2

python -m torch.distributed.launch \
    --nproc_per_node=8 \
    --master_addr=localhost \
    --master_port=24877 \
    voco_train.py \
    --resume $resume \
    --batch_size $batch_size \
    --feature_size $feature_size \
    --logdir $logdir | tee $logdir/$now.txt