now=$(date +"%Y%m%d_%H%M%S")
logdir=runs/logs_swin_B
mkdir -p $logdir

feature_size=48
batch_size=4
sw_batch_size=4

torchrun --master_port=28804 voco_train.py \
    --batch_size $batch_size \
    --sw_batch_size $sw_batch_size \
    --feature_size $feature_size \
    --logdir $logdir | tee $logdir/$now.txt