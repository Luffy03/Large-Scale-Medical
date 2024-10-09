now=$(date +"%Y%m%d_%H%M%S")
logdir=runs/logs_swin_B
mkdir -p $logdir

torchrun --master_port=28804 voco_train.py \
    --logdir $logdir | tee $logdir/$now.txt