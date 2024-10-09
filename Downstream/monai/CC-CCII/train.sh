now=$(date +"%Y%m%d_%H%M%S")
logdir=runs/logs
mkdir -p $logdir

torchrun --master_port=25584 main.py \
    --logdir $logdir | tee $logdir/$now.txt