# Copyright 2020 - 2022 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import os
from time import time
import logging
import numpy as np
import torch
import torch.distributed as dist
import torch.optim as optim
from models.voco_head import VoCoHead
from optimizers.lr_scheduler import WarmupCosineSchedule
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel
from torch.utils.tensorboard import SummaryWriter
from utils.data_utils import *

from utils.ops import *
from utils.utils import AverageMeter, distributed_all_gather
import torch.multiprocessing

torch.multiprocessing.set_sharing_strategy('file_system')

import resource
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (8192, rlimit[1]))


def main():
    def save_ckp(state, checkpoint_dir):
        torch.save(state, checkpoint_dir)

    def train(args, global_step, train_loader, val_best, scaler):
        model.train()
        run_loss = AverageMeter()

        for step, batch in enumerate(train_loader):
            t1 = time()
            img, labels, crops = batch
            img, crops = concat_image(img), concat_image(crops)
            img, crops, labels = img.cuda(), crops.cuda(), labels.cuda()

            with autocast(enabled=args.amp):
                loss = model(img, crops, labels)

            if args.amp:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if args.grad_clip:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.lrdecay:
                scheduler.step()

            optimizer.zero_grad()

            run_loss.update(loss.item(), n=args.batch_size)

            lr = optimizer.param_groups[0]["lr"]

            print_num = 1000
            if args.distributed:
                print_cond = (args.rank == 0) and (global_step % print_num == 0)
            else:
                print_cond = global_step % print_num == 0

            if print_cond:
                print("Step:{}/{}, Loss:{:.4f} "
                      "lr:{:.8f}, Time:{:.4f}".format(global_step, args.num_steps,
                                                      run_loss.avg,
                                                      lr, time() - t1))

            global_step += 1
            if args.distributed:
                val_cond = (args.rank == 0) and (global_step % args.eval_num == 0)
            else:
                val_cond = global_step % args.eval_num == 0

            if val_cond:
                checkpoint = {
                    "global_step": global_step,
                    "state_dict": model.state_dict(),
                }
                save_ckp(checkpoint, logdir + "/model_current_epoch.pt")
                save_ckp(checkpoint, logdir + "/model_step" + str(global_step) + ".pt")

        return global_step, loss, val_best

    roi = 64
    parser = argparse.ArgumentParser(description="PyTorch Training")
    parser.add_argument("--logdir", default="logs", type=str, help="directory to save logs")
    parser.add_argument("--num_steps", default=2000000, type=int, help="number of training iterations")
    parser.add_argument("--eval_num", default=20000, type=int, help="evaluation frequency")
    parser.add_argument("--warmup_steps", default=5000, type=int, help="warmup steps")
    parser.add_argument("--in_channels", default=1, type=int, help="number of input channels")
    parser.add_argument("--feature_size", default=48, type=int, help="embedding size")
    parser.add_argument("--dropout_path_rate", default=0.0, type=float, help="drop path rate")
    parser.add_argument("--use_checkpoint", default=True, help="use gradient checkpointing to save memory")
    parser.add_argument("--spatial_dims", default=3, type=int, help="spatial dimension of input data")
    parser.add_argument("--a_min", default=-175.0, type=float, help="a_min in ScaleIntensityRanged")
    parser.add_argument("--a_max", default=250.0, type=float, help="a_max in ScaleIntensityRanged")
    parser.add_argument("--b_min", default=0.0, type=float, help="b_min in ScaleIntensityRanged")
    parser.add_argument("--b_max", default=1.0, type=float, help="b_max in ScaleIntensityRanged")
    parser.add_argument("--space_x", default=1.5, type=float, help="spacing in x direction")
    parser.add_argument("--space_y", default=1.5, type=float, help="spacing in y direction")
    parser.add_argument("--space_z", default=1.5, type=float, help="spacing in z direction")
    parser.add_argument("--roi_x", default=roi, type=int, help="roi size in x direction")
    parser.add_argument("--roi_y", default=roi, type=int, help="roi size in y direction")
    parser.add_argument("--roi_z", default=roi, type=int, help="roi size in z direction")
    parser.add_argument("--batch_size", default=4, type=int, help="number of batch size")
    parser.add_argument("--sw_batch_size", default=2, type=int, help="number of sliding window batch size")
    parser.add_argument("--lr", default=1e-4, type=float, help="learning rate")
    parser.add_argument("--decay", default=1e-3, type=float, help="decay rate")
    parser.add_argument("--momentum", default=0.9, type=float, help="momentum")
    parser.add_argument("--lrdecay", default=True, help="enable learning rate decay")
    parser.add_argument("--workers", default=16, type=int, help="number of batch size")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="maximum gradient norm")
    parser.add_argument("--opt", default="adamw", type=str, help="optimization algorithm")
    parser.add_argument("--lr_schedule", default="warmup_cosine", type=str)

    # './runs/logs_10k/model_current_epoch.pt'
    parser.add_argument("--resume", default=False, type=str,
                        help="resume training")

    parser.add_argument("--local-rank", type=int, default=0, help="local rank")
    parser.add_argument("--grad_clip", default=False, help="gradient clip")
    parser.add_argument("--noamp", default=False, help="do NOT use amp for training")
    parser.add_argument("--dist-url", default="env://", help="url used to set up distributed training")
    parser.add_argument("--cache", default=True, help="use monai cache Dataset")

    args = parser.parse_args()
    logdir = args.logdir

    args.amp = True
    torch.backends.cudnn.benchmark = True
    # torch.autograd.set_detect_anomaly(True)
    args.distributed = False
    if "WORLD_SIZE" in os.environ:
        args.distributed = int(os.environ["WORLD_SIZE"]) > 1
    args.world_size = 1
    args.rank = 0

    if args.distributed:
        args.device = "cuda:%d" % args.local_rank
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method=args.dist_url)
        args.world_size = torch.distributed.get_world_size()
        args.rank = torch.distributed.get_rank()
        if args.rank == 0:
            print(
                "Training in distributed mode with multiple processes, 1 GPU per process. Process %d, total %d."
                % (args.rank, args.world_size)
            )
    else:
        torch.cuda.set_device(0)
        print("Training with a single process on 1 GPUs.")
    assert args.rank >= 0

    if args.rank == 0:
        os.makedirs(logdir, exist_ok=True)
    logger = init_log('global', logging.INFO)
    logger.propagate = 0

    model = VoCoHead(args)
    model.cuda()

    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if args.rank == 0 or args.distributed is False:
        print("Total parameters count", pytorch_total_params)

    if args.opt == "adam":
        optimizer = optim.Adam(params=model.parameters(), lr=args.lr, weight_decay=args.decay)

    elif args.opt == "adamw":
        optimizer = optim.AdamW(params=model.parameters(), lr=args.lr, amsgrad=True)

    elif args.opt == "sgd":
        optimizer = optim.SGD(params=model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.decay)

    global_step = 0
    if args.resume:
        if args.rank == 0 or args.distributed is False:
            print('resume from previous checkpoints')
        model_pt = os.path.join(args.logdir, 'model_current_epoch.pt')
        # model_pt = '/home/lwubf/VoCo/runs/logs_swin_B/model_current_epoch.pt'

        model_dict = torch.load(model_pt)
        state_dict = model_dict["state_dict"]
        if "module." in list(state_dict.keys())[0]:
            print("Tag 'module.' found in state dict - fixing!")
            for key in list(state_dict.keys()):
                state_dict[key.replace("module.", "")] = state_dict.pop(key)
        model.load_state_dict(state_dict, strict=True)
        global_step = model_dict["global_step"]

    if args.lrdecay:
        if args.lr_schedule == "warmup_cosine":
            scheduler = WarmupCosineSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=args.num_steps)

        elif args.lr_schedule == "poly":

            def lambdas(epoch):
                return (1 - float(epoch) / float(args.epochs)) ** 0.9

            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambdas)

    if args.distributed:
        model = DistributedDataParallel(model, device_ids=[args.local_rank], find_unused_parameters=True)

    train_loader = get_loader(args)

    best_val = 1e8
    if args.amp:
        scaler = GradScaler()
    else:
        scaler = None
    while global_step < args.num_steps:
        global_step, loss, best_val = train(args, global_step, train_loader, best_val, scaler)
    checkpoint = {"epoch": args.epochs, "state_dict": model.state_dict(), "optimizer": optimizer.state_dict()}

    if args.distributed:
        if dist.get_rank() == 0:
            torch.save(model.module.state_dict(), logdir + "final_model.pt")
        dist.destroy_process_group()
    else:
        torch.save(model.state_dict(), logdir + "final_model.pt")
    save_ckp(checkpoint, logdir + "/model_final_epoch.pt")


logs = set()


def init_log(name, level=logging.INFO):
    if (name, level) in logs:
        return
    logs.add((name, level))
    logger = logging.getLogger(name)
    logger.setLevel(level)
    ch = logging.StreamHandler()
    ch.setLevel(level)
    if "SLURM_PROCID" in os.environ:
        rank = int(os.environ["SLURM_PROCID"])
        logger.addFilter(lambda record: rank == 0)
    else:
        rank = 0
    format_str = "[%(asctime)s][%(levelname)8s] %(message)s"
    formatter = logging.Formatter(format_str)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return logger


if __name__ == "__main__":
    main()
