# Modified the source code according to the paper
from trainers.base_trainer import BaseTrainer
import os
import sys
import time
import math

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict


from datasets_3D import get_dataloder_3D, datasets_dict_3D
from networks import get_networks, networks_dict
from models import get_models, models_dict

from trainers.pcrl_trainer import MemoryC2L
from utils.losses import NCECriterion
from utils.tools import AverageMeter, save_np2nii
from tqdm import tqdm


class BYOLTrainer(BaseTrainer):
    def __init__(self, config):
        super(BYOLTrainer, self).__init__(config)

        assert config.model == 'BYOL'

        # init model and opt
        self.init_model()
        self.init_optimizer_and_scheduler()
        self.model_parallel()

        self.check_resume()

        # loss criterion
        self.criterion = nn.CosineSimilarity(dim=1).cuda()

    def init_model(self):
        self.encoder = get_networks(self.config)
        self.model = get_models(self.config, self.encoder)
        self.model.cuda()

    def init_optimizer_and_scheduler(self):
        self.optimizer = torch.optim.SGD(self.model.parameters(),
                                    lr=self.config.lr,
                                    momentum=self.config.momentum,
                                    weight_decay=self.config.weight_decay)

    def model_parallel(self):
        self.model = nn.DataParallel(self.model, device_ids=self.config.gpu_ids).to(self.device)

    def check_resume(self):
        self.start_epoch = 0
        if self.config.resume is not None:
            self.recorder.logger.info("Loading model from checkpoint '{}'".format(self.config.resume))
            self.resume_checkpoint = torch.load(self.config.resume, map_location=self.device)
            self.start_epoch = self.resume_checkpoint['epoch']
            self.model.load_state_dict(self.resume_checkpoint['model_state_dict'], strict=True)
            self.recorder.logger.info("Loading optimizer from checkpoint.")
            self.optimizer.load_state_dict(self.resume_checkpoint['optimizer'])

    def train(self):
        train_losses = []
        print("==> training...")
        for epoch in range(self.start_epoch, self.config.epochs + 1):

            self.adjust_learning_rate(epoch, self.config.epochs, self.config.lr, self.optimizer)

            time1 = time.time()

            loss = self.train_one_epoch(epoch, self.criterion)

            time2 = time.time()
            epoch_time = (time2 - time1)

            # save model
            if epoch % self.config.save_model_freq == 0:
                # record loss
                train_losses.append(loss)
                self.recorder.logger.info(
                    f'Epoch [{epoch}] - epoch_time: {epoch_time}, 'f'train_loss: {loss:.3f}')
                # writer
                lr = self.optimizer.param_groups[0]['lr']
                self.recorder.writer.add_scalar('lr', lr, epoch)
                self.recorder.writer.add_scalar('loss', loss, epoch)
                self.recorder.plot_loss(self.start_epoch, epoch + 1, self.config.save_model_freq, train_losses)

                # saving the model
                print('==> Saving...')

                pretrained_state_dict = OrderedDict()
                for k, v in self.model.module.encoder_q.state_dict().items():
                    name = 'module.' + k
                    pretrained_state_dict[name] = v

                pretrained_ema_state_dict = OrderedDict()
                for k, v in self.model.module.encoder_k.state_dict().items():
                    name = 'module.' + k
                    pretrained_ema_state_dict[name] = v

                state = {'state_dict': pretrained_ema_state_dict,
                         'optimizer': self.optimizer.state_dict(),
                         'epoch': epoch,
                         'online_state_dict': pretrained_state_dict,
                         'model_state_dict': self.model.state_dict()}

                save_file = os.path.join(self.recorder.save_dir, str(epoch) + '.pth')
                torch.save(state, save_file)
                self.recorder.logger.info(
                    "Saving model{} ".format(save_file))
                # help release GPU memory
                del state

            torch.cuda.empty_cache()

    def train_one_epoch(self, epoch, criterion):
        self.model.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()

        num_iter = len(self.train_dataloader)
        end = time.time()

        # train_bar = tqdm(self.train_dataloader)

        for idx, (input1, input2) in enumerate(self.train_dataloader):
            input1 = input1.cuda(non_blocking=True)
            input2 = input2.cuda(non_blocking=True)

            # measure data loading time
            data_time.update(time.time() - end)

            # compute output
            p1, p2, z1, z2 = self.model(x1=input1, x2=input2)
            loss = -2 * (criterion(p1, z2).mean() + criterion(p2, z1).mean())
            # loss2 = self.loss_fn(p1, z2) + self.loss_fn(p2, z1)
            ## 4+loss == loss2.mean()
            # print('*********loss*******', 4+loss, loss2.mean())
            losses.update(loss.item(), input1.size(0))

            # compute gradient and do SGD step
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # measure elapsed time
            batch_time.update(time.time() - end)

            # print info
            if (idx + 1) % 10 == 0:
                lr = self.optimizer.param_groups[0]['lr']
                self.recorder.logger.info(f'Epoch [{epoch}][{idx + 1}/{num_iter}]-'
                            f'data_time: {data_time.avg:.3f},     '
                            f'batch_time: {batch_time.avg:.3f},     '
                            f'lr: {lr:.5f},     '
                            f'loss: {loss:.3f}({losses.avg:.3f})')

        return losses.avg

    def eval(self, epoch):
        pass

    def adjust_learning_rate(self, epoch, epochs, lr, optimizer):
        """Sets the learning rate to the initial LR decayed by 0.2 every steep step"""
        # iterations = opt.lr_decay_epochs.split(',')
        # opt.lr_decay_epochs_list = list([])
        # for it in iterations:
        #     opt.lr_decay_epochs_list.append(int(it))
        # steps = np.sum(epoch > np.asarray(opt.lr_decay_epochs_list))
        # if steps > 0:
        #     new_lr = opt.lr * (opt.lr_decay_rate ** steps)
        #     for param_group in optimizer.param_groups:
        #         param_group['lr'] = new_lr
        lr *= 0.5 * (1. + math.cos(math.pi * epoch / epochs))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    # def adjust_learning_rate(self, epoch, epochs, lr, optimizer):
    #
    #     eta_min = lr * 0.1
    #     lr = eta_min + (lr - eta_min) * (1 + math.cos(math.pi * epoch /epochs)) / 2
    #
    #     # update optimizer lr
    #     for param_group in optimizer.param_groups:
    #         param_group['lr'] = lr

    def loss_fn(self, x, y):
        x = F.normalize(x, dim=-1, p=2)
        y = F.normalize(y, dim=-1, p=2)
        return 2 - 2 * (x * y).sum(dim=-1)






