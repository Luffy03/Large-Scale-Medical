# Adapted from https://github.com/sthalles/SimCLR
from trainers.base_trainer import *
import os
import sys
import time
import math

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from collections import OrderedDict

from networks import get_networks, networks_dict
from models import get_models, models_dict

from trainers.pcrl_trainer import MemoryC2L
from utils.losses import NT_Xent_dist
from utils.tools import AverageMeter, save_np2nii
from tqdm import tqdm
from torch.cuda.amp import GradScaler, autocast

# Require fp16
class SimCLRTrainer(BaseTrainer):
    def __init__(self, config):
        super(SimCLRTrainer, self).__init__(config)

        assert config.model == 'SimCLR'

        # init model and opt
        self.init_model()
        self.init_optimizer_and_scheduler()
        self.model_parallel()

        self.check_resume()
        # loss criterion
        # self.criterion = NT_Xent_dist(temperature=0.5, base_temperature=0.07).cuda()
        self.criterion = torch.nn.CrossEntropyLoss().to(self.device)
        self.temperature = 0.07

        # Whether or not to use 16-bit precision GPU trainin
        self.fp16_precision = True
        
    def init_model(self):
        self.model = get_networks(self.config)
        dim_mlp = self.model.fc[0].weight.shape[1]
        if '3d' in self.config.network:
            assert dim_mlp == 512
        else:
            assert dim_mlp == 1024
        self.model.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp),
                                      nn.ReLU(),
                                      nn.Linear(dim_mlp, 256))
        self.model.cuda()

    def init_optimizer_and_scheduler(self):
        if self.config.optimizer == 'sgd':
            self.optimizer = torch.optim.SGD(self.model.parameters(),
                                        lr=self.config.lr,
                                        momentum=self.config.momentum,
                                        weight_decay=self.config.weight_decay)
            self.scheduler = None

        elif self.config.optimizer == 'adam':
            self.optimizer = torch.optim.Adam(self.model.parameters(), self.config.lr, weight_decay=self.config.weight_decay)
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=len(self.train_dataloader), eta_min=0,
                                                               last_epoch=-1)

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

    def info_nce_loss(self, features, n_views=2):

        labels = torch.cat([torch.arange(self.config.train_batch) for i in range(n_views)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.to(self.device)

        features = F.normalize(features, dim=1)

        similarity_matrix = torch.matmul(features, features.T)
        assert similarity_matrix.shape == (
            2 * self.config.train_batch, 2 * self.config.train_batch)
        assert similarity_matrix.shape == labels.shape

        # discard the main diagonal from both: labels and similarities matrix
        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(self.device)
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
        # assert similarity_matrix.shape == labels.shape

        # select and combine multiple positives
        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

        # select only the negatives the negatives
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(self.device)

        logits = logits / self.temperature
        return logits, labels

    def train(self):
        train_losses = []
        top1_acc_all = []
        top5_acc_all = []

        scaler = GradScaler(enabled=self.fp16_precision)

        print("==> training...")

        for epoch in range(self.start_epoch, self.config.epochs + 1):

            if self.scheduler is None:
                if self.optimizer.param_groups[0]['lr'] > 1e-4:
                    self.adjust_learning_rate(epoch, self.config.epochs, self.config.lr, self.optimizer)
                else:
                    if epoch > 1:
                        self.scheduler.step()

            time1 = time.time()

            loss = self.train_one_epoch(epoch, scaler)

            time2 = time.time()
            epoch_time = (time2 - time1)

            # save model
            if epoch % self.config.save_model_freq == 0:
                # compute the ACC
                top1, top5 = self.accuracy(self.logits, self.labels, topk=(1, 5))
                top1_acc_all.append(top1[0])
                top5_acc_all.append(top5[0])

                # record loss
                train_losses.append(loss)
                self.recorder.logger.info(
                    f'Epoch [{epoch}] - epoch_time: {epoch_time}, 'f'train_loss: {loss:.3f}, 'f'top1 acc: {top1[0]:.4f}'
                    f','f'top5 acc: {top5[0]:.4f}')

                # writer
                lr = self.optimizer.param_groups[0]['lr']
                self.recorder.writer.add_scalar('lr', lr, epoch)
                self.recorder.writer.add_scalar('loss', loss, epoch)
                self.recorder.writer.add_scalar('acc/top1', top1[0], epoch)
                self.recorder.writer.add_scalar('acc/top5', top5[0], epoch)

                self.recorder.plot_loss(self.start_epoch, epoch + 1, self.config.save_model_freq, train_losses)

                # Save results
                data_frame = pd.DataFrame(data={'Train_Loss': train_losses,
                                                'Top1 ACC': top1_acc_all,
                                                'Top5 ACC': top5_acc_all},
                                          index=range(self.start_epoch, epoch + 1, self.config.save_model_freq))

                data_frame.to_csv(os.path.join(self.recorder.save_dir, "results.csv"), index_label='epoch')

                # saving the model
                print('==> Saving...')

                state = {'state_dict': self.model.state_dict(),
                         'optimizer': self.optimizer.state_dict(),
                         'epoch': epoch}

                save_file = os.path.join(self.recorder.save_dir, str(epoch) + '.pth')
                torch.save(state, save_file)
                self.recorder.logger.info(
                    "Saving model{} ".format(save_file))
                # help release GPU memory
                del state

            torch.cuda.empty_cache()

    def train_one_epoch(self, epoch, scaler):
        """one epoch training"""
        self.model.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()

        num_iter = len(self.train_dataloader)
        end = time.time()

        for idx, (input1, input2) in enumerate(self.train_dataloader):
            bsz = input1.shape[0]
            images = torch.cat([input1, input2], dim=0)
            images = images.to(self.device)

            # measure data time
            data_time.update(time.time() - end)

            # compute loss
            with autocast(enabled=self.fp16_precision):
                features = self.model(images)
                self.logits, self.labels = self.info_nce_loss(features)
                loss = self.criterion(self.logits, self.labels)
                losses.update(loss.item(), bsz)

            # compute gradient and do SGD step
            self.optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(self.optimizer)
            scaler.update()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # print info
            if (idx + 1) % 20 == 0:
                lr = self.optimizer.param_groups[0]['lr']
                self.recorder.logger.info(f'Epoch [{epoch}][{idx + 1}/{num_iter}] - '
                                          f'data_time: {data_time.avg:.3f},     '
                                          f'batch_time: {batch_time.avg:.3f},     '
                                          f'lr: {lr:.5f},     '
                                          f'loss: {loss:.3f}({losses.avg:.3f})')

        return losses.avg

    def accuracy(self, output, target, topk=(1,)):
        """Computes the accuracy over the k top predictions for the specified values of k"""
        with torch.no_grad():
            maxk = max(topk)
            batch_size = target.size(0)

            _, pred = output.topk(maxk, 1, True, True)
            pred = pred.t()
            correct = pred.eq(target.view(1, -1).expand_as(pred))

            res = []
            for k in topk:
                correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
                res.append(correct_k.mul_(100.0 / batch_size))
            return res

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





