from trainers.base_trainer import BaseTrainer
import torch
import torch.nn as nn
from utils.recorder import Recorder
import sys
from tqdm import tqdm
import numpy as np
import os
import imageio
from utils.tools import AverageMeter



class RKBPTrainer(BaseTrainer):
    def __init__(self, config):
        super(RKBPTrainer, self).__init__(config)
        assert config.model == 'Simple'
        self.bce_criterion = nn.BCELoss(reduction='mean')
        self.bce_criterion.to(self.device)

    def set_input(self, sample):
        input, order_label, hor_rot_label, ver_rot_label, mask_label = sample
        self.input = input.to(self.device)
        self.order_label = order_label.to(self.device)
        self.hor_rot_label = hor_rot_label.to(self.device)
        self.ver_rot_label = ver_rot_label.to(self.device)
        self.mask_label = mask_label.to(self.device)

    def forward(self):
        self.order_pred, self.hor_rot_pred, self.ver_rot_pred, self.mask_pred = self.model(self.input)
        # self.hor_rot_pred = self.hor_rot_pred.view(-1)
        # self.ver_rot_pred = self.ver_rot_pred.view(-1)
        # print('pred size', self.order_pred.size(), self.hor_rot_pred.size(), self.ver_rot_pred.size())


    def backward(self):
        self.order_loss = self.criterion(self.order_pred, self.order_label)
        self.rot_loss = self.bce_criterion(self.hor_rot_pred, self.hor_rot_label) + self.bce_criterion(self.ver_rot_pred, self.ver_rot_label)
        self.mask_loss = self.bce_criterion(self.mask_pred, self.mask_label)
        self.loss = self.order_loss + self.rot_loss + self.mask_loss
        self.loss.backward()

    def optimize_parameters(self):
        self.optimizer.zero_grad()
        self.forward()
        self.backward()
        self.optimizer.step()

    def train(self):
        best_acc = 0
        num_epoch_no_improvement = 0
        sys.stdout.flush()

        for epoch in range(self.start_epoch, self.config.epochs):
            if epoch > 1:
                if self.scheduler is not None:
                    self.scheduler.step()
            self.model.train()
            self.recorder.logger.info('Epoch: %d/%d lr %e', epoch, self.config.epochs,
                                      self.optimizer.param_groups[0]['lr'])
            train_bar = tqdm(self.train_dataloader)
            tloss = AverageMeter()
            for itr, sample in tqdm(enumerate(train_bar)):
                self.set_input(sample)
                self.optimize_parameters()
                tloss.update(round(self.loss.item(), 2), self.config.train_batch)

                if (itr + 1) % 500 == 0:
                    self.recorder.logger.info('Epoch [{}/{}], iteration {}, Train Loss: {:.6f}'
                          .format(epoch + 1, self.config.epochs, itr + 1,
                                  tloss.avg))
                    sys.stdout.flush()

            with torch.no_grad():
                # ACC
                order_valid_acc = 0
                hor_rot_valid_acc = 0
                ver_rot_valid_acc = 0
                mask_valid_acc = 0
                total = 0
                self.model.eval()
                self.recorder.logger.info("Validating....")
                for itr, sample in enumerate(self.eval_dataloader):
                    self.set_input(sample)
                    self.forward()
                    # order acc
                    order_pred = torch.softmax(self.order_pred.data, 1)
                    _, predicted_order_label = torch.max(order_pred.data, 1)

                    # print(predicted_order_label, self.order_label)
                    order_valid_acc += (predicted_order_label == self.order_label).sum().item()
                    total += self.order_label.size(0)

                    # print(self.hor_rot_pred.ge(0.5).int(), self.hor_rot_label)
                    hor_rot_valid_acc += (self.hor_rot_pred.ge(0.5).int() == self.hor_rot_label).sum().item()
                    # rot acc
                    # print(self.ver_rot_pred.ge(0.5).int(), self.ver_rot_label)
                    ver_rot_valid_acc += (self.ver_rot_pred.ge(0.5).int() == self.ver_rot_label).sum().item()

                    # mask acc
                    mask_valid_acc += (self.mask_pred.ge(0.5).int() == self.mask_label).sum()

                order_valid_acc = order_valid_acc / total
                hor_rot_valid_acc = hor_rot_valid_acc / (total * 8)
                ver_rot_valid_acc = ver_rot_valid_acc / (total * 8)
                mask_valid_acc = mask_valid_acc / (total * 8)


            # logging
            self.recorder.logger.info("Epoch {}, order acc is {:4f}, hor rot acc is {:.4f},"
                                      " ver rot acc is {:.4f}, mask acc is {:.4f}".format(epoch + 1, order_valid_acc,
                                                                      hor_rot_valid_acc,
                                                                      ver_rot_valid_acc, mask_valid_acc))
            valid_acc = (order_valid_acc + (hor_rot_valid_acc + ver_rot_valid_acc) / 2 + mask_valid_acc) / 3
            if valid_acc > best_acc:
                self.recorder.logger.info("Validation acc increases from {:.4f} to {:.4f}".format(best_acc, order_valid_acc))
                best_acc = valid_acc
                num_epoch_no_improvement = 0
                self.save_state_dict(epoch+1,  os.path.join(self.recorder.save_dir, "RCB_CT.pth"))
                self.recorder.logger.info("Saving model{} ".format(os.path.join(self.recorder.save_dir, "RCB_CT.pth")))
            else:
                self.recorder.logger.info("Validation acc does not increase from {:.4f}, num_epoch_no_improvement {}".format(best_acc,
                                                                                                          num_epoch_no_improvement))
                num_epoch_no_improvement += 1

            if num_epoch_no_improvement == self.config.patience:
                self.recorder.logger.info("Early Stopping")
                break
            sys.stdout.flush()

        self.recorder.logger_shutdown()
        self.recorder.writer.close()
        return

