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


class JigSawTrainer(BaseTrainer):
    def __init__(self, config):
        super(JigSawTrainer, self).__init__(config)
        assert config.model == 'Simple'

    def set_input(self, sample):
        input, order_label = sample
        self.input = input.to(self.device)
        self.order_label = order_label.to(self.device)

    def forward(self):
        self.order_pred,_, _ = self.model(self.input)

    def backward(self):
        self.loss = self.criterion(self.order_pred, self.order_label)
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

                order_valid_acc = order_valid_acc / total


            # logging
            self.recorder.logger.info("Epoch {}, validation order acc is {:.4f}".format(epoch + 1, order_valid_acc))

            if order_valid_acc > best_acc:
                self.recorder.logger.info("Validation acc increases from {:.4f} to {:.4f}".format(best_acc, order_valid_acc))
                best_acc = order_valid_acc
                num_epoch_no_improvement = 0
                # save model
                self.save_state_dict(epoch+1,  os.path.join(self.recorder.save_dir, "Jigsaw.pth"))
                self.recorder.logger.info("Saving model{} ".format(os.path.join(self.recorder.save_dir, "Jigsaw.pth")))
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
















