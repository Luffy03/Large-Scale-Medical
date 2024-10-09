from trainers.base_trainer import BaseTrainer
import torch
import sys
from tqdm import tqdm
import numpy as np
import os


class RotTrainer(BaseTrainer):
    def __init__(self, config):
        super(RotTrainer, self).__init__(config)
        assert config.model == 'Simple'
        self.num_rotations_per_patch = self.config.num_rotations_per_patch
        
    def set_input(self, sample):
        input, target = sample
        if self.config.network.find('3d') != -1:
            batch_size, rotations, c, x, y, z = input.size()
            input = input.view([batch_size * rotations, c, x, y, z])
            target = target.view([batch_size * rotations])
            self.input = input.to(self.device)
            self.target = target.to(self.device)
        else:
            self.input = input.to(self.device)
            self.target = target.to(self.device)

    def train(self):
        # to track the training loss as the model trains
        train_losses = []
        # to track the validation loss as the model trains
        valid_losses = []
        # to track the average training loss per epoch as the model trains
        avg_train_losses = []
        # to track the average validation loss per epoch as the model trains
        avg_valid_losses = []

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
            for itr, sample in tqdm(enumerate(train_bar)):

                self.set_input(sample)
                self.optimize_parameters()
                train_losses.append(round(self.loss.item(), 2))
                print(self.input.size(), self.target.size(), self.pred.size())
                if (itr + 1) % 500 == 0:
                    self.recorder.logger.info('Epoch [{}/{}], iteration {}, Loss: {:.6f}'
                          .format(epoch + 1, self.config.epochs, itr + 1, np.average(train_losses)))
                    sys.stdout.flush()

            if epoch % self.config.val_epoch == 0:
                with torch.no_grad():
                    # ACC
                    valid_acc = 0
                    total = 0
                    self.model.eval()
                    self.recorder.logger.info("validating....")
                    for itr, sample in enumerate(self.eval_dataloader):
                        self.set_input(sample)
                        self.forward()
                        v_loss = self.criterion(self.pred, self.target)
                        valid_losses.append(v_loss.item())
                        pred = torch.softmax(self.pred.data, 1)
                        _, predicted_label = torch.max(pred, 1)
                        count = (predicted_label == self.target).sum()
                        valid_acc += count
                        total += self.target.size(0)
                    valid_acc = valid_acc / total

                # logging
                train_loss = np.average(train_losses)
                valid_loss = np.average(valid_losses)
                avg_train_losses.append(train_loss)
                avg_valid_losses.append(valid_loss)
                self.recorder.logger.info("Epoch {}, validation loss is {:.4f}, training loss is {:.4f}, validation acc is {:.4f}".format(epoch + 1, valid_loss,
                                                                                            train_loss, valid_acc))
                # reset
                train_losses = []
                valid_losses = []

                if valid_acc > best_acc:
                    self.recorder.logger.info("Validation metric increases from {:.4f} to {:.4f}".format(best_acc, valid_acc))
                    best_acc = valid_acc
                    num_epoch_no_improvement = 0
                    # save model
                    self.save_state_dict(epoch+1,  os.path.join(self.recorder.save_dir, "SSM_ROT.pth"))
                    self.recorder.logger.info("Saving model{} ".format(os.path.join(self.recorder.save_dir, "SSM_ROT.pth")))
                else:
                    self.recorder.logger.info("Validation metric does not decrease from {:.4f}, num_epoch_no_improvement {}".format(best_acc,
                                                                                                              num_epoch_no_improvement))
                    num_epoch_no_improvement += 1
              
                sys.stdout.flush()

        self.recorder.logger_shutdown()
        self.recorder.writer.close()
        return
















