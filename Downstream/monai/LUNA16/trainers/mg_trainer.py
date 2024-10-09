from trainers.base_trainer import BaseTrainer
import torch
from utils.recorder import Recorder
import sys
from tqdm import tqdm
import numpy as np
import os
import imageio


class MGTrainer(BaseTrainer):
    def __init__(self, config):
        super(MGTrainer, self).__init__(config)
        assert config.model == 'Simple'

    def train(self):
        # to track the training loss as the model trains
        train_losses = []
        # to track the validation loss as the model trains
        valid_losses = []
        # to track the average training loss per epoch as the model trains
        avg_train_losses = []
        # to track the average validation loss per epoch as the model trains
        avg_valid_losses = []
        best_loss = 100000
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
            for itr, (image, gt) in enumerate(train_bar):
                image = image.to(self.device)
                gt = gt.to(self.device)
                pred = self.model(image)
                loss = self.criterion(pred, gt)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                train_losses.append(round(loss.item(), 2))
                print(image.size(), gt.size(), pred.size())
                if (itr + 1) % 500 == 0:
                    self.recorder.logger.info('Epoch [{}/{}], iteration {}, Loss: {:.6f}'
                          .format(epoch + 1, self.config.epochs, itr + 1, np.average(train_losses)))
                    sys.stdout.flush()

            with torch.no_grad():
                self.model.eval()
                self.recorder.logger.info("validating....")
                for itr, (image, gt) in enumerate(self.eval_dataloader):
                    image = image.to(self.device)
                    gt = gt.to(self.device)
                    pred = self.model(image)
                    v_loss = self.criterion(pred, gt)
                    valid_losses.append(v_loss.item())
            # logging
            train_loss = np.average(train_losses)
            valid_loss = np.average(valid_losses)
            avg_train_losses.append(train_loss)
            avg_valid_losses.append(valid_loss)
            self.recorder.logger.info("Epoch {}, validation loss is {:.4f}, training loss is {:.4f}".format(epoch + 1, valid_loss,
                                                                                        train_loss))
            # reset losses each epoch
            train_losses = []
            valid_losses = []

            if valid_loss < best_loss:
                self.recorder.logger.info("Validation loss decreases from {:.4f} to {:.4f}".format(best_loss, valid_loss))
                best_loss = valid_loss
                num_epoch_no_improvement = 0
                # save model
                self.save_state_dict(epoch+1,  os.path.join(self.recorder.save_dir, "Mode_Genesis.pth"))
                self.recorder.logger.info("Saving model{} ".format(os.path.join(self.recorder.save_dir, "Mode_Genesis.pth")))
            else:
                self.recorder.logger.info("Validation loss does not decrease from {:.4f}, num_epoch_no_improvement {}".format(best_loss,
                                                                                                          num_epoch_no_improvement))
                num_epoch_no_improvement += 1
            if num_epoch_no_improvement == self.config.patience:
                self.recorder.logger.info("Early Stopping")
                break
            sys.stdout.flush()

        self.recorder.logger_shutdown()
        self.recorder.writer.close()
        return

    def eval(self, epoch):
        mse = []
        with torch.no_grad():
            self.model.eval()
            self.recorder.logger.info("validating....")
            for itr, (image, gt) in enumerate(self.eval_dataloader):
                image = image.to(self.device)
                gt = gt.to(self.device)
                pred = self.model(image)
                mse.append(self.criterion(pred, gt).item())
        avg_mse = np.average(mse)
        print("VAL MSE:", avg_mse)
        return avg_mse




