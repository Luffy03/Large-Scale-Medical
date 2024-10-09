"""
This trainer is designed for 2D segmentation.
"""
from trainers.base_trainer import *
from sklearn import metrics
from utils.metrics import dice, iou
from utils.tools import save_np2image, save_tensor2image


class Seg2DROITrainer(BaseTrainer):
    def __init__(self, config):
        super(Seg2DROITrainer, self).__init__(config)
        assert config.model == 'Simple'

    def set_input(self, sample):
        input, target, mask, image_index = sample
        self.input = input.to(self.device)
        self.target = target.to(self.device)
        self.mask = mask.to(self.device)
        self.image_index = image_index

    def train(self):
        # to track the training loss as the model trains
        train_losses = []
        # to track the validation loss as the model trains
        # to track the average training loss per epoch as the model trains
        avg_train_losses = []
        # to track the average validation loss per epoch as the model trains
        avg_valid_dc = []
        avg_valid_acc = []
        avg_valid_dc_fov = []
        avg_valid_acc_fov = []

        best_dc = 0
        best_acc = 0
        num_epoch_no_improvement = 0
        sys.stdout.flush()

        batch_idx = 0
        for epoch in range(self.start_epoch, self.config.epochs):
            self.model.train()
            lr = self.optimizer.param_groups[0]['lr']
            self.recorder.logger.info('Epoch: %d/%d lr %e', epoch, self.config.epochs,
                                      lr)
            train_bar = tqdm(self.train_dataloader)
            for itr, sample in tqdm(enumerate(train_bar)):
                batch_idx += 1
                self.set_input(sample)
                self.optimize_parameters()
                train_losses.append(round(self.loss.item(), 2))
                # print('size', self.input.size(), self.target.size(), self.pred.size())
                if (itr + 1) % 20 == 0:
                    self.recorder.logger.info('Epoch [{}/{}], iteration {}, Loss: {:.6f}'
                                              .format(epoch + 1, self.config.epochs, itr + 1, np.average(train_losses)))
                    # sys.stdout.flush()

            self.recorder.logger.info('Epoch [{}/{}], Avg-Loss: {:.6f}'
                                      .format(epoch + 1, self.config.epochs, np.average(train_losses)))
            self.recorder.writer.add_scalar('Train/total_loss', np.average(train_losses), epoch)

            if epoch % self.config.val_epoch == 0:
                with torch.no_grad():
                    self.model.eval()
                    self.recorder.logger.info("validating....")
                    gts = []
                    preds = []
                    masks = []
                    for itr, (image, gt, mask, index) in tqdm(enumerate(self.eval_dataloader)):
                        image = image.to(self.device)
                        gt = gt.to(self.device)
                        mask = mask.to(self.device)
                        pred = self.model(image)
                        # to numpy
                        gt = gt.cpu().int().numpy()
                        mask = mask.cpu().int().numpy()
                        pred = pred.cpu().numpy()
                        gts.extend(gt)
                        preds.extend(pred)
                        masks.extend(mask)

                    gts = np.array(gts).squeeze()
                    masks = np.array(masks).squeeze()
                    preds = np.array(preds).squeeze()

                    # calculate the dice and iou
                    eval_metric = SegMetric_Numpy()
                    eval_metric_in_fov = SegMetric_Numpy()
                    DCS_all = []
                    IOU_all = []
                    for i in tqdm(range(gts.shape[0])):
                        eval_metric.update(preds[i], gts[i])
                        eval_metric_in_fov.update_in_FOV(preds[i], gts[i], masks[i])
                        acc, SE, SP, PC, F1, js, dc = eval_metric.get_current
                        DCS_all.append(dc)
                        IOU_all.append(js)

                    acc, SE, SP, PC, F1, js, dc = eval_metric.get_avg
                    acc2, SE2, SP2, PC2, F12, js2, dc2 = eval_metric_in_fov.get_avg

                    print("y:  {} | {:.1f} ~ {:.1f}".format(gts.shape, np.min(gts), np.max(gts)))
                    print("p:  {} | {:.1f} ~ {:.1f}".format(preds.shape, np.min(preds), np.max(preds)))
                    print("[AVG] Dice = {:.2f}%".format(100.0 * dc))
                    print("[AVG] Dice-In-FOV  = {:.2f}%".format(100.0 * dc2))
                    print("[AVG] ACC  = {:.2f}%".format(100.0 * acc))
                    print("[AVG] ACC-In-FOV  = {:.2f}%".format(100.0 * acc2))

                    self.recorder.writer.add_scalar('Val/dc', dc2, epoch)
                    self.recorder.writer.add_scalar('Val/acc', acc2, epoch)

                # logging
                train_loss = np.average(train_losses)
                avg_train_losses.append(train_loss)
                avg_valid_dc.append(dc)
                avg_valid_acc.append(acc)
                avg_valid_dc_fov.append(dc2)
                avg_valid_acc_fov.append(acc2)
                self.recorder.logger.info(
                    "Epoch {}, validation metric is DC-{:.4f}, ACC-{:.4f}, training loss is {:.4f}".format(epoch + 1,
                                                                                                           dc2, acc2,
                                                                                                           train_loss))
                # Save results
                data_frame = pd.DataFrame(data={'Train_Loss': avg_train_losses,
                                                'Val_Dice': avg_valid_dc,
                                                'Val_Dice_FOV': avg_valid_dc_fov,
                                                'Val_ACC': avg_valid_acc,
                                                'Val_ACC_FOV': avg_valid_acc_fov
                                                },
                                          index=range(self.start_epoch, epoch + 1, self.config.val_epoch))

                data_frame.to_csv(os.path.join(self.recorder.save_dir, "results.csv"), index_label='epoch')

                # plot loss
                self.recorder.plot_val_metrics(self.start_epoch, epoch + 1, self.config.val_epoch,
                                               avg_valid_dc_fov)
                self.recorder.plot_loss(self.start_epoch, epoch + 1, self.config.val_epoch, avg_train_losses)

                # reset
                train_losses = []

                # early stopping
                if dc2 > best_dc:
                    self.recorder.logger.info(
                        "AVG-Validation metric increases from {:.4f} to {:.4f}".format(best_dc, dc2))
                    best_dc = dc2
                    num_epoch_no_improvement = 0
                    self.model_best_dc_path = os.path.join(self.recorder.save_dir, "model_best_dc.pth")
                    self.save_state_dict(epoch + 1, self.model_best_dc_path)
                    self.recorder.logger.info(
                        "Saving model{} ".format(os.path.join(self.recorder.save_dir, "model_best_dc.pth")))

                if acc2 > best_acc:
                    self.recorder.logger.info(
                        "Validation ACC increases from {:.4f} to {:.4f}".format(best_acc, acc2))
                    best_acc = acc2
                    num_epoch_no_improvement = 0
                    self.model_best_acc_path = os.path.join(self.recorder.save_dir, "model_best_acc.pth")
                    self.save_state_dict(epoch + 1, self.model_best_acc_path)
                    self.recorder.logger.info(
                        "Saving model{} ".format(os.path.join(self.recorder.save_dir, "model_best_acc.pth")))

                if hasattr(self.config, "save_epochs") and self.config.save_epochs == True:
                    self.save_state_dict(epoch + 1,
                                         os.path.join(self.recorder.save_dir, 'epoch_' + str(epoch) + "_model.pth"))
                if dc2 <= best_dc and acc2 <= best_acc:
                    self.recorder.logger.info(
                        "Validation Dice does not increase from {:.4f}, num_epoch_no_improvement {}".format(
                            best_dc,
                            num_epoch_no_improvement))
                    # num_epoch_no_improvement += 1
                    num_epoch_no_improvement += self.config.val_epoch

                    self.save_state_dict(epoch + 1, os.path.join(self.recorder.save_dir, "latest_model.pth"))
                    self.recorder.logger.info(
                        "Saving model{} ".format(os.path.join(self.recorder.save_dir, "latest_model.pth")))

                if num_epoch_no_improvement == self.config.patience:
                    self.recorder.logger.info("Early Stopping")
                    break

                sys.stdout.flush()

                if self.scheduler is not None:
                    self.scheduler.step()

        self.recorder.logger_shutdown()
        self.recorder.writer.close()
        return
