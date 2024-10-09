"""
This trainer is designed for 3D segmentation, which only focuses on a small region of interest.
"""
from trainers.base_trainer import *
from sklearn import metrics
from utils.metrics import dice, iou


class Seg3DROITrainer(BaseTrainer):
    def __init__(self, config):
        super(Seg3DROITrainer, self).__init__(config)
        assert config.model == 'Simple'

    def train(self):
        # to track the training loss as the model trains
        train_losses = []
        # to track the validation loss as the model trains
        # to track the average training loss per epoch as the model trains
        avg_train_losses = []
        # to track the average validation loss per epoch as the model trains
        avg_valid_metrics = []

        best_metric = 0
        num_epoch_no_improvement = 0
        sys.stdout.flush()

        for epoch in range(self.start_epoch, self.config.epochs):
            self.model.train()
            self.recorder.logger.info('Epoch: %d/%d lr %e', epoch, self.config.epochs,
                                      self.optimizer.param_groups[0]['lr'])
            if self.fine_tuning_scheme != 'full':
                self.check_freezing_epoch(epoch)
            train_bar = tqdm(self.train_dataloader)
            for itr, sample in tqdm(enumerate(train_bar)):
                self.set_input(sample)
                self.optimize_parameters()
                train_losses.append(round(self.loss.item(), 2))
                print('size', self.input.size(), self.target.size(), self.pred.size())
                if (itr + 1) % 50 == 0:
                    self.recorder.logger.info('Epoch [{}/{}], iteration {}, Loss: {:.6f}'
                          .format(epoch + 1, self.config.epochs, itr + 1, np.average(train_losses)))
                    sys.stdout.flush()

            self.recorder.writer.add_scalar('Train/total_loss', np.average(train_losses), epoch)

            if epoch % self.config.val_epoch == 0 :
                with torch.no_grad():
                    self.model.eval()
                    self.recorder.logger.info("validating....")
                    gts = []
                    preds = []
                    for itr, (image, gt, index) in tqdm(enumerate(self.eval_dataloader)):
                        image = image.to(self.device)
                        gt = gt.to(self.device)
                        pred = self.model(image)
                        # to numpy
                        gt = gt.cpu().int().numpy()
                        pred = pred.cpu().numpy()
                        # pred = np.where(pred>0.5, 1, 0)
                        gts.extend(gt)
                        preds.extend(pred)

                    gts = np.array(gts).squeeze()
                    preds = np.array(preds).squeeze()

                    assert len(preds.shape) == 4 and len(gts.shape) == 4

                    # calculate the dice and iou
                    eval_metric = SegMetric_Numpy()
                    DCS_all = []
                    IOU_all = []
                    for i in tqdm(range(gts.shape[0])):
                        eval_metric.update(preds[i], gts[i])
                        acc, SE, SP, PC, F1, js, dc = eval_metric.get_current
                        self.recorder.logger.info(
                            "Val-Epoch {} cur-patient {}  dice:{:.3f} iou:{:.3f} acc:{:.3f} SE:{:.3f} SP:{:.3f} F1:{:.3f} PC:{:.3f}"
                                .format(epoch, index, dc, js, acc, SE, SP, F1, PC))
                        DCS_all.append(dc)
                        IOU_all.append(js)

                    dice_avg, iou_avg = np.average(DCS_all), np.average(IOU_all)
                    acc, SE, SP, PC, F1, js, dc = eval_metric.get_avg


                    print("y:  {} | {:.1f} ~ {:.1f}".format(gts.shape, np.min(gts), np.max(gts)))
                    print("p:  {} | {:.1f} ~ {:.1f}".format(preds.shape, np.min(preds), np.max(preds)))
                    print("[ALL]  Dice = {:.2f}%".format(100.0 * dice(preds, gts)))
                    print("[ALL]  IoU  = {:.2f}%".format(100.0 * iou(preds, gts)))
                    print("[AVG] Dice = {:.2f}%".format(100.0 * dc))
                    print("[AVG] IoU  = {:.2f}%".format(100.0 * js))

                    valid_metric = dice_avg

                    self.recorder.writer.add_scalar('Val/metric', valid_metric, epoch)


                # logging
                train_loss = np.average(train_losses)
                avg_train_losses.append(train_loss)
                avg_valid_metrics.append(valid_metric)
                self.recorder.logger.info("Epoch {}, validation metric is {:.4f}, training loss is {:.4f}".format(epoch + 1, valid_metric,
                                                                                        train_loss))
                # Save results
                data_frame = pd.DataFrame(data={'Train_Loss':avg_train_losses,
                                                'Val_AUC': avg_valid_metrics},
                                          index=range(self.start_epoch, epoch + 1, self.config.val_epoch))

                data_frame.to_csv(os.path.join(self.recorder.save_dir, "results.csv"), index_label='epoch')

                # plot loss
                self.recorder.plot_val_metrics(self.start_epoch, epoch + 1, self.config.val_epoch,
                                               avg_valid_metrics)
                self.recorder.plot_loss(self.start_epoch, epoch+1, self.config.val_epoch, avg_train_losses)

                # reset
                train_losses = []


                # early stopping
                if valid_metric > best_metric:
                    self.recorder.logger.info(
                        "Validation metric increases from {:.4f} to {:.4f}".format(best_metric, valid_metric))
                    best_metric = valid_metric
                    num_epoch_no_improvement = 0
                    self.save_state_dict(epoch + 1, os.path.join(self.recorder.save_dir, "model_best.pth"))
                    self.recorder.logger.info(
                        "Saving model{} ".format(os.path.join(self.recorder.save_dir, "model_best.pth")))
                else:
                    self.recorder.logger.info(
                        "Validation metric does not increase from {:.4f}, num_epoch_no_improvement {}".format(
                            best_metric,
                            num_epoch_no_improvement))
                    num_epoch_no_improvement += 1

                if num_epoch_no_improvement == self.config.patience:
                    self.recorder.logger.info("Early Stopping")
                    break
                sys.stdout.flush()

                if self.scheduler is not None:
                    self.scheduler.step()

        self.recorder.logger_shutdown()
        self.recorder.writer.close()
        return







