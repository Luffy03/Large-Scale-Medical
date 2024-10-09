from trainers.base_trainer import *
from sklearn import metrics
from torch.utils.data import DataLoader


class ClassificationTrainer(BaseTrainer):
    def __init__(self, config):
        super(ClassificationTrainer, self).__init__(config)
        assert config.model == 'Simple'

    def random_sampler(self):
        self.train_dataset.random_sampler()
        self.train_dataloader = DataLoader(dataset=self.train_dataset,
                            batch_size=self.config.train_batch,
                            shuffle=True,
                            num_workers=self.config.num_workers,
                            pin_memory=True,
                            drop_last=True)
        self.recorder.logger.info('Random sample the training data')

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
            if self.config.sample_type == 'random':
                if self.config.random_sample_ratio is not None:
                    self.random_sampler()
            train_bar = tqdm(self.train_dataloader)
            for itr, sample in tqdm(enumerate(train_bar)):
                self.set_input(sample)
                self.optimize_parameters()
                train_losses.append(round(self.loss.item(), 2))
                # print('size', self.input.size(), self.target.size(), self.pred.size())
                if (itr + 1) % 500 == 0:
                    self.recorder.logger.info('Epoch [{}/{}], iteration {}, Loss: {:.6f}'
                          .format(epoch + 1, self.config.epochs, itr + 1, np.average(train_losses)))
                    sys.stdout.flush()

            self.recorder.writer.add_scalar('Train/total_loss', np.average(train_losses), epoch)

            if epoch % self.config.val_epoch == 0:
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
                        gt = gt.data.cpu().numpy()
                        pred = pred.data.cpu().numpy()
                        gts.extend(gt)
                        preds.extend(pred)

                    gts = np.asarray(gts)
                    preds = np.asarray(preds)

                    #print("inputs:  {} | {:.1f} ~ {:.1f}".format(image.shape, np.min(image), np.max(image)))
                    # print("gts:  {} | {:.1f} ~ {:.1f}".format(gts.shape, np.min(gts), np.max(gts)))
                    # print("preds:  {} | {:.1f} ~ {:.1f}".format(preds.shape, np.min(preds), np.max(preds)))

                    fpr, tpr, thresholds = metrics.roc_curve(gts, preds, pos_label=1)
                    auc = 100.0 * metrics.auc(fpr, tpr)
                    valid_metric = auc

                    self.recorder.writer.add_scalar('Val/metric', valid_metric, epoch)

                # logging
                train_loss = np.average(train_losses)
                avg_train_losses.append(train_loss)
                avg_valid_metrics.append(valid_metric)
                self.recorder.logger.info(
                    "Epoch {}, validation metric is {:.4f}, training loss is {:.4f}".format(epoch + 1, valid_metric,
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

                if num_epoch_no_improvement == self.config.patience: #or epoch >= self.config.epochs:
                    self.recorder.logger.info("Early Stopping")
                    break
                sys.stdout.flush()

                if self.scheduler is not None:
                    self.scheduler.step()

        self.recorder.logger_shutdown()
        self.recorder.writer.close()
        return






