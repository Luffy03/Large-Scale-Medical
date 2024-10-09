from trainers.base_trainer import *
from utils.metrics import ClsEstimator


class Classification2DTrainer(BaseTrainer):
    def __init__(self, config):
        super(Classification2DTrainer, self).__init__(config)
        assert config.model == 'Simple'
        self.estimator = ClsEstimator(self.config.loss, num_classes=self.config.class_num
                                      , labels=['0', '1', '2', '3', '4'])

    def forward(self):
        """
        Define forward behavior here.
        Args:
            sample (tuple): an input-target pair
        """

        self.pred = self.model(self.input)

    def backward(self):
        if self.config.loss == 'mse':
            self.target = self.target.unsqueeze(1)
        self.target = select_target_type(self.target, self.config.loss)
        self.loss = self.criterion(self.pred, self.target)
        self.loss.backward()

    def optimize_parameters(self):
        self.optimizer.zero_grad()
        self.forward()
        self.backward()
        self.optimizer.step()

    def train(self):
        # to track the training loss as the model trains
        train_losses = []
        # to track the validation loss as the model trains
        # to track the average training loss per epoch as the model trains
        avg_train_losses = []
        # to track the average validation loss per epoch as the model trains
        avg_valid_accs = []
        avg_valid_kappas = []

        best_acc = 0
        best_kappa = 0
        num_epoch_no_improvement = 0
        sys.stdout.flush()

        for epoch in range(self.start_epoch, self.config.epochs):
            self.model.train()
            self.recorder.logger.info('Epoch: %d/%d lr %e', epoch, self.config.epochs,
                                      self.optimizer.param_groups[0]['lr'])
            train_bar = tqdm(self.train_dataloader)
            for itr, sample in tqdm(enumerate(train_bar)):
                self.set_input(sample)
                self.optimize_parameters()
                train_losses.append(round(self.loss.item(), 2))
                if (itr + 1) % 20 == 0:
                    self.recorder.logger.info('Epoch [{}/{}], iteration {}, Loss: {:.6f}'
                                              .format(epoch + 1, self.config.epochs, itr + 1, np.average(train_losses)))
                    sys.stdout.flush()
            self.recorder.writer.add_scalar('Train/total_loss', np.average(train_losses), epoch)

            if epoch % self.config.val_epoch == 0:
                self.estimator.reset()
                with torch.no_grad():
                    self.model.eval()
                    self.recorder.logger.info("validating....")
                    gts = []
                    preds = []
                    for itr, (image, gt, index) in tqdm(enumerate(self.eval_dataloader)):
                        image = image.to(self.device)
                        gt = gt.to(self.device)
                        pred = self.model(image)
                        if self.config.out_ch > 1:
                            pred = torch.softmax(pred, dim=1)
                        else:
                            pred = pred.squeeze()
                            print('pred', pred.size())
                        self.estimator.update(pred, gt)
                        # to numpy
                        image = image.cpu().numpy()
                        gt = gt.cpu().numpy()
                        pred = pred.cpu().numpy()
                        gts.extend(gt)
                        preds.extend(pred)

                    acc = self.estimator.get_accuracy(-1) * 100
                    kappa = self.estimator.get_kappa(-1) * 100
                    valid_metric = kappa
                    self.recorder.writer.add_scalar('Val/acc', acc, epoch)
                    self.recorder.writer.add_scalar('Val/kappa', kappa, epoch)

                # logging
                train_loss = np.average(train_losses)
                avg_train_losses.append(train_loss)
                avg_valid_accs.append(acc)
                avg_valid_kappas.append(kappa)
                self.recorder.logger.info(
                    "Epoch {}, validation metric is {:.4f}, training loss is {:.4f}".format(epoch + 1, valid_metric,
                                                                                            train_loss))
                # Save results
                data_frame = pd.DataFrame(data={'Train_Loss': avg_train_losses,
                                                'Val_ACC': avg_valid_accs,
                                                'Val_Kappa': avg_valid_kappas},
                                          index=range(self.start_epoch, epoch + 1, self.config.val_epoch))

                data_frame.to_csv(os.path.join(self.recorder.save_dir, "results.csv"), index_label='epoch')

                # plot loss
                self.recorder.plot_val_metrics(self.start_epoch, epoch + 1, self.config.val_epoch,
                                               avg_valid_accs)
                self.recorder.plot_loss(self.start_epoch, epoch + 1, self.config.val_epoch, avg_train_losses)

                # reset
                train_losses = []

                # early stopping
                if kappa > best_kappa:
                    self.recorder.logger.info(
                        "Validation kappa increases from {:.4f} to {:.4f}".format(best_kappa, kappa))
                    best_kappa = kappa
                    num_epoch_no_improvement = 0
                    self.save_state_dict(epoch + 1, os.path.join(self.recorder.save_dir, "model_best_kappa.pth"))
                    self.recorder.logger.info(
                        "Saving model{} ".format(os.path.join(self.recorder.save_dir, "model_best_kappa.pth")))
                else:
                    self.recorder.logger.info(
                        "Validation kappa does not increase from {:.4f}, num_epoch_no_improvement {}".format(
                            best_kappa,
                            num_epoch_no_improvement))
                    num_epoch_no_improvement += 1

                if acc > best_acc:
                    self.recorder.logger.info(
                        "Validation acc increases from {:.4f} to {:.4f}".format(best_acc, acc))
                    best_acc = acc
                    num_epoch_no_improvement = 0
                    self.save_state_dict(epoch + 1, os.path.join(self.recorder.save_dir, "model_best_acc.pth"))
                    self.recorder.logger.info(
                        "Saving model{} ".format(os.path.join(self.recorder.save_dir, "model_best_acc.pth")))
                else:
                    self.recorder.logger.info(
                        "Validation acc does not increase from {:.4f}, num_epoch_no_improvement {}".format(
                            best_acc,
                            num_epoch_no_improvement))
                    num_epoch_no_improvement += 1

                if num_epoch_no_improvement >= self.config.patience:  # or epoch >= self.config.epochs:
                    self.recorder.logger.info("Early Stopping")
                    break
                sys.stdout.flush()

                if self.scheduler is not None:
                    self.scheduler.step()

        self.recorder.logger_shutdown()
        self.recorder.writer.close()
        return

