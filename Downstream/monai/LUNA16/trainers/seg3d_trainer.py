"""
This trainer is designed for 3D segmentation, in which full CT volumes are used.
So the training data need to be randomly cropped
and the sliding window strategy is adopted in the test stage.
"""

from trainers.base_trainer import *
from utils.tools import save_np2nii, Aggregate_3DSeg_tool, one_hot_reverse
import skimage.measure as measure
import copy
import skimage.morphology as morphology
from utils.metrics import dice as cal_dice
import torch.nn.functional as F


class Seg3DTrainer(BaseTrainer):
    def __init__(self, config):
        super(Seg3DTrainer, self).__init__(config)
        assert config.model == 'Simple'
        self.num_class = self.config.class_num

    def train(self):
        """
        Train stage.
        """
        best_metric = 0.0
        num_epoch_no_improvement = 0
        results = {'train_loss': [], 'val_dice': [], 'val_iou': []}
        for epoch in range(self.start_epoch, self.config.epochs):
            self.recorder.logger.info('Epoch: %d/%d lr %e', epoch, self.config.epochs,
                                      self.optimizer.param_groups[0]['lr'])
            self.model.train()
            self.network.train()
            if self.fine_tuning_scheme != 'full':
                self.check_freezing_epoch(epoch)
            tloss_r = AverageMeter()
            train_bar = tqdm(self.train_dataloader)
            for itr, sample in tqdm(enumerate(train_bar)):
                self.set_input(sample)
                self.optimize_parameters()
                tloss_r.update(self.loss.item(), self.input.size(0))
                # train_bar.set_description('[Training Epoch %d/%d]' % (epoch, self.config.epoch))
                train_bar.set_postfix(loss=tloss_r.avg)

                if epoch % 20 == 0 and itr % 5 == 0:
                    save_path = os.path.join(self.recorder.save_dir, 'train_patch_results/epoch_' + str(epoch)+ '_iter_'+ str(itr))
                    # pred : the first channel--the 0th sample(B=1), the second channel--the 0th class(K=1)

                    if self.num_class == 1:
                        input_array = self.input[0][0].cpu().detach().numpy()
                        pred_array = self.pred[0][0].cpu().detach().numpy()
                        target_array = self.target[0][0].cpu().detach().numpy()
                        print('np dice', cal_dice(pred_array, target_array))
                        # images
                        pred_array = np.where(pred_array > 0.5, np.ones_like(pred_array), np.zeros_like(pred_array))
                        target_array = np.where(target_array > 0.5, np.ones_like(target_array), np.zeros_like(target_array))
                        print('save', pred_array.shape, np.min(pred_array), np.max(pred_array))

                        self.recorder.save_3D_images(input_array, pred_array.astype(np.uint8), target_array.astype(np.uint8), self.image_index[0], save_path)
                    else:
                        if self.config.normalization == None:
                            self.pred = F.softmax(self.pred, 1)
                        input_array = self.input[0][0].cpu().detach().numpy()
                        pred_array = self.pred[0].cpu().detach().numpy()
                        target_array = self.target[0].cpu().detach().numpy()
                        print('max value:', pred_array.max(), target_array.max(),
                              pred_array.shape, target_array.shape)

                        # pred/target array: [K, D, H, W]
                        print('np dice', cal_dice(pred_array[0], target_array[0]),
                              cal_dice(pred_array[1], target_array[1]), cal_dice(pred_array[2], target_array[2]))

                        pred_array = np.argmax(pred_array, 0)
                        target_array = np.argmax(target_array, 0)
                          # pred/target array: [D, H, W]
                        print('save', pred_array.shape, np.min(pred_array), np.max(pred_array))

                        self.recorder.save_3D_images(input_array, pred_array.astype(np.uint8),
                                                     target_array.astype(np.uint8), self.image_index[0], save_path)

            self.recorder.logger.info("Epoch {} , Train-loss:{:.3f}".format(epoch, tloss_r.avg))
            self.recorder.writer.add_scalar('Train/total_loss', tloss_r.avg, epoch)
            sys.stdout.flush()

            if epoch % self.config.val_freq == 0:
                self.recorder.logger.info("***************Validation at epoch {}****************".format(epoch))

                dice, iou = self.test_all_cases(epoch)

                self.recorder.writer.add_scalar('Val/dice', dice, epoch)
                self.recorder.writer.add_scalar('Val/iou', iou, epoch)

                results['train_loss'].append(tloss_r.avg)
                results['val_dice'].append(dice)
                results['val_iou'].append(iou)

                data_frame = pd.DataFrame(data={'Train_Loss': results['train_loss'],
                                                'Val_Dice': results['val_dice'],
                                                'Val_IOU': results['val_iou']},
                                          index=range(self.start_epoch, epoch + 1, self.config.val_freq))

                data_frame.to_csv(os.path.join(self.recorder.save_dir, "results.csv"), index_label='epoch')

                self.recorder.plot_loss(self.start_epoch, epoch+1, self.config.val_freq, results['train_loss'])
                self.recorder.plot_val_metrics(self.start_epoch, epoch + 1, self.config.val_freq, results['val_dice'])

                # early stopping
                if dice > best_metric:
                    self.recorder.logger.info(
                        "Validation metric increases from {:.4f} to {:.4f}".format(best_metric, dice))
                    best_metric = dice
                    num_epoch_no_improvement = 0
                    full_path = os.path.join(self.recorder.save_dir, "best_model.pth")
                    self.save_state_dict(epoch, full_path)
                    self.recorder.logger.info("Saving the best model at epoch {} to '{}'".format(epoch, full_path))
                else:
                    self.recorder.logger.info(
                        "Validation metric does not increase from {:.4f}, num_epoch_no_improvement {}".format(
                           best_metric,
                            num_epoch_no_improvement))
                    num_epoch_no_improvement += 1

                    if num_epoch_no_improvement == self.config.patience:
                        self.recorder.logger.info("Early Stopping")
                        break

                    # if self.config.object == 'liver':
                    #     if epoch >= 460:
                    #         self.recorder.logger.info("Too much epochs")
                    #         break

            # if epoch % self.config.save_model_freq == 0:
            #     full_path = os.path.join(self.recorder.save_dir, "latest_model.pth")
            #     self.save_state_dict(epoch, full_path)
            #     self.recorder.logger.info("Saving the latest model at epoch {} to '{}'".format(epoch, full_path))

            if self.scheduler is not None:
                self.scheduler.step()

        self.recorder.logger_shutdown()
        self.recorder.writer.close()
        return

    def test_all_cases(self, epoch):

        if self.num_class == 1:
            # Only calculate the dice of one class
            eval_metric = SegMetric_Numpy()
            DCS_all = []
            self.model.eval()
            with torch.no_grad():
                for iter, sample in tqdm(enumerate(self.eval_dataloader)):
                    if epoch < 150 and iter > 1:
                        break
                    # sample: patches [B=1, N, C, D, H, W], label [B=1, K, D, H, W], image_info: org_shape, new_shape, image_index
                    pred_aggregated, label, image_index = self.test_one_case(sample, epoch)
                    # pred_aggregated : nd.array [K, D, H, W] label: nd.array [K, D, H, W]
                    eval_metric.update(pred_aggregated[0], label[0])
                    acc, SE, SP, PC, F1, js, dc = eval_metric.get_current
                    self.recorder.logger.info(
                        "Val-Epoch {} cur-patient {}  dice:{:.3f} iou:{:.3f} acc:{:.3f} SE:{:.3f} SP:{:.3f} F1:{:.3f} PC:{:.3f}"
                       .format(epoch, image_index, dc, js, acc, SE, SP, F1, PC))
                    DCS_all.append(dc)
                acc, SE, SP, PC, F1, js, dc = eval_metric.get_avg
                self.recorder.logger.info(
                    "Val-Epoch {} avg dice:{:.3f} iou:{:.3f} acc:{:.3f} SE:{:.3f} SP:{:.3f} F1:{:.3f} PC:{:.3f} "
                    "".format(epoch, dc, js, acc, SE, SP, F1, PC))
                self.recorder.logger.info(" max dice:{:.3f} min dice:{:.3f} ".format(max(DCS_all), min(DCS_all)))
                # to save memory
                del pred_aggregated
                del label

                return dc, js
        else:
            # Calculate the dice of different class, respectively.
            eval_metrics = []
            for class_i in range(self.config.class_num - 1):
                eval_metric_i = SegMetric_Numpy()
                eval_metrics.append(eval_metric_i)
            self.model.eval()
            with torch.no_grad():
                for iter, sample in enumerate(self.eval_dataloader):
                    if epoch < 150 and iter > 1:
                        break
                    pred_aggregated, label, image_index = self.test_one_case(sample, epoch)
                    for class_i in range(self.config.class_num - 1):
                        # pred_aggregated : nd.array [K, D, H, W] label: nd.array [K, D, H, W]
                        pred_array = pred_aggregated[class_i + 1]
                        target_array = label[class_i + 1]
                        eval_metrics[class_i].update(pred_array, target_array)
                        acc, SE, SP, PC, F1, js, dc = eval_metrics[class_i].get_current
                        self.recorder.logger.info(
                            "Val-Epoch {} cur-patient {}  class {} dice:{:.3f} iou:{:.3f} acc:{:.3f} SE:{:.3f} SP:{:.3f} F1:{:.3f} PC:{:.3f}"
                            .format(epoch, image_index, class_i + 1, dc, js, acc, SE, SP, F1, PC))

            dice_avg_all_classes = 0.0
            iou_avg_all_classes = 0.0
            for class_i in range(self.config.class_num - 1):
                acc, SE, SP, PC, F1, js, dc = eval_metrics[class_i].get_avg
                self.recorder.logger.info(" Val-Epoch{} Case_Avg Class {} dice:{:.3f} iou:{:.3f} acc:{:.3f} SE:{:.3f} SP:{:.3f}"
                                 " F1:{:.3f} PC:{:.3f} ".format(epoch, class_i, dc, js, acc, SE, SP, F1, PC))

                dice_avg_all_classes += dc
                iou_avg_all_classes += js

            dice_avg_all_classes = dice_avg_all_classes / (self.config.class_num - 1)
            iou_avg_all_classes = iou_avg_all_classes / (self.config.class_num - 1)

            self.recorder.logger.info(
                " Test Case_Avg Class_Avg dice:{:.3f} iou:{:.3f}".format(dice_avg_all_classes, iou_avg_all_classes))
            # to save memory
            del pred_aggregated
            del label

            return dice_avg_all_classes, iou_avg_all_classes

    def test_one_case(self, sample, epoch):
        # sample: patches [B=1, N, C, D, H, W], label [B=1, K, D, H, W], image_info: org_shape, new_shape, image_index
        image, full_image, patches, label, image_info = sample

        patches = patches[0].to(self.device)
        org_shape = image_info['org_shape'][0].numpy()
        new_shape = image_info['new_shape'][0].numpy()
        image_index = image_info['image_index'][0]
        self.aggregater = Aggregate_3DSeg_tool(num_classes=self.config.class_num,
                                               img_org_shape=org_shape,
                                               img_new_shape=new_shape,
                                               C=self.config.test_cut_params)
        for i in tqdm(range(patches.size()[0])):

            patch = patches[i].unsqueeze(0)
            # patch [B=1, C, pd, ph, pw]
            pred = self.network(patch).cpu()
            # pred [B=1, K, pd, ph, pw]
            print('pred-----', pred.size())
            if self.config.normalization == None:
                if self.num_class == 1:
                    pred = F.sigmoid(pred)
                else:
                    pred = F.softmax(pred, 1)
            # pred = np.where(pred > 0.5, 1, 0)
            self.aggregater.add_patch_result(pred)

        pred_aggregated = self.aggregater.recompone_overlap()
        if self.num_class == 1:
            pred_aggregated = torch.where(pred_aggregated > 0.5, torch.ones_like(pred_aggregated), torch.zeros_like(pred_aggregated)).int()
        else:
            pred_aggregated = torch.argmax(pred_aggregated, 0).int()
            print(pred_aggregated.shape, '****** pred aggg multi class')
            pred_aggregated = self.create_one_hot_label(pred_aggregated)

        # to numpy [K, D, H, W]
        pred_aggregated = pred_aggregated.cpu().numpy()
        label = label[0].int().cpu().numpy()

        # save images
        # if epoch % 50 == 0:
        #     save_predictions_path = os.path.join(self.recorder.save_dir, 'test_results' + str(epoch))
        #
        #     image_cur = image[0][0].cpu().numpy()
        #     full_image_cur = full_image[0][0].cpu().numpy()
        #     label_all_classes = one_hot_reverse(label)
        #     pred_aggregated_all_classes = one_hot_reverse(pred_aggregated)
        #
        #     save_np2nii(pred_aggregated_all_classes, save_predictions_path, 'pre' + image_index)
        #     print('saved aggregate pred shape', pred_aggregated_all_classes.shape)
        #     save_np2nii(image_cur, save_predictions_path, 'img' + image_index)
        #     print('saved image shape', image_cur.shape)
        #     save_np2nii(full_image_cur, save_predictions_path, 'padding_img' + image_index)
        #     print('saved full image shape', full_image_cur.shape)
        #     save_np2nii(label_all_classes, save_predictions_path, 'label' + image_index)
        #     print('saved label shape', label_all_classes.shape)

        return pred_aggregated, label, image_index

    def post_processing(self, pred_seg):
        pass

    def create_one_hot_label(self, label):
        """
        Inputlabel:[D,H,W].
        Outputlabel:[K,D,H,W].The output label contains the background class in the 0th channel.
        """
        onehot_label = torch.zeros(
            (self.num_class, label.shape[0], label.shape[1], label.shape[2]))
        for i in range(self.num_class):
            onehot_label[i, :, :, :] = (label == i).int()

        return onehot_label

