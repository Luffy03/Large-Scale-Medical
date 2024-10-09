# Modified the source code according to the paper
from trainers.base_trainer import BaseTrainer
import os
import sys
import time
import math

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from collections import OrderedDict


from datasets_3D import get_dataloder_3D, datasets_dict_3D
from networks import get_networks, networks_dict
from models import get_models, models_dict

from trainers.pcrl_trainer import MemoryC2L
from utils.losses import NCECriterion
from utils.tools import AverageMeter, save_np2nii
from tqdm import tqdm

try:
    from apex import amp, optimizers
except ImportError:
    pass


class MYPCRL3DTrainer(BaseTrainer):
    def __init__(self, config):
        super(MYPCRL3DTrainer, self).__init__(config)

        assert config.model == 'PCRL_Model'

        # init model and opt
        self.init_model()
        self.init_optimizer_and_scheduler()
        self.model_parallel()

        nce_k = 16384
        nce_t = self.config.moco_t
        nce_m = 0.5
        self.n_data = len(self.train_dataloader)
        # set the contrast memory
        self.contrast = MemoryC2L(128, self.n_data, nce_k, nce_t, False).cuda()

        self.check_resume()

    def init_model(self):
        base_network = self.config.network

        self.config.network = base_network + '_encoder'

        self.config.is_student = True
        self.encoder = get_networks(self.config)

        self.config.is_student = False
        self.encoder_ema = get_networks(self.config)

        self.config.network = base_network + '_decoder'
        self.decoder = get_networks(self.config)

        self.model = get_models(self.config, [self.encoder, self.encoder_ema, self.decoder])
        self.model.cuda()


    def init_optimizer_and_scheduler(self):
        self.optimizer = torch.optim.SGD(self.model.parameters(),
                                    lr=self.config.lr,
                                    momentum=self.config.momentum,
                                    weight_decay=self.config.weight_decay)

    def model_parallel(self):
        if self.config.use_amp:
            self.model, self.optimizer = amp.initialize(self.model, self.optimizer, opt_level='O1')

        self.model = nn.DataParallel(self.model, device_ids=self.config.gpu_ids).to(self.device)

    def check_resume(self):
        self.start_epoch = 0
        if self.config.resume is not None:
            self.recorder.logger.info("Loading model and ema_model from checkpoint '{}'".format(self.config.resume))
            self.resume_checkpoint = torch.load(self.config.resume, map_location=self.device)
            self.start_epoch = self.resume_checkpoint['epoch']
            self.model.load_state_dict(self.resume_checkpoint['model'], strict=True)
            self.recorder.logger.info("Loading optimizer from checkpoint.")
            self.optimizer.load_state_dict(self.resume_checkpoint['optimizer'])
            self.recorder.logger.info("Loading contrast from checkpoint.")
            self.contrast.load_state_dict(self.resume_checkpoint['contrast'])

    def train(self):
        torch.autograd.set_detect_anomaly(True)
        criterion = NCECriterion(self.n_data).cuda()
        criterion2 = nn.MSELoss().cuda()

        self.model.module.moment_update(self.model.module.encoder, self.model.module.encoder_ema, 0)

        for epoch in range(self.start_epoch, self.config.epochs + 1):

            self.adjust_learning_rate(epoch, self.config.epochs, self.config.lr, self.optimizer)
            print("==> training...")

            time1 = time.time()

            loss, prob = self.train_rep_C2L(epoch, self.contrast, criterion, criterion2)
            time2 = time.time()
            print('epoch {}, loss {:.2f} total time {:.2f}'.format(epoch, loss, time2 - time1))
            # save model
            if epoch % self.config.save_model_freq == 0:
                # saving the model
                print('==> Saving...')
                pretrained_state_dict = OrderedDict()
                for k, v in self.model.module.encoder.state_dict().items():
                    name = 'module.' + k
                    pretrained_state_dict[name] = v
                for k, v in self.model.module.decoder.state_dict().items():
                    name = 'module.' + k
                    pretrained_state_dict[name] = v

                pretrained_ema_state_dict = OrderedDict()
                for k, v in self.model.module.encoder_ema.state_dict().items():
                    name = 'module.' + k
                    pretrained_ema_state_dict[name] = v
                for k, v in self.model.module.decoder.state_dict().items():
                    name = 'module.' + k
                    pretrained_ema_state_dict[name] = v

                state = {'state_dict': pretrained_state_dict ,
                         'contrast': self.contrast.state_dict(),
                         'optimizer': self.optimizer.state_dict(),
                         'epoch': epoch,
                         'ema_state_dict': pretrained_ema_state_dict,
                         'model': self.model.state_dict()}

                save_file = os.path.join(self.recorder.save_dir, str(epoch) + '.pth')
                torch.save(state, save_file)
                self.recorder.logger.info(
                    "Saving model{} ".format(save_file))
                # help release GPU memory
                del state
            if epoch == 242:
                break

            torch.cuda.empty_cache()

    def train_rep_C2L(self, epoch, contrast, criterion, criterion2):
        """
        one epoch training for instance discrimination
        """

        self.model.module.encoder.train()
        self.model.module.decoder.train()
        self.model.module.encoder_ema.eval()

        def set_bn_train(m):
            classname = m.__class__.__name__
            if classname.find('BatchNorm') != -1:
                m.train()

        self.model.module.encoder_ema.apply(set_bn_train)

        batch_time = AverageMeter()
        data_time = AverageMeter()
        c2l_loss_meter = AverageMeter()
        mg_loss_meter = AverageMeter()
        prob_meter = AverageMeter()

        end = time.time()
        for itr, (input1, input2, mask1, mask2, gt1, gt2, mask_h, aug_tensor1, aug_tensor2, aug_tensor_h) in tqdm(enumerate(self.train_dataloader)):
            data_time.update(time.time() - end)
            bsz = input1.size(0)
            x1 = input1.float().to(self.device)
            x2 = input2.float().to(self.device)
            mask1 = mask1.float().to(self.device)
            mask2 = mask2.float().to(self.device)
            mask_h = mask_h.float().to(self.device)
            aug_tensor1 = aug_tensor1.float().to(self.device)
            aug_tensor2 = aug_tensor2.float().to(self.device)
            gt1 = gt1.float().to(self.device)
            gt2 = gt2.float().to(self.device)
            aug_tensor_h = aug_tensor_h.float().to(self.device)
            # ===================forward=====================
            feat_k, feat_q, feat_mixed, Pre_To_x1, Pre_Tm_x2, Pre_Th_x = self.model(x1,
                                                                                    x2,
                                                                                    aug_tensor1,
                                                                                    aug_tensor2,
                                                                                    aug_tensor_h)
            out = contrast(self.Normalize(feat_q), self.Normalize(feat_k))
            out2 = contrast(self.Normalize(feat_q), self.Normalize(feat_mixed))

            c2l_loss = (criterion(out) + criterion(out2)) / 2.
            mg_loss = (criterion2(Pre_To_x1, mask1) + criterion2(Pre_Tm_x2,
            mask2) + criterion2(Pre_Th_x,  mask_h)) / 3.
            loss = c2l_loss + mg_loss
            prob = out[:, 0].mean()

            # ===================backward=====================
            self.optimizer.zero_grad()
            if self.config.use_amp:
                with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
            self.optimizer.step()

            # ===================meters=====================
            mg_loss_meter.update(mg_loss.item(), bsz)
            prob_meter.update(prob.item(), bsz)
            c2l_loss_meter.update(c2l_loss.item(), bsz)

            self.model.module.update_moving_average()

            torch.cuda.synchronize()
            batch_time.update(time.time() - end)
            end = time.time()

            # print info
            if (itr + 1) % 5 == 0:
                self.recorder.logger.info('Train:[{0}][{1}/{2}]\t'
                      'LR {lr}, BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'c2l loss {c2l_loss.val:.3f} ({c2l_loss.avg:.3f})\t'
                      'mg loss {mg_loss.val:.3f} ({mg_loss.avg:.3f})\t'
                      'prob {prob.val:.3f} ({prob.avg:.3f})'.format(
                    epoch, itr + 1, len(self.train_dataloader), lr = self.current_lr, batch_time=batch_time,
                    data_time=data_time, c2l_loss=c2l_loss_meter, mg_loss=mg_loss_meter, prob=prob_meter))
                print(out.shape)
                sys.stdout.flush()

            # save training images
            if epoch % 5 == 0 and (itr+1) % 300 == 0:
                input1_np = input1[0][0].cpu().numpy()
                gt1_np = gt1[0][0].cpu().numpy()
                pred1_np = Pre_To_x1[0][0].detach().cpu().numpy()
                mask1_np = mask1[0][0].cpu().numpy()

                input2_np = input2[0][0].cpu().numpy()
                gt2_np = gt2[0][0].cpu().numpy()
                mask2_np = mask2[0][0].cpu().numpy()
                pred2_np = Pre_Tm_x2[0][0].detach().cpu().numpy()

                pred_alpha_np = Pre_Th_x[0][0].detach().cpu().numpy()
                mask_alpha_np = mask_h[0][0].cpu().numpy()
                #
                # mixed_out1_np = mixed_unet_out[0][0].detach().cpu().numpy()
                # mixed_gt1_np = mixed_gt1[0][0].detach().cpu().numpy()
                image_index = str(itr)
                assert len(input1_np.shape) == 3
                save_path = os.path.join(self.recorder.save_dir, 'train_patch_results_epo_'
                                         + str(epoch)+ '_itr_' + str(itr))

                save_np2nii(input1_np, save_path, 'input1_' + image_index)
                save_np2nii(gt1_np, save_path, 'gt1_' + image_index)
                save_np2nii(mask1_np, save_path, 'mask1_' + image_index)
                save_np2nii(pred1_np, save_path, 'pred1_' + image_index)

                save_np2nii(input2_np, save_path, 'input2_' + image_index)
                save_np2nii(gt2_np, save_path, 'gt2_' + image_index)
                save_np2nii(mask2_np, save_path, 'mask2_' + image_index)
                save_np2nii(pred2_np, save_path, 'pred2_' + image_index)

                save_np2nii(pred_alpha_np, save_path, 'pred_alpha_' + image_index)
                save_np2nii(mask_alpha_np, save_path, 'mask_alpha_' + image_index)


        return mg_loss_meter.avg, prob_meter.avg

    def eval(self, epoch):
        pass

    def Normalize(self, x):
        norm_x = x.pow(2).sum(1, keepdim=True).pow(1. / 2.)
        x = x.div(norm_x)
        return x

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
        self.current_lr = lr




