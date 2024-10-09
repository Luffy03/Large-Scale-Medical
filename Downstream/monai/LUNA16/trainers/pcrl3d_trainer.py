from trainers.base_trainer import BaseTrainer
import os
import sys
import time
import math

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn


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


class PCRL3DTrainer(BaseTrainer):
    def __init__(self, config):
        super(PCRL3DTrainer, self).__init__(config)

        assert config.model == 'PCRL_Model'

        # init model and optimizer
        self.init_model()
        self.init_optimizer_and_scheduler()
        self.optimizer_ema = torch.optim.SGD(self.model_ema.parameters(), lr=0, momentum=0, weight_decay=0)
        self.model_parallel()
        # init variables for contrastive learning
        nce_k = 16384
        nce_t = self.config.moco_t
        nce_m = 0.5
        self.n_data = len(self.train_dataloader)
        # set the contrast memory
        self.contrast = MemoryC2L(128, self.n_data, nce_k, nce_t, False).cuda()

        self.check_resume()

    def init_model(self):
        self.config.is_student = True
        self.model = get_networks(self.config)
        self.config.is_student = False
        self.model_ema = get_networks(self.config)
        self.model.cuda()
        self.model_ema.cuda()

    def model_parallel(self):
        if self.config.use_amp:
            self.model, self.optimizer = amp.initialize(self.model, self.optimizer, opt_level='O1')
            self.model_ema, self.optimizer_ema = amp.initialize(self.model_ema, self.optimizer_ema, opt_level='O1')

        self.model = nn.DataParallel(self.model, device_ids=self.config.gpu_ids).to(self.device)
        self.model_ema = nn.DataParallel(self.model_ema, device_ids=self.config.gpu_ids).to(self.device)

    def check_resume(self):
        self.start_epoch = 0
        if self.config.resume is not None:
            self.recorder.logger.info("Loading model and ema_model from checkpoint '{}'".format(self.config.resume))
            self.resume_checkpoint = torch.load(self.config.resume, map_location=self.device)
            self.start_epoch = self.resume_checkpoint['epoch']
            self.model.load_state_dict(self.resume_checkpoint['state_dict'], strict=True)
            self.model_ema.load_state_dict(self.resume_checkpoint['model_ema'], strict=True)
            self.recorder.logger.info("Loading optimizer from checkpoint.")
            self.optimizer.load_state_dict(self.resume_checkpoint['optimizer'])
            self.recorder.logger.info("Loading contrast from checkpoint.")
            self.contrast.load_state_dict(self.resume_checkpoint['contrast'])

    def train(self):
        torch.autograd.set_detect_anomaly(True)
        criterion = NCECriterion(self.n_data).cuda()
        criterion2 = nn.MSELoss().cuda()

        self.moment_update(self.model, self.model_ema, 0)

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
                state = {'state_dict': self.model.state_dict(),
                         'contrast': self.contrast.state_dict(),
                         'optimizer': self.optimizer.state_dict(),
                         'epoch': epoch,
                         'model_ema': self.model_ema.state_dict()}

                save_file = os.path.join(self.recorder.save_dir, str(epoch) + '.pth')
                torch.save(state, save_file)
                # help release GPU memory
                del state
            if epoch == 242:
                break

            torch.cuda.empty_cache()

    def train_rep_C2L(self, epoch, contrast, criterion, criterion2):
        """
        one epoch training for instance discrimination
        """

        self.model.train()
        self.model_ema.eval()

        def set_bn_train(m):
            classname = m.__class__.__name__
            if classname.find('BatchNorm') != -1:
                m.train()

        self.model_ema.apply(set_bn_train)

        batch_time = AverageMeter()
        data_time = AverageMeter()
        c2l_loss_meter = AverageMeter()
        mg_loss_meter = AverageMeter()
        prob_meter = AverageMeter()

        end = time.time()
        for itr, (input1, input2, mask1, mask2, gt1, gt2, aug_tensor1, aug_tensor2) in tqdm(enumerate(self.train_dataloader)):
            data_time.update(time.time() - end)
            bsz = input1.size(0)
            x1 = input1.float().cuda()
            x2 = input2.float().cuda()
            mask1 = mask1.float().cuda()
            mask2 = mask2.float().cuda()
            aug_tensor1 = aug_tensor1.float().cuda()
            aug_tensor2 = aug_tensor2.float().cuda()
            gt1 = gt1.float().cuda()
            gt2 = gt2.float().cuda()
            # ===================forward=====================
            # ids for ShuffleBN
            shuffle_ids, reverse_ids = self.get_shuffle_ids(bsz)
            alpha1 = np.random.beta(1., 1.)
            alpha1 = max(alpha1, 1 - alpha1)
            with torch.no_grad():
                x2 = x2[shuffle_ids]
                feat_k, feats_k = self.model_ema(x2)
                feats_k = [tmp[reverse_ids] for tmp in feats_k]
                feat_k = feat_k[reverse_ids]
                x2 = x2[reverse_ids]
            feat_q, unet_out_alpha, unet_out = self.model(x1, feats_k, alpha1, aug_tensor1, aug_tensor2)
            out = contrast(self.Normalize(feat_q), self.Normalize(feat_k))
            #### mixup
            mixed_x1, mixed_feat1, lam1, index = self.mixup_data_pair(x=x1.clone(),
                                                            y=feat_q.clone())
            mixed_x2, mixed_feat2, _, _ = self.mixup_data_pair(x=x2.clone(), y=feat_k.clone(),
                                                     index=index, lam=lam1)
            mixed_gt1, _, _ = self.mixup_data(x=gt1.clone(),index=index, lam=lam1)
            mixed_gt2,  _, _ = self.mixup_data(x=gt2.clone(), index=index, lam=lam1)

            alpha2 = np.random.beta(1., 1.)
            alpha2 = max(alpha2, 1 - alpha2)
            with torch.no_grad():
                mixed_x2 = mixed_x2[shuffle_ids]
                mixed_feat_k, mixed_feats_k = self.model_ema(mixed_x2)
                mixed_feats_k = [tmp[reverse_ids] for tmp in mixed_feats_k]
                mixed_feat_k = mixed_feat_k[reverse_ids]
                mixed_x2 = mixed_x2[reverse_ids]

            mixed_feat_q, mixed_unet_out_alpha, mixed_unet_out = self.model(mixed_x1, mixed_feats_k, alpha2, aug_tensor1,
                                                                       aug_tensor2, mixup=True)
            mixed_feat_q_norm = self.Normalize(mixed_feat_q)
            mixed_feat_k_norm = self.Normalize(mixed_feat_k)
            mixed_feat1_norm = self.Normalize(mixed_feat1)
            mixed_feat2_norm = self.Normalize(mixed_feat2)

            out2 = contrast(mixed_feat_q_norm, mixed_feat_k_norm)
            out3 = contrast(mixed_feat_q_norm, mixed_feat2_norm)
            c2l_loss = (criterion(out) + criterion(out2) + criterion(out3)) / 3.
            c2l_loss_meter.update(c2l_loss.item(), bsz)
            mask_alpha = alpha1 * mask1 + (1 - alpha1) * mask2
            mg_loss = (criterion2(unet_out, mask1) + criterion2(mixed_unet_out, mixed_gt1)
                       + criterion2(unet_out_alpha, mask_alpha) +
                       criterion2(mixed_unet_out_alpha, alpha2 * mixed_gt1 + (1 - alpha2) * mixed_gt2)) / 4.
            loss = c2l_loss + mg_loss
            prob = out[:, 0].mean()

            ############### wo mix-up for ablation study
            #  c2l_loss = criterion(out)
            ####### unet_out = Unet(x1, aug_tensor1) ; mask1 = Taug_tensor1(x1)
            ####### unet_out_alpha = unet(x1+x2, augtensor1+aug-tensor2), alpha1 * mask1 + (1 - alpha1) * mask2))
            ####### mixed_unet_out = Unet(mixed_x1) ; mixed_gt = gt(mixed_x1)
            ####### mixed_unet_out_alphat = Unet(mixed_x1, mixed_x2) ; alpha2 * mixed_gt1 + (1 - alpha2) * mixed_gt2)
            # mg_loss = (criterion2(unet_out, mask1) + criterion2(unet_out_alpha,
            # mask_alpha)) / 2.
            #
            # loss = c2l_loss + mg_loss
            # prob = out[:, 0].mean()

            # ===================backward=====================
            self.optimizer.zero_grad()
            #loss.backward()
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
            self.optimizer.step()

            # ===================meters=====================
            mg_loss_meter.update(mg_loss.item(), bsz)
            prob_meter.update(prob.item(), bsz)
            c2l_loss_meter.update(c2l_loss.item(), bsz)

            self.moment_update(self.model, self.model_ema, 0.999)

            torch.cuda.synchronize()
            batch_time.update(time.time() - end)
            end = time.time()

            # print info
            if (itr + 1) % 5 == 0:
                self.recorder.logger.info('Train: [{0}][{1}/{2}]\t'
                      'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'c2l loss {c2l_loss.val:.3f} ({c2l_loss.avg:.3f})\t'
                      'mg loss {mg_loss.val:.3f} ({mg_loss.avg:.3f})\t'
                      'prob {prob.val:.3f} ({prob.avg:.3f})'.format(
                    epoch, itr + 1, len(self.train_dataloader), batch_time=batch_time,
                    data_time=data_time, c2l_loss=c2l_loss_meter, mg_loss=mg_loss_meter, prob=prob_meter))
                print(out.shape)
                sys.stdout.flush()

            # save training images
            if epoch % 5 == 0 and (itr+1) % 300 == 0:
                input1_np = input1[0][0].cpu().numpy()
                pred1_np = unet_out[0][0].detach().cpu().numpy()
                mask1_np = mask1[0][0].cpu().numpy()

                input2_np = input2[0][0].cpu().numpy()
                mask2_np = mask2[0][0].cpu().numpy()
                pred_alpha_np = unet_out_alpha[0][0].detach().cpu().numpy()
                mask_alpha_np = mask_alpha[0][0].cpu().numpy()
                #
                # mixed_out1_np = mixed_unet_out[0][0].detach().cpu().numpy()
                # mixed_gt1_np = mixed_gt1[0][0].detach().cpu().numpy()
                image_index = str(itr)
                assert len(input1_np.shape) == 3
                save_path = os.path.join(self.recorder.save_dir, 'test_patch_results' + str(epoch))

                save_np2nii(input1_np, save_path, 'input1_' + image_index)
                save_np2nii(mask1_np, save_path, 'mask1_' + image_index)
                save_np2nii(pred1_np, save_path, 'pred1_' + image_index)

                save_np2nii(input2_np, save_path, 'input2_' + image_index)
                save_np2nii(mask2_np, save_path, 'mask2_' + image_index)

                save_np2nii(pred_alpha_np, save_path, 'pred_alpha_' + image_index)
                save_np2nii(mask_alpha_np, save_path, 'mask_alpha_' + image_index)
                # save_np2nii(mixed_out1_np, save_path, 'mixed_pred1' + image_index)
                # save_np2nii(mixed_gt1_np, save_path, 'mixed_gt1' + image_index)

        return mg_loss_meter.avg, prob_meter.avg

    def eval(self, epoch):
        pass

    def Normalize(self, x):
        norm_x = x.pow(2).sum(1, keepdim=True).pow(1. / 2.)
        x = x.div(norm_x)
        return x

    def moment_update(self, model, model_ema, m):
        """ model_ema = m * model_ema + (1 - m) model """
        for p1, p2 in zip(model.parameters(), model_ema.parameters()):
            #p2.data.mul_(m).add_(1 - m, p1.detach().data)
            p2.data.mul_(m).add_(p1.detach().data, alpha=1-m)

    def get_shuffle_ids(self, bsz):
        """generate shuffle ids for ShuffleBN"""
        forward_inds = torch.randperm(bsz).long().cuda()
        backward_inds = torch.zeros(bsz).long().cuda()
        value = torch.arange(bsz).long().cuda()
        backward_inds.index_copy_(0, forward_inds, value)
        return forward_inds, backward_inds

    def mixup_data_pair(self, x, y, alpha=1.0, index=None, lam=None, use_cuda=True):
        '''Returns mixed inputs, pairs of targets, and lambda'''
        if lam is None:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = lam

        lam = max(lam, 1 - lam)
        batch_size = x.size()[0]
        if index is None:
            index = torch.randperm(batch_size).cuda()
        else:
            index = index

        mixed_x = lam * x + (1 - lam) * x[index, :]
        mixed_y = lam * y + (1 - lam) * y[index]
        return mixed_x, mixed_y, lam, index

    def mixup_data(self, x, alpha=1.0, index=None, lam=None, use_cuda=True):
        '''Returns mixed data, and lambda'''
        if lam is None:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = lam

        lam = max(lam, 1 - lam)
        batch_size = x.size()[0]
        if index is None:
            index = torch.randperm(batch_size).cuda()
        else:
            index = index

        mixed_x = lam * x + (1 - lam) * x[index, :]
        return mixed_x, lam, index

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


