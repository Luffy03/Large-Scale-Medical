"""
Our modified PCRL Model according to the description in the paper.
https://openaccess.thecvf.com/content/ICCV2021/papers/Zhou_Preservational_Learning_Improves_Self-Supervised_Medical_Image_Models_by_Reconstructing_Diverse_ICCV_2021_paper.pdf
"""
import torch
import torch.nn as nn
import numpy as np

class PCRLModel3d(nn.Module):
    def __init__(self, encoder, encoder_ema, decoder):
        super(PCRLModel3d, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.encoder_ema = encoder_ema
        # self.set_requires_grad(self.encoder_ema, False)

    def set_requires_grad(self, model, val):
        for p in model.parameters():
            p.requires_grad = val

    def update_moving_average(self):
        assert self.encoder_ema is not None, 'ema encoder has not been created yet'
        self.moment_update(self.encoder, self.encoder_ema, 0.999)

    def moment_update(self, model, model_ema, m):
        """ model_ema = m * model_ema + (1 - m) model """
        for p1, p2 in zip(model.parameters(), model_ema.parameters()):
            # p2.data.mul_(m).add_(1 - m, p1.detach().data)
            p2.data.mul_(m).add_(p1.detach().data, alpha=1 - m)
            if m == 0:
                assert (p1 == p2).all()

    def get_shuffle_ids(self, bsz):
        """generate shuffle ids for ShuffleBN"""
        forward_inds = torch.randperm(bsz).long().cuda()
        backward_inds = torch.zeros(bsz).long().cuda()
        value = torch.arange(bsz).long().cuda()
        backward_inds.index_copy_(0, forward_inds, value)
        return forward_inds, backward_inds

    def forward(self, x1, x2, aug_tensor1, aug_tensor2, aug_tensor_h):
        bsz = x1.shape[0]

        # ===================Encoder O=====================
        feat_q, feats_q = self.encoder(x1)
        # ===================Encoder M=====================
        # Shuffle BN for feat_k
        shuffle_ids, reverse_ids = self.get_shuffle_ids(bsz)
        with torch.no_grad():
            x2 = x2[shuffle_ids]
            feat_k, feats_k = self.encoder_ema(x2)
            feats_k = [tmp[reverse_ids] for tmp in feats_k]
            feat_k = feat_k[reverse_ids]
            x2 = x2[reverse_ids]

            feat_k = feat_k.detach()
            feats_k = [feat.detach() for feat in feats_k]

        # ===================Encoder Hybrid=====================
        alpha = np.random.beta(1., 1.)
        alpha = max(alpha, 1 - alpha)

        out512_o, skip_out64_o, skip_out128_o, skip_out256_o = feats_q
        out512_ema, skip_out64_ema, skip_out128_ema, skip_out256_ema = feats_k
        # from layer_i to layer_L
        out512_alpha = alpha * out512_o + (1 - alpha) * out512_ema
        skip_out256_alpha = alpha * skip_out256_o + (1 - alpha) * skip_out256_ema
        skip_out128_alpha = alpha * skip_out128_o + (1 - alpha) * skip_out128_ema
        skip_out64_alpha = alpha * skip_out64_o + (1 - alpha) * skip_out64_ema

        feats_mixed = [out512_alpha, skip_out64_alpha, skip_out128_alpha, skip_out256_alpha]
        feat_mixed = self.encoder(out512_alpha, mixup=True)

        # Decoder
        Pre_To_x1 = self.decoder(feats_q, aug_tensor1)
        Pre_Tm_x2 = self.decoder(feats_k, aug_tensor2)
        Pre_Th_x = self.decoder(feats_mixed, aug_tensor_h)

        return feat_k, feat_q, feat_mixed, Pre_To_x1, Pre_Tm_x2, Pre_Th_x


class PCRLModel2d(nn.Module):
    def __init__(self, encoder, encoder_ema, decoder):
        super(PCRLModel2d, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.encoder_ema = encoder_ema
        # self.set_requires_grad(self.encoder_ema, False)

    def set_requires_grad(self, model, val):
        for p in model.parameters():
            p.requires_grad = val

    def update_moving_average(self):
        assert self.encoder_ema is not None, 'ema encoder has not been created yet'
        self.moment_update(self.encoder, self.encoder_ema, 0.999)

    def moment_update(self, model, model_ema, m):
        """ model_ema = m * model_ema + (1 - m) model """
        for p1, p2 in zip(model.parameters(), model_ema.parameters()):
            # p2.data.mul_(m).add_(1 - m, p1.detach().data)
            p2.data.mul_(m).add_(p1.detach().data, alpha=1 - m)
            if m == 0:
                assert (p1 == p2).all()

    def get_shuffle_ids(self, bsz):
        """generate shuffle ids for ShuffleBN"""
        forward_inds = torch.randperm(bsz).long().cuda()
        backward_inds = torch.zeros(bsz).long().cuda()
        value = torch.arange(bsz).long().cuda()
        backward_inds.index_copy_(0, forward_inds, value)
        return forward_inds, backward_inds

    def forward(self, x1, x2, aug_tensor1, aug_tensor2, aug_tensor_h):
        bsz = x1.shape[0]

        # ===================Encoder O=====================
        feat_q, feats_q = self.encoder(x1)
        # ===================Encoder M=====================
        # Shuffle BN for feat_k
        shuffle_ids, reverse_ids = self.get_shuffle_ids(bsz)
        with torch.no_grad():
            x2 = x2[shuffle_ids]
            feat_k, feats_k = self.encoder_ema(x2)
            feats_k = [tmp[reverse_ids] for tmp in feats_k]
            feat_k = feat_k[reverse_ids]
            x2 = x2[reverse_ids]

            feat_k = feat_k.detach()
            feats_k = [feat.detach() for feat in feats_k]

        # ===================Encoder Hybrid=====================
        alpha = np.random.beta(1., 1.)
        alpha = max(alpha, 1 - alpha)

        x5_o, x4_o, x3_o, x2_o, x1_o = feats_q
        x5_ema, x4_ema, x3_ema, x2_ema, x1_ema  = feats_k
        # from layer_i to layer_L
        x5_alpha = alpha * x5_o  + (1 - alpha) * x5_ema
        x4_alpha = alpha * x4_o + (1 - alpha) * x4_ema
        x3_alpha = alpha * x3_o + (1 - alpha) * x3_ema
        x2_alpha = alpha * x2_o + (1 - alpha) * x2_ema
        x1_alpha = alpha * x1_o + (1 - alpha) * x1_ema

        feats_mixed = [x5_alpha, x4_alpha, x3_alpha, x2_alpha, x1_alpha]
        feat_mixed = self.encoder(x5_alpha, mixup=True)

        # Decoder
        Pre_To_x1 = self.decoder(feats_q, aug_tensor1)
        Pre_Tm_x2 = self.decoder(feats_k, aug_tensor2)
        Pre_Th_x = self.decoder(feats_mixed, aug_tensor_h)

        return feat_k, feat_q, feat_mixed, Pre_To_x1, Pre_Tm_x2, Pre_Th_x
    

