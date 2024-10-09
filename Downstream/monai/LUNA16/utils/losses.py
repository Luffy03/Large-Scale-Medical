import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np


class CrossentropyND(torch.nn.CrossEntropyLoss):
    """
    For 3D volumes.
    Network has to have NO NONLINEARITY!
    """
    def forward(self, inp, target):
        target = target.long()
        num_classes = inp.size()[1]

        i0 = 1
        i1 = 2

        while i1 < len(inp.shape): # this is ugly but torch only allows to transpose two axes at once
            inp = inp.transpose(i0, i1)
            i0 += 1
            i1 += 1

        inp = inp.contiguous()
        inp = inp.view(-1, num_classes)

        target = target.view(-1,)

        return super(CrossentropyND, self).forward(inp, target)


class BCEDiceLoss(nn.Module):
    def __init__(self, bce_weight=0.5, dice_weight=1):
        super(BCEDiceLoss, self).__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.BCE_loss = torch.nn.BCELoss(reduction='mean')

    def forward(self, input, target):
        '''
        :param input: (N,*), input must be the original probability image
        :param target: (N,*) * is any other dims but be the same with input,
        : shape is  N -1 or  N 1 H W
        :return:  sigmod + BCELoss +  sigmod + DiceLoss
        '''
        # N sample`s average
        bce = self.BCE_loss(input, target)
        smooth = 1e-5
        num = target.size(0)
        input = input.view(num, -1)
        target = target.view(num, -1)
        intersection = (input * target)
        dice = (2. * intersection.sum(1) + smooth) / (input.sum(1) + target.sum(1) + smooth)
        dice = 1 - dice.sum() / num
        return self.bce_weight * bce + self.dice_weight * dice


class DiceLoss(nn.Module):
    def __init__(self, dice_weight=1, smooth=1e-5):
        super(DiceLoss, self).__init__()
        self.dice_weight = dice_weight
        self.smooth = smooth

    def forward(self, input, target):
        # input = torch.sigmoid(input)
        num = target.size(0)
        input = input.view(num, -1)
        target = target.view(num, -1)
        intersection = (input * target)
        #print(intersection.sum(1), input.sum(1), target.sum(1))
        dice = (2. * intersection.sum(1) + self.smooth) / (input.sum(1) + target.sum(1) + self.smooth)
        print('********dice in dice loss*******', dice)
        self.loss = 1 - dice.sum() / num
        return self.loss * self.dice_weight


class SoftDiceLoss(nn.Module):
    def __init__(self):
        super(SoftDiceLoss, self).__init__()

    def forward(self, probs, targets):
        num = targets.size(0)
        smooth = 1

        m1 = probs.view(num, -1)
        m2 = targets.view(num, -1)
        intersection = (m1 * m2)

        score = (2. * intersection.sum(1) + smooth) / (m1.sum(1) + m2.sum(1) + smooth)
        score = 1 - score.sum() / num
        return score


class MultiDiceLoss(nn.Module):
    def __init__(self, weights, smooth=1e-5):
        super(MultiDiceLoss, self).__init__()
        self.weights = weights
        self.dice_loss = DiceLoss(smooth=smooth)

    def forward(self, logits, targets):
        assert torch.max(logits) <= 1. and torch.min(logits) >= 0.
        class_num = logits.size()[1]
        assert len(self.weights) == class_num
        loss = 0
        for i in range(0, class_num):
            loss_i = self.weights[i] * self.dice_loss(logits[:, i], targets[:, i])
            loss += loss_i
        loss = loss / np.sum(self.weights)
        return loss


class CE_Dice_Loss(nn.Module):
    def __init__(self, dice_weight):
        super(CE_Dice_Loss, self).__init__()
        self.dice_loss = MultiDiceLoss(dice_weight)
        self.ce_loss = torch.nn.CrossEntropyLoss(reduction='mean')

    def forward(self, logits, targets):
        class_num = logits.size()[1]

        probs = logits.transpose(1, 2).transpose(2, 3).transpose(3, 4).contiguous().view(-1, class_num)

        target_ce = torch.argmax(targets, dim=1)
        target_ce = target_ce.view(target_ce.numel())

        ce_loss = self.ce_loss(probs, target_ce)

        logits = F.softmax(logits, 1)
        dice_loss = self.dice_loss(logits, targets)
        print('dice loss in ce_dice', dice_loss)
        print('ce loss in ce_dice', ce_loss)

        return ce_loss + dice_loss


class CE_Dice_lossv2(nn.Module):
    def __init__(self, dice_kwargs, ce_kwargs):
        super(CE_Dice_lossv2, self).__init__()
        self.ce = CrossentropyND(**ce_kwargs)
        self.dc = MultiDiceLoss(**dice_kwargs)

    def forward(self, net_output, target):
        ce_loss = self.ce(net_output, target)
        logits = F.softmax(net_output, 1)
        dc_loss = self.dc(logits, target)
        loss = ce_loss + dc_loss
        return loss


class BCE_Dice_Loss(nn.Module):
    def __init__(self, bce_weight, dice_weight):
        super(BCE_Dice_Loss, self).__init__()
        self.dice_loss = MultiDiceLoss(dice_weight)
        self.bce_loss = torch.nn.BCELoss()
        self.bce_weight = bce_weight

    def forward(self, logits, targets):
        bce_loss = self.bce_loss(logits, targets)
        dice_loss = self.dice_loss(logits, targets)
        return self.bce_weight*bce_loss+dice_loss


class TverskyLoss(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, pred, target, alpha=0.3, beta=0.7):

        # squeeze channel
        pred = pred.squeeze(dim=1)
        target = target.squeeze(dim=1)

        smooth = 1

        # dice系数的定义
        dice = (pred * target).sum(dim=1).sum(dim=1).sum(dim=1) / ((pred * target).sum(dim=1).sum(dim=1).sum(dim=1)
                                                                   +
                                                                   alpha * (pred * (1 - target)).sum(dim=1).sum(
                    dim=1).sum(dim=1) + beta * ((1 - pred) * target).sum(dim=1).sum(dim=1).sum(dim=1) + smooth)

        # 返回的是dice距离
        return torch.clamp((1 - dice).mean(), 0, 2)

def flatten(tensor):
    """Flattens a given tensor such that the channel axis is first.
    The shapes are transformed as follows:
       (N, C, D, H, W) -> (C, N * D * H * W)
    """
    # number of channels
    C = tensor.size(1)
    # new axis order
    axis_order = (1, 0) + tuple(range(2, tensor.dim()))
    # Transpose: (N, C, D, H, W) -> (C, N, D, H, W)
    transposed = tensor.permute(axis_order)
    # Flatten: (C, N, D, H, W) -> (C, N * D * H * W)
    return transposed.contiguous().view(C, -1)




##################################
# Contrast Loss For PCRL
###################################

class NCECriterion(nn.Module):
    """
    Eq. (12): L_{NCE}
    """

    def __init__(self, n_data):
        super(NCECriterion, self).__init__()
        self.n_data = n_data

    def forward(self, x):
        bsz = x.shape[0]
        m = x.size(1) - 1
        eps = 1e-5

        # noise distribution
        Pn = 1 / float(self.n_data)

        # loss for positive pair
        P_pos = x.select(1, 0)
        log_D1 = torch.div(P_pos, P_pos.add(m * Pn + eps)).log_()

        # loss for K negative pair
        P_neg = x.narrow(1, 1, m)
        log_D0 = torch.div(P_neg.clone().fill_(m * Pn), P_neg.add(m * Pn + eps)).log_()

        loss = - (log_D1.sum(0) + log_D0.view(-1, 1).sum(0)) / bsz

        return loss


# Contrast Loss for SimCLR
# Adapted from https://github.com/HobbitLong/SupContrast/blob/master/losses.py


import torch.distributed as dist
import diffdist


def diff_gather(z):
    '''
        Return: gather list of z
    '''
    gather_z = [torch.zeros_like(z) for _ in range(torch.distributed.get_world_size())]
    gather_z = diffdist.functional.all_gather(gather_z, z)
    # gather_z = torch.cat(gather_z, dim=0)
    return gather_z


class NT_Xent_dist(nn.Module):
    def __init__(self, temperature=0.07, base_temperature=0.07):
        super(NT_Xent_dist, self).__init__()
        self.temperature = temperature
        self.gather_op = diff_gather
        self.world_size = dist.get_world_size()
        self.base_temperature = base_temperature

    def forward(self, feat1, feat2):
        """
        implement based on pos_mask & neg_mask; could also use torch.diag & nn.CrossEntropyLoss
        Args:
            feat1, feat2: feats of view1, view2; feat1.shape == feat2.shape == (batch_size, C)
        Returns:
            A loss scalar.
        """

        bsz_gpu = feat1.shape[0]
        N = bsz_gpu * self.world_size  # total batch_size

        # compute logits
        feat1 = torch.cat(self.gather_op(feat1))
        feat2 = torch.cat(self.gather_op(feat2))
        features = torch.cat([feat1, feat2], dim=0)
        features = F.normalize(features, dim=1)
        logits = torch.matmul(features, features.T) / self.temperature

        # for numerical stability
        logits_max, _ = torch.max(logits, dim=1, keepdim=True)
        logits = logits - logits_max.detach()

        # neg_mask: denominator; mask-out self-contrast cases
        neg_mask = ~torch.eye(2 * N, dtype=torch.bool).cuda()
        # pos_mask: numerator; single positive pair
        pos_mask = torch.zeros((2 * N, 2 * N), dtype=torch.bool).cuda()
        pos_mask[:N, N:] = torch.eye(N)
        pos_mask[N:, :N] = torch.eye(N)

        # compute log_prob
        exp_logits = torch.exp(logits)[neg_mask].view(2 * N, -1)  # on different gpus
        log_prob = logits[pos_mask] - torch.log(exp_logits.sum(1))

        # loss
        loss = -(self.temperature / self.base_temperature) * log_prob.mean()
        return loss
