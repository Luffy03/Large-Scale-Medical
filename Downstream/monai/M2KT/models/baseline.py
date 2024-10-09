import torch
import torch.nn as nn
import numpy as np

from modules.visual_extractor import VisualExtractor
from modules.Transformer import TransformerModel


class LAMRGModel(nn.Module):
    def __init__(self, args, tokenizer):
        super(LAMRGModel, self).__init__()
        self.args = args
        self.tokenizer = tokenizer
        self.visual_extractor = VisualExtractor(args)
        self.encoder_decoder = TransformerModel(args, tokenizer)
        self.proj = nn.Linear(args.num_labels, args.d_vf)

        self.m = nn.Parameter(torch.FloatTensor(1, args.num_labels, 40, args.d_vf))
        self.init_weight()

    def init_weight(self):
        nn.init.normal_(self.m, 0, 1 / self.args.num_labels)

    def __str__(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + '\nTrainable parameters: {}'.format(params)

    def forward_iu_xray(self, images, targets=None, labels=None, mode='train'):
        att_feats_0, fc_feats_0, labels_0 = self.visual_extractor(images[:, 0])
        att_feats_1, fc_feats_1, labels_1 = self.visual_extractor(images[:, 1])

        fc_feats = torch.cat((fc_feats_0, fc_feats_1), dim=1)
        att_feats = torch.cat((att_feats_0, att_feats_1), dim=1)

        out_labels = labels_0
        return att_feats, fc_feats, out_labels

    def forward_mimic_cxr(self, images, targets=None, labels=None, mode='train'):
        att_feats, fc_feats, out_labels = self.visual_extractor(images)
        return att_feats, fc_feats, out_labels

    def forward(self, images, targets=None, labels=None, mode='train'):
        if self.args.dataset_name == 'iu_xray':
            att_feats, fc_feats, out_labels = self.forward_iu_xray(images, targets, labels)
        else:
            att_feats, fc_feats, out_labels = self.forward_mimic_cxr(images, targets, labels)

        if mode == 'train':
            output = self.encoder_decoder(fc_feats, att_feats, targets, mode='forward')
        elif mode == 'sample':
            output, _ = self.encoder_decoder(fc_feats, att_feats, opt=self.args, mode='sample')
        else:
            raise ValueError
        return output, out_labels
