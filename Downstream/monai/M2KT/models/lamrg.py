import ipdb

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from modules.visual_extractor import VisualExtractor
from modules.Transformer import TransformerModel
from modules.text_encoder import TextEncoder, MHA_FF


class _LAMRG(nn.Module):
    def __init__(self, args, tokenizer):
        super(_LAMRG, self).__init__()
        self.args = args
        self.tokenizer = tokenizer
        self.visual_extractor = VisualExtractor(args)
        self.encoder_decoder = TransformerModel(args, tokenizer)
        self.proj = nn.Linear(args.num_labels, args.d_vf)

        self._init_weight(self.proj)

    @staticmethod
    def _init_weight(f):
        nn.init.kaiming_normal_(f.weight)
        f.bias.data.fill_(0)

    def __str__(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + '\nTrainable parameters: {}'.format(params)

    def forward_iu_xray(self, images):
        att_feats_0, fc_feats_0, labels_0 = self.visual_extractor(images[:, 0])
        att_feats_1, fc_feats_1, labels_1 = self.visual_extractor(images[:, 1])

        fc_feats = torch.mean(torch.stack([fc_feats_0, fc_feats_1]), dim=0)
        att_feats = torch.cat((att_feats_0, att_feats_1), dim=1)

        out_labels = torch.mean(torch.stack([labels_0, labels_1]), dim=0)
        return att_feats, fc_feats, out_labels

    def forward_mimic_cxr(self, images):
        att_feats, fc_feats, out_labels = self.visual_extractor(images)
        return att_feats, fc_feats, out_labels
    
    def forward_ctrg(self, images):
        
        att_feats, avg_feats, _ = self.visual_extractor(images)
        '''
        att_feats = []  # local features
        avg_feats = []  # global features

        for i in range(images.size(1)):
            att_feat, avg_feat, _ = self.visual_extractor(images[:, i])
            att_feats.append(att_feat)
            avg_feats.append(avg_feat)

        att_feats = torch.cat(att_feats, dim=1)
        avg_feats = torch.cat(avg_feats, dim=1)
        '''
        return att_feats, avg_feats

    def forward(self, images, targets=None, labels=None, mode='train'):
        if self.args.dataset_name == 'iu_xray':
            att_feats, fc_feats, out_labels = self.forward_iu_xray(images)
        else:
            att_feats, fc_feats, out_labels = self.forward_mimic_cxr(images)

        label_feats = self.proj(out_labels).unsqueeze(1)
        att_feats = torch.cat((att_feats, label_feats), dim=1)
        if mode == 'train':
            output = self.encoder_decoder(fc_feats, att_feats, targets, mode='forward')
        elif mode == 'sample':
            output, _ = self.encoder_decoder(fc_feats, att_feats, opt=self.args, mode='sample')
        else:
            raise ValueError
        return output, out_labels


class LAMRGModel(_LAMRG):
    def __init__(self, args, tokenizer):
        super(LAMRGModel, self).__init__(args, tokenizer)

        self.m = nn.Parameter(torch.FloatTensor(1, args.num_labels, 40, args.d_vf))
        self.init_m()

        if args.dataset_name == 'iu_xray':
            self.forward = self.forward_iu_xray
        else:
            self.forward = self.forward_mimic_cxr

    def init_m(self):
        nn.init.normal_(self.m, 0, 1 / self.args.num_labels)

    def forward_iu_xray(self, images, targets=None, labels=None, mode='train'):
        att_feats_0, fc_feats_0, labels_0 = self.visual_extractor(images[:, 0])
        att_feats_1, fc_feats_1, labels_1 = self.visual_extractor(images[:, 1])

        fc_feats = torch.cat((fc_feats_0, fc_feats_1), dim=1)
        att_feats = torch.cat((att_feats_0, att_feats_1), dim=1)

        out_labels = labels_0

        bs, nf, d_f = att_feats.shape
        _, n_l, n_m, d_f = self.m.shape
        m = labels[:, :, None, None] * self.m.expand(bs, n_l, n_m, d_f)
        m = m.reshape(bs, -1, d_f)
        att_feats = torch.cat((att_feats, m), dim=1)

        if mode == 'train':
            output = self.encoder_decoder(fc_feats, att_feats, targets, mode='forward')
        elif mode == 'sample':
            output, _ = self.encoder_decoder(fc_feats, att_feats, mode='sample')
        else:
            raise ValueError
        return output, out_labels

    def forward_mimic_cxr(self, images, targets=None, labels=None, mode='train'):

        att_feats, fc_feats, out_labels = self.visual_extractor(images)

        bs, nf, d_f = att_feats.shape
        _, n_l, n_m, d_f = self.m.shape
        m = labels[:, :, None, None] * self.m.expand(bs, n_l, n_m, d_f)
        m = m.reshape(bs, -1, d_f)
        att_feats = torch.cat((att_feats, m), dim=1)

        if mode == 'train':
            output = self.encoder_decoder(fc_feats, att_feats, targets, mode='forward')
        elif mode == 'sample':
            output, _ = self.encoder_decoder(fc_feats, att_feats, mode='sample')
        else:
            raise ValueError
        return output, out_labels


class BasicModel(LAMRGModel):
    def forward_iu_xray(self, images, targets=None, labels=None, mode='train'):
        att_feats_0, fc_feats_0, labels_0 = self.visual_extractor(images[:, 0])
        att_feats_1, fc_feats_1, labels_1 = self.visual_extractor(images[:, 1])

        fc_feats = torch.cat((fc_feats_0, fc_feats_1), dim=1)
        att_feats = torch.cat((att_feats_0, att_feats_1), dim=1)

        out_labels = labels_0

        if mode == 'train':
            output = self.encoder_decoder(fc_feats, att_feats, targets, mode='forward')
        elif mode == 'sample':
            output, _ = self.encoder_decoder(fc_feats, att_feats, mode='sample')
        else:
            raise ValueError
        return output, out_labels

    def forward_mimic_cxr(self, images, targets=None, labels=None, mode='train'):

        att_feats, fc_feats, out_labels = self.visual_extractor(images)

        if mode == 'train':
            output = self.encoder_decoder(fc_feats, att_feats, targets, mode='forward')
        elif mode == 'sample':
            output, _ = self.encoder_decoder(fc_feats, att_feats, mode='sample')
        else:
            raise ValueError
        return output, out_labels


class LAMRGModel_base(_LAMRG):
    def __init__(self, args, tokenizer):
        super(LAMRGModel_base, self).__init__(args, tokenizer)
        self.args = args
        self.d_model = args.d_model
        self.d_ff = args.d_ff
        self.num_layers = args.num_layers
        self.num_labels = args.num_labels
        self.tgt_vocab = len(tokenizer.idx2token) + 1
        self.h = args.num_heads
        self.num_slots = args.num_slots
        self.d_vf = args.d_vf
        self.dropout = args.dropout

        self.txt_encoder = TextEncoder(self.d_model, self.d_ff, self.num_layers, self.tgt_vocab, self.num_labels,
                                       self.h, self.dropout)
        self.prior_memory = MHA_FF(self.d_model, self.d_ff, self.h, self.dropout)
        self.select_prior = MHA_FF(self.d_model, self.d_ff, self.h, self.dropout)
        self.memory = self.init_memory()

        self.linear_mem = nn.Linear(self.d_model, self.d_model)
        self.linear_label = nn.Linear(args.num_labels, args.d_model)
        self.linear_feat = nn.Linear(args.d_model, args.d_vf)
        self.linear_fcfeat = nn.Linear(args.d_vf, args.d_model)
        self.init_weight_()

    def init_weight_(self):
        nn.init.kaiming_normal_(self.linear_mem.weight)
        self.linear_mem.bias.data.fill_(0)
        nn.init.kaiming_normal_(self.linear_label.weight)
        self.linear_label.bias.data.fill_(0)
        nn.init.kaiming_normal_(self.linear_feat.weight)
        self.linear_feat.bias.data.fill_(0)

    def init_memory(self):
        memory = nn.Parameter(torch.eye(self.num_slots, device='cuda').unsqueeze(0))
        if self.d_model > self.num_slots:
            diff = self.d_model - self.num_slots
            pad = torch.zeros((1, self.num_slots, diff), device='cuda')
            memory = torch.cat([memory, pad], -1)
        elif self.d_model < self.num_slots:
            memory = memory[:, :, :self.d_model]
        return memory

    def forward(self, images, targets=None, labels=None, mode='train'):
        if self.args.dataset_name == 'iu_xray':
            att_feats, fc_feats, vis_labels = self.forward_iu_xray(images)
        else:
            att_feats, fc_feats, vis_labels = self.forward_mimic_cxr(images)

        z_img = self.linear_fcfeat(fc_feats)

        bsz = att_feats.shape[0]
        # memory = self.linear_mem(self.memory).expand(bsz, -1, -1)
        memory = self.memory.expand(bsz, -1, -1)
        if mode == 'train':
            txt_feats, z_txt, txt_labels = self.txt_encoder(targets)
            memory = self.prior_memory(memory, txt_feats)

        label_feats = self.linear_label(vis_labels).unsqueeze(1)
        prior = self.select_prior(label_feats, memory)
        att_feats = torch.cat((att_feats, self.linear_feat(prior)), dim=1)

        if mode == 'train':
            output = self.encoder_decoder(fc_feats, att_feats, targets, mode='forward')
            return (output, vis_labels, txt_labels, z_img, z_txt)
        elif mode == 'sample':
            # ipdb.set_trace()
            output, _ = self.encoder_decoder(fc_feats, att_feats, opt=self.args, mode='sample')
            return output, vis_labels
        else:
            raise ValueError


class LAMRGModel_VoCo(LAMRGModel_base):
    def __init__(self, args, tokenizer):
        super(LAMRGModel_VoCo, self).__init__(args, tokenizer)
        self.args = args
        self.d_model = args.d_model
        self.d_ff = args.d_ff
        self.num_layers = args.num_layers
        self.num_slices = args.num_slices
        self.num_labels = args.num_labels
        self.tgt_vocab = len(tokenizer.idx2token) + 1
        self.h = args.num_heads
        self.num_slots = args.num_slots
        self.d_vf = args.d_vf
        self.dropout = args.dropout

        self.txt_encoder = TextEncoder(self.d_model, self.d_ff, self.num_layers, self.tgt_vocab, self.num_labels,
                                       self.h, self.dropout)
        self.update_memory = MHA_FF(self.d_model, self.d_ff, args.num_memory_heads, self.dropout)
        self.select_prior = MHA_FF(self.d_model, self.d_ff, args.num_memory_heads, self.dropout)
        self.memory, self.mask = self.init_memory()

        self.get_mem = nn.Linear(self.d_model, self.d_model)
##        self.linear_z = nn.Linear(self.d_vf*self.num_slices, self.d_model)
        self.linear_z = nn.Linear(self.d_vf, self.d_model)
        self.linear_feat = nn.Linear(self.d_model, self.d_vf)

        self.classifier = nn.Linear(self.d_model, self.num_labels)
        self.embed_labels = nn.Linear(1, self.d_model)

        self.init_weight()

    def init_weight(self):
        self._init_weight(self.linear_z)
        self._init_weight(self.get_mem)
        self._init_weight(self.linear_feat)
        self._init_weight(self.classifier)
        self._init_weight(self.embed_labels)

    def init_memory(self):
        memory = nn.Parameter(torch.eye(self.num_slots).unsqueeze(0))
        if self.d_model > self.num_slots:
            diff = self.d_model - self.num_slots
            pad = torch.zeros((1, self.num_slots, diff))
            memory = torch.cat([memory, pad], -1)
        elif self.d_model < self.num_slots:
            memory = memory[:, :, :self.d_model]
        mask = torch.ones((self.num_slots, self.d_model))
        mask[:, self.num_slots:] = 0
        return memory, mask

    def forward(self, images, targets=None, labels=None, mode='train'):
        bsz = images.shape[0]

        att_feats, avg_feats = self.forward_ctrg(images)

        z_img = self.linear_z(avg_feats)
        vis_labels = self.classifier(z_img)

        memory = self.get_mem(self.memory.to(images)).expand(bsz, -1, -1)
        mask = self.mask.to(images).expand(bsz, -1, -1)
        if mode == 'train':
            txt_feats, z_txt, txt_labels = self.txt_encoder(targets)
            memory = self.update_memory(memory, txt_feats, mask)

        emb_labels = self.embed_labels(vis_labels.unsqueeze(-1))
        prior = self.select_prior(emb_labels, memory)
        att_feats = torch.cat((att_feats, self.linear_feat(prior)), dim=1)

        if mode == 'train':
            output = self.encoder_decoder(avg_feats, att_feats, targets, mode='forward')
            return (output, vis_labels, txt_labels, z_img, z_txt)
        elif mode == 'sample':
            output, _ = self.encoder_decoder(avg_feats, att_feats, opt=self.args, mode='sample')
            return output, vis_labels
        else:
            raise ValueError
