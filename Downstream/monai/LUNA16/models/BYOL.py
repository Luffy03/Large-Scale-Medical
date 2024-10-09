# Adapted from https://github.com/xyupeng/ContrastiveCrop/blob/305ff6d06f7e81a5a9f7f80480d81adc11436f9b/models/byol.py
import torch
import torch.nn as nn
import torch.nn.functional as F

import copy
from functools import wraps

def singleton(cache_key):
    def inner_fn(fn):
        @wraps(fn)
        def wrapper(self, *args, **kwargs):
            instance = getattr(self, cache_key)
            if instance is not None:
                return instance

            instance = fn(self, *args, **kwargs)
            setattr(self, cache_key, instance)
            return instance
        return wrapper
    return inner_fn


def set_requires_grad(model, val):
    for p in model.parameters():
        p.requires_grad = val


class BYOL(nn.Module):
    """
    Build a BYOL model. https://arxiv.org/abs/2006.07733
    """
    def __init__(self, encoder_q, hidden_dim=4096, pred_dim=256, encoder_channel=512, m=0.996):
        """
        encoder_q: online network
        encoder_k: target network
        dim: feature dimension (default: 4096)
        pred_dim: hidden dimension of the predictor (default: 256)
        """
        super(BYOL, self).__init__()

        self.encoder_q = encoder_q
        self.encoder_k = copy.deepcopy(self.encoder_q)
        set_requires_grad(self.encoder_k, False)
        self.m = m

        # projector
        # encoder_dim = self.encoder_q.fc.weight.shape[1]
        self.encoder_q.fc = nn.Sequential(#self.encoder_q.fc,
                                          nn.Linear(encoder_channel, hidden_dim),
                                          nn.BatchNorm1d(hidden_dim),
                                          nn.ReLU(inplace=True),
                                          nn.Linear(hidden_dim, pred_dim))

        self.encoder_k.fc = nn.Sequential(# self.encoder_k.fc,
                                          nn.Linear(encoder_channel, hidden_dim),
                                          nn.BatchNorm1d(hidden_dim),
                                          nn.ReLU(inplace=True),
                                          nn.Linear(hidden_dim, pred_dim))

        self.predictor = nn.Sequential(nn.Linear(pred_dim, hidden_dim),
                                       nn.BatchNorm1d(hidden_dim),
                                       nn.ReLU(inplace=True),
                                       nn.Linear(hidden_dim, pred_dim))

    # @singleton('encoder_k')
    # def _get_encoder_k(self):
    #     encoder_k = copy.deepcopy(self.encoder_q)
    #     set_requires_grad(encoder_k, False)
    #     return encoder_k

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    def forward(self, x1, x2):
        """
        Input:
            x1: first views of images
            x2: second views of images
        """

        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder

        p1 = self.predictor(self.encoder_q(x1))  # NxC
        z2 = self.encoder_k(x2)  # NxC

        p2 = self.predictor(self.encoder_q(x2))  # NxC
        z1 = self.encoder_k(x1)  # NxC

        return p1, p2, z1.detach(), z2.detach()
