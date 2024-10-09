from typing import *
import math
import torch
from torch import nn
import torch.nn.functional as F


class ResBlock3d(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__()

        hidden_channels = min(in_channels, out_channels)
        self.layers = nn.Sequential(
            nn.BatchNorm3d(in_channels),
            nn.ReLU(),
            nn.Conv3d(in_channels, hidden_channels, **kwargs),
            nn.BatchNorm3d(hidden_channels),
            nn.ReLU(),
            nn.Conv3d(hidden_channels, out_channels, **kwargs)
        )

        if in_channels != out_channels:
            self.skip = nn.Conv3d(in_channels, out_channels, kernel_size=1)
        else:
            self.skip = nn.Identity()

    def forward(self, x):
        return self.layers(x) + self.skip(x)


class StackMoreLayers(nn.Module):
    def __init__(self, layer, channels, **kwargs):
        super().__init__()

        self.layers = nn.ModuleList([
            layer(c_in, c_out, **kwargs)
            for c_in, c_out in zip(channels, channels[1:])
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class Lambda(nn.Module):
    def __init__(self, func, **kwargs):
        super().__init__()

        self.func = func
        self.kwargs = kwargs

    def forward(self, *args, **kwargs):
        return self.func(*args, **kwargs, **self.kwargs)


class PartialConv3d(nn.Conv3d):
    def forward(
            self,
            x: torch.Tensor,
            mask: Optional[torch.Tensor] = None
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Partial convolution.

        Args:
            x (torch.Tensor): tensor of size (n, c, h, w, d).
            mask (Optional[torch.Tensor], optional): tensor of size (n, h, w, d). Defaults to None.

        Returns:
            Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]: 
        """
        assert self.padding_mode == 'zeros'

        if mask is None:
            return super().forward(x)

        mask.unsqueeze_(1)  # (n, 1, h, w, d)
        x = F.conv3d(x * mask, self.weight, None, self.stride, self.padding, self.dilation, self.groups)

        with torch.no_grad():
            weight = torch.ones(1, 1, *self.kernel_size).to(mask)
            num_valid = F.conv3d(mask, weight, None, self.stride, self.padding, self.dilation)
            mask = torch.clamp(num_valid, 0, 1)

        x *= math.prod(self.kernel_size) / torch.clamp(num_valid, 1)
        if self.bias is not None:
            x += self.bias.view(-1, 1, 1, 1)
        x *= mask
        mask.squeeze_(1)  # (n, h, w, d)

        return x, mask
