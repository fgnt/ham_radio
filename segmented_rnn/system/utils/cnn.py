"""inspired by padertorch.contrib.je.modules.conv"""
import math

import numpy as np
import torch
import torch.nn.functional as F
from padertorch.base import Module
from segmented_rnn.system.utils import to_list, to_pair


class Pad(Module):

    def __init__(self, side='both', mode='constant'):
        super().__init__()
        self.side = side
        self.mode = mode

    def forward(self, x, size):
        sides = to_list(self.side, x.dim() - 2)
        sizes = to_list(size, x.dim() - 2)
        pad = []
        for side, size in list(zip(sides, sizes))[::-1]:
            if side is None or size < 1:
                pad.extend([0, 0])
            elif side == 'front':
                pad.extend([size, 0])
            elif side == 'both':
                pad.extend([size // 2, math.ceil(size / 2)])
            elif side == 'end':
                pad.extend([0, size])
            else:
                raise ValueError(f'pad side {side} unknown')

        x = F.pad(x, tuple(pad), mode=self.mode)
        return x


class Cut(Module):

    def __init__(self, side='both'):
        super().__init__()
        self.side = side

    def forward(self, x, size):
        sides = to_list(self.side, x.dim() - 2)
        sizes = to_list(size, x.dim() - 2)
        slc = [slice(None)] * x.dim()
        for i, (side, size) in enumerate(zip(sides, sizes)):
            idx = 2 + i
            if side is None or size < 1:
                continue
            elif side == 'front':
                slc[idx] = slice(size, x.shape[idx])
            elif side == 'both':
                slc[idx] = slice(size // 2, -math.ceil(size / 2))
            elif side == 'end':
                slc[idx] = slice(0, -size)
            else:
                raise ValueError
        x = x[tuple(slc)]
        return x


class Pool1d(Module):
    def __init__(self, pooling, pool_size, padding='both'):
        super().__init__()
        self.pool_size = pool_size
        self.pooling = pooling
        self.padding = padding

    def forward(self, x):
        if self.pool_size < 2:
            return x, None
        if self.padding is not None:
            pad_size = self.pool_size - 1 - (
                    (x.shape[-1] - 1) % self.pool_size)
            x = Pad(side=self.padding)(x, size=pad_size)
        x = Cut(side='both')(x, size=x.shape[2] % self.pool_size)
        if self.pooling == 'max':
            x, pool_indices = torch.nn.MaxPool1d(
                kernel_size=self.pool_size, return_indices=True
            )(x)
        elif self.pooling == 'avg':
            x = torch.nn.AvgPool1d(kernel_size=self.pool_size)(x)
            pool_indices = None
        else:
            raise ValueError(f'{self.pooling} pooling unknown.')
        return x, pool_indices


class Unpool1d(Module):
    def __init__(self, pool_size):
        super().__init__()
        self.pool_size = pool_size

    def forward(self, x, indices=None):
        if self.pool_size < 2:
            return x
        if indices is None:
            x = F.interpolate(x, scale_factor=self.pool_size)
        else:
            x = torch.nn.MaxUnpool1d(kernel_size=self.pool_size)(
                x, indices=indices
            )
        return x


class Pool2d(Module):
    def __init__(self, pooling, pool_size, padding='both'):
        super().__init__()
        self.pooling = pooling
        self.pool_size = to_pair(pool_size)
        self.padding = to_pair(padding)

    def forward(self, x):
        if all(np.array(self.pool_size) < 2):
            return x, None
        pad_size = (
            self.pool_size[0] - 1 - ((x.shape[-2] - 1) % self.pool_size[0]),
            self.pool_size[1] - 1 - ((x.shape[-1] - 1) % self.pool_size[1])
        )
        pad_size = np.where([pad is None for pad in self.padding], 0, pad_size)
        if any(pad_size > 0):
            x = Pad(side=self.padding)(x, size=pad_size)
        x = Cut(side='both')(x, size=np.array(x.shape[2:]) % self.pool_size)
        if self.pooling == 'max':
            x, pool_indices = torch.nn.MaxPool2d(
                kernel_size=self.pool_size, return_indices=True
            )(x)
        elif self.pooling == 'avg':
            x = torch.nn.AvgPool2d(kernel_size=self.pool_size)(x)
            pool_indices = None
        else:
            raise ValueError(f'{self.pooling} pooling unknown.')
        return x, pool_indices


class Unpool2d(Module):
    def __init__(self, pool_size):
        super().__init__()
        self.pool_size = to_pair(pool_size)

    def forward(self, x, indices=None, pad_size=None):
        if all(np.array(self.pool_size) < 2):
            return x
        if indices is None:
            x = F.interpolate(x, scale_factor=self.pool_size)
        else:
            x = torch.nn.MaxUnpool2d(kernel_size=self.pool_size)(
                x, indices=indices
            )
        return x

