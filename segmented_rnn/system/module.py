"""inspired by padertorch.contrib.je.modules.conv"""
from copy import copy

import numpy as np
import torch
import torch.nn.functional as F
from padertorch.base import Module
from padertorch.ops.mappings import ACTIVATION_FN_MAP
from padertorch.utils import to_list
from torch import nn

from segmented_rnn.system.utils import Pool1d, Pool2d
from segmented_rnn.system.utils import to_pair, Pad, Cut


class _Conv(Module):
    conv_cls = None

    @property
    def is_2d(self):
        return self.conv_cls in [nn.Conv2d]

    def __init__(
            self, in_channels, out_channels, kernel_size, dropout=0.,
            padding='both', dilation=1, stride=1, bias=True, norm=None,
            activation='relu', pooling=None, pool_size=1
    ):
        """

        Args:
            in_channels:
            out_channels:
            kernel_size:
            dilation:
            stride:
            bias:
            dropout:
            norm: may be None or 'batch'
            activation:
            pooling:
            pool_size:
        """
        super().__init__()
        if self.is_2d:
            padding = to_pair(padding)
            kernel_size = to_pair(kernel_size)
            dilation = to_pair(dilation)
            stride = to_pair(stride)
        self.dropout = dropout
        self.padding = padding
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.stride = stride
        self.activation = ACTIVATION_FN_MAP[activation]()
        self.pooling = pooling
        self.pool_size = pool_size

        self.conv = self.conv_cls(
            in_channels, out_channels,
            kernel_size=kernel_size, dilation=dilation, stride=stride,
            bias=bias
        )
        torch.nn.init.xavier_uniform_(self.conv.weight)
        if bias:
            torch.nn.init.zeros_(self.conv.bias)

        if norm is None:
            self.norm = None
        elif norm == 'batch':
            self.norm = nn.BatchNorm2d(out_channels) if self.is_2d \
                else nn.BatchNorm1d(out_channels)
        else:
            raise ValueError(f'{norm} normalization not known.')

    def forward(self, x, pool_indices=None, out_shape=None):
        in_shape = x.shape[2:]
        x = self.unpool(x, pool_indices)
        if self.training and self.dropout > 0.:
            x = F.dropout(x, self.dropout)

        x = self.pad(x)

        y = self.conv(x)

        if self.norm is not None:
            y = self.norm(y)

        y = self.activation(y)

        y = self.unpad(y, out_shape)

        y, pool_indices = self.pool(y)

        return y

    def pool(self, x):
        if self.pooling is None or self.pool_size == 1:
            return x, None

        if self.is_2d:
            pool = Pool2d(
                pooling=self.pooling,
                pool_size=self.pool_size,
                padding=self.padding
            )
        else:
            pool = Pool1d(
                pooling=self.pooling,
                pool_size=self.pool_size,
                padding=self.padding
            )
        x, pool_indices = pool(x)
        return x, pool_indices

    def unpool(self, x, pool_indices=None):
        assert pool_indices is None, (
            self.pooling, self.pool_size, pool_indices is None)
        return x

    def pad(self, x):
        padding = [pad is not None for pad in to_list(self.padding)]
        if any(padding):
            size = (
                    np.array(self.dilation) * (np.array(self.kernel_size) - 1)
                    - ((np.array(x.shape[2:]) - 1) % np.array(self.stride))
            ).tolist()
            x = Pad(side=self.padding)(x, size=size)
        if not all(padding):
            size = (
                    (np.array(x.shape[2:]) - np.array(self.kernel_size))
                    % np.array(self.stride)
            ).tolist()
            x = Cut(
                side=('both' if not pad else None for pad in padding))(x, size)
        return x

    def unpad(self, y, out_shape=None):
        if out_shape is not None:
            assert self.is_transpose
            size = np.array(y.shape[2:]) - np.array(out_shape)
            padding = [
                'both' if side is None else side
                for side in to_list(self.padding)
            ]
            if any(size > 0):
                y = Cut(side=padding)(y, size=size)
            if any(size < 0):
                y = Pad(side=padding, mode='constant')(y, size=-size)
        return y

    def get_out_shape(self, in_shape):
        out_shape = np.array(in_shape)
        out_shape_ = out_shape - (
                np.array(self.dilation) * (np.array(self.kernel_size) - 1)
        )
        out_shape = np.where(
            [pad is None for pad in to_list(self.padding)],
            out_shape_, out_shape
        )
        out_shape = np.ceil(out_shape / np.array(self.stride))
        if self.pooling is not None:
            out_shape = out_shape / np.array(self.pool_size)
            out_shape_ = np.floor(out_shape)
            out_shape = np.where(
                [pad is None for pad in to_list(self.padding)],
                out_shape_, out_shape
            )
            out_shape = np.ceil(out_shape)
        return out_shape.astype(np.int64)


class Conv1d(_Conv):
    conv_cls = nn.Conv1d


class Conv2d(_Conv):
    conv_cls = nn.Conv2d


class _CNN(Module):
    conv_cls = None

    def __init__(
            self, in_channels, hidden_channels, out_channels, kernel_size,
            num_layers, dropout=0., padding='both', dilation=1, stride=1,
            norm=None, activation='relu', pooling='max', pool_size=1
    ):
        super().__init__()

        self.in_channels = in_channels
        self.num_layers = num_layers
        num_hidden_layers = num_layers - int(out_channels is not None)
        self.hidden_channels = to_list(
            hidden_channels, num_hidden_layers
        )
        self.kernel_sizes = to_list(kernel_size, num_layers)
        self.paddings = to_list(padding, num_layers)
        self.dilations = to_list(dilation, num_layers)
        self.strides = to_list(stride, num_layers)
        self.poolings = to_list(pooling, num_layers)
        self.pool_sizes = to_list(pool_size, num_layers)
        self.out_channels = out_channels if out_channels else hidden_channels[
            -1]

        convs = list()
        for i in range(num_hidden_layers):
            hidden_channels = self.hidden_channels[i]
            convs.append(self.conv_cls(
                in_channels=in_channels, out_channels=hidden_channels,
                kernel_size=self.kernel_sizes[i], dilation=self.dilations[i],
                stride=self.strides[i], padding=self.paddings[i], norm=norm,
                dropout=dropout, activation=activation,
                pooling=self.poolings[i], pool_size=self.pool_sizes[i],
            ))
            in_channels = hidden_channels
        if out_channels is not None:
            convs.append(self.conv_cls(
                in_channels=in_channels, out_channels=out_channels,
                kernel_size=self.kernel_sizes[-1], dilation=self.dilations[-1],
                stride=self.strides[-1], padding=self.paddings[-1], norm=None,
                dropout=dropout, activation='identity',
                pooling=self.poolings[-1], pool_size=self.pool_sizes[-1],
            ))

        self.convs = nn.ModuleList(convs)

    def forward(self, x, pool_indices=None, out_shapes=None):
        pool_indices = to_list(copy(pool_indices), self.num_layers)[::-1]
        shapes = to_list(copy(out_shapes), self.num_layers)[::-1]
        for i, conv in enumerate(self.convs):
            x = conv(x, pool_indices[i], shapes[i])
            if isinstance(x, tuple):
                x, pool_indices[i], shapes[i] = x
        return x

    def get_out_shape(self, in_shape):
        out_shape = in_shape
        for conv in self.convs:
            out_shape = conv.get_out_shape(out_shape)
        return out_shape


class CNN1d(_CNN):
    conv_cls = Conv1d


class CNN2d(_CNN):
    conv_cls = Conv2d
