"""inspired by padertorch.contrib.je.modules.conv"""
from copy import copy

import numpy as np
import torch
import torch.nn.functional as F
from ham_radio.system.utils import Pool1d, Pool2d
from ham_radio.system.utils import to_pair, Pad, Cut
from padertorch.base import Module
from padertorch.ops.mappings import ACTIVATION_FN_MAP
from padertorch.utils import to_list
from torch import nn


class _Conv(Module):
    conv_cls = None

    @property
    def is_2d(self):
        return self.conv_cls in [nn.Conv2d]

    def __init__(
            self, in_channels, out_channels, kernel_size, dropout=0.,
            padding='both', pad_mode='constant', dilation=1, stride=1,
            bias=True, norm=None, activation='relu', pooling=None,
            pool_size=1, groups=1,
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
        self.pad_mode = pad_mode
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.stride = stride
        self.activation = ACTIVATION_FN_MAP[activation]()
        self.pooling = pooling
        self.pool_size = pool_size

        self.conv = self.conv_cls(
            in_channels, out_channels,
            kernel_size=kernel_size, dilation=dilation, stride=stride,
            bias=bias, groups=1
        )
        torch.nn.init.xavier_uniform_(self.conv.weight)
        if bias:
            torch.nn.init.zeros_(self.conv.bias)

        if norm is None:
            self.norm = None
        elif norm == 'batch':
            self.norm = nn.BatchNorm2d(out_channels) if self.is_2d \
                else nn.BatchNorm1d(out_channels)
        elif callable(norm):
            self.norm = norm
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
            x = Pad(side=self.padding, mode=self.pad_mode)(x, size=size)
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
        """
        >>> cnn = Conv1d(4, 20, 5, stride=2, pooling=False, padding=None)
        >>> signal = torch.rand((5, 4, 103))
        >>> out = cnn(signal)
        >>> out_shape = cnn.get_out_shape(out.shape[-1])
        >>> curr_size = out.shape[-1] - cnn.kernel_size + 1
        >>> curr_size = ((curr_size - 1) // cnn.stride) + 1
        >>> np.equal(out_shape, curr_size)[0]
        True

        Args:
            in_shape:

        Returns:

        """
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

    def get_in_shape(self, out_shape):
        """
        >>> cnn = Conv1d(4, 20, 5, stride=2)
        >>> signal = torch.rand((5, 4, 103))
        >>> out = cnn(signal)
        >>> in_shape = cnn.get_in_shape(out.shape[-1])
        >>> np.equal(in_shape, signal.shape[-1])[0]
        True
        >>> cnn = Conv1d(4, 20, 5, stride=2, padding=None)
        >>> signal = torch.rand((5, 4, 57))
        >>> out = cnn(signal)
        >>> in_shape = cnn.get_in_shape(out.shape[-1])
        >>> np.equal(in_shape, signal.shape[-1])[0]
        True
        >>> curr_size = (out.shape[-1] - 1) * cnn.stride + 1
        >>> curr_size = curr_size + cnn.kernel_size - 1
        >>> np.equal(in_shape, curr_size)[0]
        True

        Args:
            out_shape:

        Returns:

        """
        out_shape = np.array(out_shape - 1) * np.array(self.stride) + 1

        in_shape = out_shape + (
                np.array(self.dilation) * (np.array(self.kernel_size) - 1)
        )

        in_shape = np.where(
            [pad is None for pad in to_list(self.padding)],
            in_shape, out_shape
        )
        assert self.pooling is None, 'Not implemented yet'
        # if self.pooling is not None:
        #     in_shape = in_shape / np.array(self.pool_size)
        #     in_shape = np.floor(in_shape)
        #     in_shape = np.where(
        #         [pad is None for pad in to_list(self.padding)],
        #         in_shape, out_shape
        #     )
        #     in_shape = np.ceil(in_shape)
        return in_shape.astype(np.int64)


class Conv1d(_Conv):
    conv_cls = nn.Conv1d


class Conv2d(_Conv):
    conv_cls = nn.Conv2d


class _CNN(Module):
    conv_cls = None

    def __init__(
            self, in_channels, hidden_channels, out_channels, kernel_size,
            num_layers, dropout=0., padding='both', pad_mode='constant',
            dilation=1, stride=1, norm=None, activation='relu',
            pooling='max', pool_size=1
    ):
        super().__init__()

        self.in_channels = in_channels
        self.num_layers = num_layers
        num_hidden_layers = num_layers - int(out_channels is not None)
        assert num_hidden_layers > 0, num_hidden_layers
        self.hidden_channels = to_list(
            hidden_channels, num_hidden_layers
        )

        self.kernel_sizes = to_list(kernel_size, num_layers)
        self.paddings = to_list(padding, num_layers)
        self.pad_mode = to_list(pad_mode, num_layers)
        self.dilations = to_list(dilation, num_layers)
        self.strides = to_list(stride, num_layers)
        self.poolings = to_list(pooling, num_layers)
        self.pool_sizes = to_list(pool_size, num_layers)
        self.out_channels = out_channels if out_channels else \
        self.hidden_channels[
            -1]

        convs = list()
        for i in range(num_hidden_layers):
            hidden_channels = self.hidden_channels[i]
            convs.append(self.conv_cls(
                in_channels=in_channels, out_channels=hidden_channels,
                kernel_size=self.kernel_sizes[i], dilation=self.dilations[i],
                stride=self.strides[i], padding=self.paddings[i],
                pad_mode=self.pad_mode[i], norm=norm,
                dropout=dropout, activation=activation,
                pooling=self.poolings[i], pool_size=self.pool_sizes[i],
            ))
            in_channels = hidden_channels
        if out_channels is not None:
            convs.append(self.conv_cls(
                in_channels=in_channels, out_channels=out_channels,
                kernel_size=self.kernel_sizes[-1], dilation=self.dilations[-1],
                stride=self.strides[-1], padding=self.paddings[-1],
                pad_mode=self.pad_mode[-1], norm=None,
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

    def get_input_shape(self, out_shape):
        in_shape = out_shape
        for conv in self.convs:
            in_shape = conv.get_in_shape(in_shape)
        return in_shape


class CNN1d(_CNN):
    conv_cls = Conv1d


class CNN2d(_CNN):
    conv_cls = Conv2d
