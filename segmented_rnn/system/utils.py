"""inspired by padertorch.contrib.je.modules.conv"""
import math

import numpy as np
import torch
import torch.nn.functional as F
from padertorch.base import Module
from padertorch.configurable import Configurable
from padertorch.data.utils import pad_tensor, collate_fn
from padertorch.utils import to_list


def to_pair(x):
    return tuple(to_list(x, 2))


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


class Padder(Configurable):
    def __init__(
            self,
            to_torch: bool = True,
            sort_by_key: str = None,
            padding: bool = True,
            padding_keys: list = None
    ):
        """

        :param to_torch: if true converts numpy arrays to torch.Tensor
            if they are not strings or complex
        :param sort_by_key: sort the batch by a key from the batch
            packed_sequence needs sorted batch with decreasing sequence_length
        :param padding: if False only collates the batch,
            if True all numpy arrays with one variable dim size are padded
        :param padding_keys: list of keys, if no keys are specified all
            keys from the batch are used
        """
        assert not to_torch ^ (padding and to_torch)
        self.to_torch = to_torch
        self.padding = padding
        self.padding_keys = padding_keys
        self.sort_by_key = sort_by_key

    def pad_batch(self, batch):
        if isinstance(batch[0], np.ndarray):
            if batch[0].ndim > 0:
                dims = np.array(
                    [[idx for idx in array.shape] for array in batch]).T
                axis = [idx for idx, dim in enumerate(dims)
                        if not all(dim == dim[0])]

                assert len(axis) in [0, 1], (
                    f'only one axis is allowed to differ, '
                    f'axis={axis} and dims={dims}'
                )
                dtypes = [vec.dtype for vec in batch]
                assert dtypes.count(dtypes[-1]) == len(dtypes), dtypes
                if len(axis) == 1:
                    axis = axis[0]
                    pad = max(dims[axis])
                    array = np.stack([pad_tensor(vec, pad, axis)
                                      for vec in batch], axis=0)
                else:
                    array = np.stack(batch, axis=0)
                array = array.astype(dtypes[0])
                complex_dtypes = [np.complex64, np.complex128]
                if self.to_torch and not array.dtype.kind in {'U', 'S'} \
                        and not array.dtype in complex_dtypes:
                    return torch.from_numpy(array)
                else:
                    return array
            else:
                return np.array(batch)
        elif isinstance(batch[0], int):
            return np.array(batch)
        else:
            return batch

    def sort(self, batch):
        return sorted(batch, key=lambda x: x[self.sort_by_key], reverse=True)

    def __call__(self, unsorted_batch):
        # assumes batch to be a list of dicts
        # ToDo: do we automatically sort by sequence length?

        if self.sort_by_key:
            batch = self.sort(unsorted_batch)
        else:
            batch = unsorted_batch

        nested_batch = collate_fn(batch)

        if self.padding:
            if self.padding_keys is None:
                padding_keys = nested_batch.keys()
            else:
                assert len(self.padding_keys) > 0, \
                    'Empty padding key list was provided default is None'
                padding_keys = self.padding_keys

            def nested_padding(value, key):
                if isinstance(value, dict):
                    return {k: nested_padding(v, k) for k, v in value.items()}
                else:
                    if key in padding_keys:
                        return self.pad_batch(value)
                    else:
                        return value

            return {key: nested_padding(value, key) for key, value in
                    nested_batch.items()}
        else:
            assert self.padding_keys is None or len(self.padding_keys) == 0, (
                'Padding keys have to be None or empty if padding is set to '
                'False, but they are:', self.padding_keys
            )
            return nested_batch
