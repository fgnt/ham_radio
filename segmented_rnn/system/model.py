import math
from collections import defaultdict

import numpy as np
import padertorch as pt
import torch
import torch.nn.functional as F
from einops import rearrange
from paderbox.array import segment_axis
from padertorch.summary.tbx_utils import spectrogram_to_image, mask_to_image
from segmented_rnn import keys as K

from .module import CNN1d, CNN2d, Pool1d


class BinomialClassifier(pt.Model):
    """
    >>> cnn = BinomialClassifier(**{\
        'label_key': 'labels',\
        'cnn_2d': CNN2d(**{\
            'in_channels': 1,\
            'hidden_channels': 32,\
            'num_layers': 3,\
            'out_channels': 16,\
            'kernel_size': 3\
        }),\
        'cnn_1d': CNN1d(**{\
            'in_channels': 1024,\
            'hidden_channels': 32,\
            'num_layers': 3,\
            'out_channels': 10,\
            'kernel_size': 3\
        }),\
        'pooling': Pool1d('max', 10)\
    })
    >>> inputs = {\
        'speech_features': torch.zeros(4, 1, 100, 64),\
        'target_vad': torch.zeros(4, 1, 100),\
        'num_frames': [100]*4\
    }
    >>> outputs = cnn(inputs)
    >>> outputs[0].shape
    torch.Size([4, 10, 100])
    >>> review = cnn.review(inputs, outputs)

    >>> cnn_gru = BinomialClassifier(**{\
        'label_key': 'labels',\
        'cnn_2d': CNN2d(**{\
            'in_channels': 1,\
            'hidden_channels': 32,\
            'num_layers': 3,\
            'out_channels': 16,\
            'kernel_size': 3\
        }),\
        'cnn_1d': CNN1d(**{\
            'in_channels': 1024,\
            'hidden_channels': 32,\
            'num_layers': 3,\
            'out_channels': 10,\
            'kernel_size': 3\
        }),\
        'rnn': torch.nn.GRU(**{\
                'input_size':1024,\
                'hidden_size':256,\
                'num_layers':2,\
                'dropout':0.5,\
                'bidirectional':True\
        }),\
        'pooling': Pool1d('max', 10),\
        'smooth_with_rnn': True\
    })
    >>> inputs = {\
        'speech_features': torch.zeros(3, 1, 400, 64),\
        'target_vad': torch.zeros(3, 1, 400),\
        'num_frames': [500]*3\
    }
    >>> outputs = cnn_gru(inputs)
    >>> outputs[0].shape
    torch.Size([3, 10, 16])
    """

    @classmethod
    def finalize_dogmatic_config(cls, config):
        if 'rnn' in config.keys() and config['rnn'] is not None:
            if 'cnn_1d' in config.keys():
                assert config['cnn_1d'] is None, config
            # we assume that the parameter out_channels was not used in cnn_2d
            config['rnn'] = dict(
                factory=torch.nn.GRU,
                input_size=config['cnn_2d']['hidden_channels'][-1],
                hidden_size=256,
                num_layers=2,
                dropout=0.5,
                bidirectional=True,
                batch_first=False
            )
        return config

    def __init__(
            self,
            cnn_2d: CNN2d,
            cnn_1d: CNN1d = None,
            rnn: torch.nn.GRU = None,
            pooling=None,
            *,
            input_norm='l2_norm',
            recall_weight=1.,
            activation='sigmoid',
            smooth_with_rnn=False,
            window_length=50,
            window_shift=25
    ):
        super().__init__()
        self._cnn_2d = cnn_2d
        self._cnn_1d = cnn_1d
        self._rnn = rnn
        if rnn:
            self.rnn_dnn = pt.modules.fully_connected_stack(
                input_size=rnn.hidden_size * 2,
                hidden_size=None,
                output_size=cnn_2d.out_channels,
                activation='relu',
                dropout=0.5
            )
        self.pooling = pooling
        self.recall_weight = recall_weight
        self.activiation = pt.mappings.ACTIVATION_FN_MAP[activation]()
        self.norm = input_norm
        self.smooth_with_rnn = smooth_with_rnn
        self.window_length = window_length
        self.window_shift = window_shift

    def cnn_2d(self, x, seq_len=None):
        if self._cnn_2d is not None:
            x = self._cnn_2d(x)
            if seq_len is not None:
                in_shape = [(128, n) for n in seq_len]
                out_shape = self._cnn_2d.get_out_shape(in_shape)
                seq_len = [s[-1] for s in out_shape]

        if x.dim() != 3:
            assert x.dim() == 4
            x = rearrange(x, 'b c f t -> b (c f) t')
        return x, seq_len

    def cnn_1d(self, x, seq_len=None):
        if self._cnn_1d is not None:
            x = self._cnn_1d(x)
            if seq_len is not None:
                seq_len = self._cnn_1d.get_out_shape(seq_len)
        return x, seq_len

    def rnn(self, x, seq_len=None):
        if self._rnn is not None:
            if self.smooth_with_rnn:
                batch_size = x.shape[0]
                x_list = self.segment(x)
                x_list = rearrange(x_list, 'b f l t -> (b l) t f')
                x, _ = self._rnn(x_list)
                x = rearrange(x[..., -1, :], '(b l) f -> b l f', b=batch_size)
                if seq_len is not None:
                    seq_len = [math.ceil(seq / self.window_shift)
                               for seq in seq_len]
            else:
                x = x.permute(0, 2, 1)
                x, _ = self._rnn(x)
            x = self.rnn_dnn(x)
            x = x.permute(0, 2, 1)
        return x, seq_len

    def segment(self, x):
        size = self.window_length - 1 - ((x.shape[-1] - 1) % self.window_shift)
        pad = []
        pad.extend([size // 2, math.ceil(size / 2)])
        x = F.pad(x, tuple(pad), mode='constant')
        return segment_axis(
            x, self.window_length, self.window_shift, axis=-1, end=None)

    def forward(self, inputs):
        x = inputs[K.SPEECH_FEATURES]
        if self.norm is not None:
            if self.norm == 'l2_norm':
                x -= torch.mean(x, dim=-2, keepdim=True)
                x /= (torch.std(x, dim=-2, keepdim=True) + 1e-10)
            else:
                raise NotImplementedError('Only the l2_norm is implemented at'
                                          'the moment', self.norm)
        seq_len = inputs['num_frames']
        x = x.permute(0, 1, 3, 2)
        x, seq_len = self.cnn_2d(x, seq_len)

        y, seq_len = self.rnn(x, seq_len)

        z, seq_len = self.cnn_1d(y, seq_len)

        return 1e-3 + (1. - 2e-3) * self.activiation(z), seq_len

    def maybe_pool(self, scores):
        if self.pooling is not None:
            scores, _ = self.pooling(scores.permute(0, 2, 1))
            scores = scores[..., 0]
        else:
            scores = scores
        return scores

    def review(self, inputs, outputs):
        # compute loss
        loss_list = list()
        score_list = list()
        target_list = list()
        scalars = defaultdict(lambda: [0, 0, 0, 0])
        for idx, target in enumerate(inputs[K.TARGET_VAD]):
            scores = outputs[0][idx][None]
            seq_len = outputs[1][idx]
            scores = self.maybe_pool(scores)
            scores = scores[..., :seq_len]
            if self.smooth_with_rnn:
                target = self.segment(target)
                target = torch.max(target, dim=-1)[0]

            targets = target[..., :scores.shape[-1]]
            scores = scores[..., :targets.shape[-1]]
            if self.weakly_supervised:
                targets = self.auto_pooling(targets)
            assert targets.dim() == scores.dim(), (targets.shape, scores.shape)
            loss = -(
                    self.recall_weight * targets * torch.log(scores)
                    + (1. - targets) * torch.log(1. - scores)
            ).sum()
            # loss = nn.BCELoss(reduction='none')(scores, targets).sum(-1)
            loss_list.append(loss)
            target_list.append(targets.sum())
            score_list.append(scores.sum())
            for thres in [0.3, 0.5]:
                decision = (scores.detach() > thres).float()
                scalars[thres][0] += (decision * targets).sum()
                scalars[thres][1] += (decision * (1. - targets)).sum()
                scalars[thres][2] += ((1. - decision) * (1. - targets)).sum()
                scalars[thres][3] += ((1. - decision) * targets).sum()

        # create review including metrics and visualizations
        scalar_names = [
            'true_pos_{thres}',
            'false_pos_{thres}',
            'true_neg_{thres}',
            'false_neg_{thres}'
        ]
        review = dict(
            loss=sum(loss_list),
            scalars={scalar_names[idx].format(thres=thres): scalar
                     for thres, value in scalars.items() for idx, scalar in
                     enumerate(value)
                     },
            histograms=dict(),
            images=self.add_images(inputs, targets, scores)
        )
        return review

    def add_images(self, inputs, targets, scores):
        features = inputs[K.SPEECH_FEATURES][0][:3]
        targets = targets[..., None].repeat(1, 1, 30)
        scores = scores[..., None].repeat(1, 1, 30)
        return dict(
            features=spectrogram_to_image(features.permute(1, 0, 2)),
            targets=mask_to_image(targets.permute(1, 0, 2)),
            scores=mask_to_image(scores.permute(1, 0, 2)),
        )

    def modify_summary(self, summary):
        summary = super().modify_summary(summary)
        # compute precision, recall and fscore for each decision threshold
        scalar_names = [
            'true_pos_{thres}',
            'false_pos_{thres}',
            'true_neg_{thres}',
            'false_neg_{thres}'
        ]
        for thres in [0.3, 0.5]:
            if all([
                key in summary['scalars']
                for key in [name.format(thres=thres) for name in scalar_names]
            ]):
                tp, fp, tn, fn = [
                    np.sum(summary['scalars'][name.format(thres=thres)])
                    for name in scalar_names
                ]
                p = tp / (tp + fp)
                r = tp / (tp + fn)
                pfn = fn / (fn + tp)
                pfp = fp / (fp + tn)
                summary['scalars'][f'precision_{thres}'] = p
                summary['scalars'][f'recall_{thres}'] = r
                summary['scalars'][f'f1_{thres}'] = 2 * (p * r) / (p + r)
                summary['scalars'][f'dcf_{thres}'] = 0.75 * pfn + 0.25 * pfp
        return summary

    def get_per_frame_vad(self, model_out, threshold,
                          transform=None, segment_length=400):
        if self.smooth_with_rnn:
            shift = self.window_shift
            slc = self.window_length - 1 - ((segment_length - 1) % shift)
            num_segments = math.ceil(segment_length / shift)
            model_out_np = model_out.reshape(-1, num_segments)
            model_out_np[model_out_np > threshold] = 1
            model_out_np[model_out_np < 1] = 0
            model_out = transform.activity_frequency_to_time(
                model_out_np, self.window_length, shift)
            model_out = model_out[..., slc // 2: -int(np.ceil(slc / 2))]
            assert model_out.shape[-1] == segment_length, model_out.shape
            return model_out
        else:
            batch_size = model_out.shape[0]
            return smooth_vad(model_out.reshape(batch_size, -1).copy(),
                              threshold=threshold)

    def auto_pooling(self, prob):
        exp = torch.exp(self.temperature * prob)
        exp_sum = exp / torch.sum(exp, dim=-1, keepdim=True)
        return torch.sum(prob * exp_sum, dim=-1, keepdim=True)


def smooth_vad(vad_pred, window=25, divisor=1, threshold=0.1):
    """
    >>> vad_pred = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.2, 0.1])
    >>> smooth_vad(vad_pred, window=3, divisor=1, threshold=0.3)
    array([0., 0., 1., 1., 1., 1., 1., 1., 0.])
    >>> smooth_vad(vad_pred, window=5, divisor=1, threshold=0.5)
    array([0., 0., 0., 0., 1., 1., 1., 1., 0.])
    >>> smooth_vad(vad_pred, window=5, divisor=2, threshold=0.5)
    array([0., 0., 0., 1., 1., 1., 1., 1., 1.])
    >>> smooth_vad(vad_pred[None, None], window=5, divisor=2, threshold=0.5)
    array([[[0., 0., 0., 1., 1., 1., 1., 1., 1.]]])
    """
    vad_pred = vad_pred.copy()
    vad_pred[vad_pred > threshold] = 1.
    vad_pred[vad_pred < 1] = 0.
    shift = window // 2
    padding = [(0, 0)] * (vad_pred.ndim - 1) + [(shift, shift)]
    vad_padded = np.pad(vad_pred, padding, 'edge')
    vad_segmented = segment_axis(vad_padded, window, 1, end='pad')
    vad_segmented = np.sum(vad_segmented, axis=-1)
    vad_pred[vad_segmented >= shift // divisor] = 1
    vad_pred[vad_segmented < shift // divisor] = 0
    return vad_pred
