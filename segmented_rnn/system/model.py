from collections import defaultdict

import numpy as np
import padertorch as pt
import torch
from einops import rearrange
from padertorch.summary.tbx_utils import spectrogram_to_image, mask_to_image
from padertorch.contrib.jensheit.eval_sad import smooth_vad
from ham_radio import keys as K
from ham_radio.system.module import CNN1d, CNN2d, Pool1d


class BinomialClassifier(pt.Model):
    """
    >>> cnn = BinomialClassifier(**{\
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
     >>> rnn = BinomialClassifier(**{\
        'cnn_2d': CNN2d(**{\
            'in_channels': 1,\
            'hidden_channels': 32,\
            'num_layers': 3,\
            'out_channels': 8,\
            'kernel_size': 3\
        }),\
        'rnn': torch.nn.GRU(**{\
                'input_size':512,\
                'hidden_size':256,\
                'num_layers':2,\
                'dropout':0.5,\
                'bidirectional':True,\
                'batch_first':False\
        }),\
        'pooling': Pool1d('max', 10),\
        'cnn_1d':None\
    })
    >>> inputs = {\
        'speech_features': torch.zeros(3, 1, 400, 64),\
        'target_vad': torch.zeros(3, 1, 400),\
        'num_frames': [500]*3\
    }
    >>> outputs = rnn(inputs)
    >>> outputs[0].shape
    torch.Size([3, 10, 400])
    """

    def __init__(
            self,
            cnn_2d: CNN2d,
            pooling: Pool1d,
            cnn_1d: CNN1d = None,
            rnn: torch.nn.GRU = None,
            *,
            input_norm='l2_norm',
            recall_weight=1.,
            activation='sigmoid',
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
                output_size=pooling.pool_size,
                activation='relu',
                dropout=0.5
            )
        self.pooling = pooling
        self.recall_weight = recall_weight
        self.activiation = pt.mappings.ACTIVATION_FN_MAP[activation]()
        self.norm = input_norm
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
            x = x.permute(0, 2, 1)
            x, _ = self._rnn(x)
            x = self.rnn_dnn(x)
            x = x.permute(0, 2, 1)
        return x, seq_len


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
        x = rearrange(x, 'b c t f -> b c f t')
        x, seq_len = self.cnn_2d(x, seq_len)

        y, seq_len = self.rnn(x, seq_len)

        z, seq_len = self.cnn_1d(y, seq_len)

        return 1e-3 + (1. - 2e-3) * self.activiation(z), seq_len

    def maybe_pool(self, scores):
        if self.pooling is not None:
            scores, _ = torch.max(
                scores.permute(0, 2, 1), keepdim=True, dim=-1)
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
            targets = target[..., :scores.shape[-1]]
            scores = scores[..., :targets.shape[-1]]
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
            batch_size = model_out.shape[0]
            return smooth_vad(model_out.reshape(batch_size, -1).copy(),
                              threshold=threshold)