import dlp_mpi
import sacred
import torch
from paderbox.array import segment_axis
import numpy as np
from pathlib import Path
from paderbox.io import load_json

import padertorch as pt
from segmented_rnn.system.provider import RadioProvider, Transformer
from segmented_rnn.system.model import BinomialClassifier

ex = sacred.Experiment('Test Plath')
sample_rate = 8000

@ex.config
def config():
    model_dir = None
    num_ths = 201
    dataset = 'test'
    checkpoint='ckpt_best_loss.pth'
    segments=400
    buffer = 1
    out_dir = None


def adjust_annotation_fn(annotation, sample_rate, buffer_zone=1):
    '''

    Args:
        annotation:
        sample_rate:
        buffer_zone:

    Returns:
    >>> annotation = np.array([0, 1, 1, 1, 0, 0, 0, 1])
    >>> adjust_annotation_fn(annotation, 1)
    array([5, 1, 1, 1, 5, 0, 5, 1], dtype=int32)
    >>> annotation = np.array([0, 1, 1, 1, 0, 0, 0, 1])
    >>> adjust_annotation_fn(annotation, 2)
    array([5, 1, 1, 1, 5, 5, 5, 1], dtype=int32)
    '''
    buffer_zone = int(buffer_zone * sample_rate)
    indices = np.where(annotation[:-1] != annotation[1:])[0]
    if len(indices) == 0:
        return annotation
    elif len(indices) % 2 != 0:
        indices = np.concatenate([indices, [len(annotation)]], axis=0)
    start_end = np.split(indices, len(indices) // 2)
    annotation = annotation.astype(np.int32)
    for start, end in start_end:
        start += 1
        end += 1
        start_slice = slice(start - buffer_zone, start, 1)
        annotation[start_slice][annotation[start_slice] != 1] = 5
        end_slice = slice(end, end + buffer_zone, 1)
        annotation[end_slice][annotation[end_slice] != 1] = 5
    return annotation


def get_tp_fp_tn_fn(
        annotation, vad, sample_rate=8000, adjust_annotation=True,
        ignore_buffer=False
):
    """
    >>> annotation = np.array([0, 1, 1, 1, 0, 0, 0, 1])
    >>> vad = np.array([0, 1, 1, 1, 0, 0, 0, 1])
    >>> get_tp_fp_tn_fn(annotation, vad, 1)
    (4, 0, 4, 0)
    >>> annotation = np.array([0, 1, 1, 1, 0, 0, 0, 1])
    >>> vad = np.array([1, 1, 1, 1, 0, 0, 0, 1])
    >>> get_tp_fp_tn_fn(annotation, vad, 1)
    (4, 0, 3, 0)
    >>> annotation = np.array([0, 1, 1, 1, 0, 0, 0, 1])
    >>> vad = np.array([0, 1, 1, 1, 0, 1, 0, 1])
    >>> get_tp_fp_tn_fn(annotation, vad, 1)
    (4, 1, 3, 0)
    >>> annotation = np.array([0, 1, 1, 1, 0, 0, 0, 1])
    >>> vad = np.array([0, 1, 1, 1, 0, 0, 0, 0])
    >>> get_tp_fp_tn_fn(annotation, vad, 1)
    (3, 0, 4, 1)
    >>> annotation = np.array([0, 1, 1, 1, 0, 0, 0, 1])
    >>> vad = np.array([1, 1, 1, 1, 1, 0, 1, 1])
    >>> get_tp_fp_tn_fn(annotation, vad, 1)
    (4, 0, 1, 0)
    >>> annotation = np.array([0, 1, 1, 1, 0, 0, 0, 1])
    >>> vad = np.array([0, 1, 1, 1, 0, 1, 0, 0])
    >>> get_tp_fp_tn_fn(annotation, vad, 12)
    (3, 0, 3, 1)
    >>> rng = np.random.RandomState(seed=3)
    >>> annotation = rng.randint(0, 2, size=(32000))
    >>> vad = rng.randint(0, 2, size=(32000))
    >>> get_tp_fp_tn_fn(annotation, vad)
    (8090, 2, 7937, 7978)

    :param annotation:
    :param vad:
    :param sample_rate:
    :param adjust_annotation:
    :return:
    """
    assert len(annotation) == len(vad), (len(annotation), len(vad))
    assert annotation.ndim == 1, annotation.shape
    assert vad.ndim == 1, vad.shape
    if adjust_annotation:
        annotation = adjust_annotation_fn(annotation, sample_rate)

    vad = np.round(vad).astype(np.int32) * 10
    result = vad + annotation.astype(np.int32)
    tp = result[result == 11].shape[0]
    fp = result[result == 10].shape[0]
    tn = result[result == 0].shape[0]
    if not ignore_buffer:
        tn += result[result == 5].shape[0]
    fn = result[result == 1].shape[0]
    return tp, fp, tn, fn


@ex.automain
def main(model_dir, num_ths, dataset,
         buffer, out_dir, checkpoint, segments):
    model_dir = Path(model_dir).expanduser().resolve()
    model_cls = BinomialClassifier
    model = model_cls.from_config_and_checkpoint(
        config_path=model_dir / 'init.json',
        checkpoint_path=model_dir / 'checkpoints' / checkpoint,
        in_config_path='trainer_opts.model',
        in_checkpoint_path='model',
    )

    provider_opts = load_json(model_dir / 'init.json')['provider_opts']
    provider_opts['database'][
        'factory'] = 'segmented_rnn.database.HamRadioLibrispeech'
    provider_opts['transform']['factory'] = Transformer
    provider = RadioProvider.from_config(provider_opts)
    provider.batch_size_eval = 1
    ds = provider.get_predict_iterator(dataset=dataset)
    model.eval()

    tp_fp_tn_fn = np.zeros((num_ths, 4), dtype=int)
    for ex_eval in dlp_mpi.split_managed(
            ds,
            is_indexable=True,
            allow_single_worker=True,
            progress_bar=True
    ):
        ex_eval['speech_features'] = segment_axis(
            ex_eval['speech_features'], segments, shift=segments, axis=-2
        ).reshape(-1, 1, 400, ex_eval['speech_features'].shape[-1])

        ex_eval_torch = pt.data.example_to_device(ex_eval)
        model_out_freq = model(ex_eval_torch)
        model_out_freq = torch.max(
            model_out_freq[0], dim=1)[0].detach().numpy()

        ex_vad = ex_eval['target_vad'][0][0]
        ali = provider.transform.activity_frequency_to_time(
            ex_vad, 400, 160)
        ali = adjust_annotation_fn(ali, sample_rate, buffer_zone=buffer)
        for idx, th in enumerate(np.linspace(0, 1, num_ths)):
            th = np.round(th, 2)
            model_out = model.get_per_frame_vad(
                model_out_freq.copy(), th, provider.transform,
                segment_length=segments
            ).reshape(-1)
            vad = provider.transform.activity_frequency_to_time(
                model_out, 400, 160)
            num_samples = min(ali.shape[-1], vad.shape[-1])
            out = get_tp_fp_tn_fn(
                ali[:num_samples], vad[:num_samples], sample_rate=8000,
                adjust_annotation=False
            )
            tp_fp_tn_fn[idx] = [tp_fp_tn_fn[idx][idy] + o for idy, o in
                                enumerate(out)]


    dlp_mpi.barrier()
    tp_fp_tn_fn_gather = dlp_mpi.gather(tp_fp_tn_fn, root=dlp_mpi.MASTER)
    if dlp_mpi.IS_MASTER:
        if out_dir is None:
            out_dir = model_dir
        else:
            out_dir = Path(out_dir).expanduser().resolve()
        tp_fp_tn_fn = np.zeros((num_ths, 4), dtype=int)
        for array in tp_fp_tn_fn_gather:
            tp_fp_tn_fn += array

        if not dataset == 'test':
            sub = '_' + '_'.join(pt.utils.to_list(dataset))
        else:
            sub = ''
        (out_dir / f'tp_fp_tn_fn_{sub}.txt').write_text('\n'.join([
            ' '.join([str(v) for v in value]) for value in tp_fp_tn_fn.tolist()
        ]))
