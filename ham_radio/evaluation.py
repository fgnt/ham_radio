import os
from pathlib import Path

import ham_radio.keys as K
import numpy as np
import paderbox as pb
import padertorch as pt
import sacred
import torch
from tqdm import tqdm
from ham_radio.system.data import RadioProvider
from ham_radio.system.model import SADModel
from paderbox.array.interval import ArrayInterval
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from time import time


ex = sacred.Experiment('Test SAD')
sample_rate = 8000

@ex.config
def config():
    model_dir = None
    num_ths = 201
    dataset = 'eval'
    checkpoint = 'ckpt_best_loss.pth'
    segments = None
    buffer = 1
    out_dir = None
    num_jobs = os.cpu_count()


def adjust_annotation_fn(annotation, sample_rate, buffer_zone=1):
    '''

    Args:
        annotation:
        sample_rate:
        buffer_zone: num secs around speech activity which are not scored

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
        annotation, vad, sample_rate=8000, adjust_annotation=True
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
    fn = result[result == 1].shape[0]
    return tp, fp, tn, fn


@ex.capture
def evaluate(example, model, provider, segments, num_ths, buffer):
    audio = provider.read_audio(example)[K.AUDIO_DATA][K.OBSERVATION]
    # print(torch_example['speech_features'][0].shape)
    tic = time()
    torch_example = provider.transform.audio_to_input_dict(
        audio=audio,
        segments=segments, padder=provider.collate
    )
    with torch.no_grad():
        segmented_model_out, seq_len = model(torch_example)

    segmented_model_out = torch.max(
        segmented_model_out, dim=1)[0].detach().numpy()
    toc = (time() - tic) / example['num_samples'] * 8000
    annotation = ArrayInterval.from_str(*example[K.ALIGNMENT_ACTIVITY])[:]
    annotation = adjust_annotation_fn(
        annotation, sample_rate, buffer_zone=buffer)

    if isinstance(num_ths, list):
        ths = num_ths
        num_ths = len(num_ths)
    elif isinstance(num_ths, int):
        ths = np.linspace(0, 1, num_ths)
    else:
        raise ValueError

    tp_fp_tn_fn = np.zeros((num_ths, 4), dtype=np.int32)
    for idx, th in enumerate(ths):
        th = np.round(th, 4)
        model_out = model.get_per_frame_vad(
            segmented_model_out.copy(), th
        ).reshape(-1)
        vad = provider.transform.activity_frequency_to_time(
            model_out, provider.transform.stft.window_length,
            provider.transform.stft.shift
        )
        num_samples = min(annotation.shape[-1], vad.shape[-1])
        # num_samples = min(annotation.shape[-1], vad.shape[-1])
        tp_fp_tn_fn[idx] = np.array(get_tp_fp_tn_fn(
            annotation[:num_samples], vad[:num_samples],
            sample_rate=sample_rate, adjust_annotation=False
        ))
    return ths, tp_fp_tn_fn, vad, toc


@ex.automain
def main(model_dir, dataset, out_dir, checkpoint, num_jobs):
    model_dir = Path(model_dir).expanduser().resolve()
    model_cls = SADModel
    model = model_cls.from_config_and_checkpoint(
        config_path=model_dir / 'init.json',
        checkpoint_path=model_dir / 'checkpoints' / checkpoint,
        in_config_path='trainer_opts.model',
        in_checkpoint_path='model',
    )

    provider_opts = pb.io.load_json(model_dir / 'init.json')['provider_opts']
    provider = RadioProvider.from_config(provider_opts)

    tp_fp_tn_fn = list()
    iterable = provider.database.get_dataset(dataset)
    thresholds = list()
    timeing = list()
    with ThreadPoolExecutor(num_jobs) as ex:
        for ths, result, _, tic in tqdm(ex.map(
                partial(evaluate, model=model, provider=provider),
                iterable
        ), total=len(iterable)):
            thresholds.append(ths)
            tp_fp_tn_fn.append(result)
            timeing.append(tic)
    assert len(tp_fp_tn_fn) == len(iterable), (len(tp_fp_tn_fn), len(iterable))
    tp_fp_tn_fn = np.sum(tp_fp_tn_fn, axis=0)
    assert all([np.array_equal(thresholds[0], ths) for ths in thresholds]), thresholds
    thresholds = thresholds[0]
    if out_dir is None:
        out_dir = model_dir
    else:
        out_dir = Path(out_dir).expanduser().resolve()

    dset_name = '_'.join(pt.utils.to_list(dataset))
    out_dict = {'sad': {
        ths: value for ths, value in zip(thresholds, tp_fp_tn_fn.tolist())}}
    out_dict['timeing'] = np.mean(timeing)
    pb.io.dump_json(out_dict, out_dir / f'tp_fp_tn_fn_{dset_name}.json')
