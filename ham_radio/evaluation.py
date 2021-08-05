from pathlib import Path

import dlp_mpi
import ham_radio.keys as K
import numpy as np
import paderbox as pb
import padertorch as pt
import sacred
import torch
from ham_radio.system.data import RadioProvider
from ham_radio.system.model import SADModel
from paderbox.array.interval import ArrayInterval

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


@ex.automain
def main(model_dir, num_ths, dataset, buffer, out_dir, checkpoint, segments):
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
    transform = provider.transform
    padder = provider.collate

    tp_fp_tn_fn = np.zeros((num_ths, 4), dtype=int)
    for example in dlp_mpi.split_managed(
            provider.database.get_dataset(dataset),
            is_indexable=True,
            allow_single_worker=True,
            progress_bar=True
    ):
        torch_example = transform.audio_to_input_dict(
            audio=provider.read_audio(example)[K.AUDIO_DATA][K.OBSERVATION],
            segments=segments, padder=padder
        )
        segmented_model_out = model(torch_example)
        segmented_model_out = torch.max(
            segmented_model_out[0], dim=1)[0].detach().numpy()

        annotation = ArrayInterval.from_str(*example[K.ALIGNMENT_ACTIVITY])[:]
        annotation = adjust_annotation_fn(
            annotation, sample_rate, buffer_zone=buffer)
        for idx, th in enumerate(np.linspace(0, 1, num_ths)):
            th = np.round(th, 2)
            model_out = model.get_per_frame_vad(
                segmented_model_out.copy(), th
            ).reshape(-1)
            vad = provider.transform.activity_frequency_to_time(
                model_out, transform.stft.window_length, transform.stft.shift)
            num_samples = min(annotation.shape[-1], vad.shape[-1])
            out = get_tp_fp_tn_fn(
                annotation[:num_samples], vad[:num_samples],
                sample_rate=sample_rate, adjust_annotation=False
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

        dset_name = '_' + '_'.join(pt.utils.to_list(dataset))
        (out_dir / f'tp_fp_tn_fn_{dset_name}.txt').write_text('\n'.join([
            ' '.join([str(v) for v in value]) for value in tp_fp_tn_fn.tolist()
        ]))
