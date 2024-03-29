import collections
from copy import deepcopy
from functools import partial
from hashlib import md5
from random import shuffle as shuffle_fn

import numpy as np
import paderbox as pb
import padertorch as pt
from ham_radio import keys as K
from ham_radio.system.utils import Padder
from paderbox.array import segment_axis
from paderbox.array.interval import ArrayInterval
from paderbox.transform import STFT
from padertorch.base import Configurable
from lazy_dataset.core import FilterException


class Transformer(Configurable):

    @classmethod
    def finalize_dogmatic_config(cls, config):
        super().finalize_dogmatic_config(config)
        config['stft'] = dict(factory=STFT, size=256, shift=80,
                              window_length=200, fading=False)

    def __init__(self, stft, log=False):
        self.stft = stft
        self.log = log

    def inverse(self, signal):
        return self.stft.inverse(signal)

    def __call__(self, example):
        if isinstance(example, (list, tuple, collections.Generator)):
            return [self.transform(ex) for ex in example]
        else:
            return self.transform(example)

    def transform(self, example):
        example.update({key: self.maybe_add_channel(signal) for key, signal in
                        example.items() if isinstance(signal, np.ndarray)})
        num_samp = example[K.NUM_SAMPLES]
        obs = example[K.OBSERVATION][..., :num_samp]
        example[K.SPEECH_FEATURES] = self.transform_obs(obs)
        example[K.NUM_FRAMES] = example[K.SPEECH_FEATURES].shape[-2]
        if K.TARGET_TIME_VAD in example:
            example[K.TARGET_VAD] = self.activity_time_to_frequency(
                example[K.TARGET_TIME_VAD],
                self.stft.window_length, self.stft.shift, self.stft.pad
            )
            example[K.TARGET_TIME_VAD] = example[K.TARGET_TIME_VAD].astype(
                np.float32)
            example[K.TARGET_VAD] = example[K.TARGET_VAD].astype(np.float32)
        return example

    def transform_obs(self, signal):
        signal_stft = self.stft(signal)
        feature = np.abs(signal_stft)

        if self.log:
            feature = np.where(feature == 0, np.finfo(float).eps, feature)
            feature = np.log(feature)
        return feature.astype(np.float32)

    def audio_to_input_dict(self, audio, padder=None, segments=400):
        audio_dict = {
            K.OBSERVATION: audio,
            K.NUM_SAMPLES: audio.shape[-1]
        }
        input_dict = self.__call__(audio_dict)
        feature_size = input_dict[K.SPEECH_FEATURES].shape[-1]
        channels = input_dict[K.SPEECH_FEATURES].shape[0]
        if not isinstance(audio, (tuple, list)):
            input_dict = [input_dict]
        if padder is not None:
            input_dict = padder(input_dict)
        if segments is not None:
            input_dict[K.SPEECH_FEATURES] = segment_axis(
                input_dict[K.SPEECH_FEATURES], segments, shift=segments,
                axis=-2
            ).reshape(-1, channels, segments, feature_size)
        return pt.data.example_to_device(input_dict)

    @staticmethod
    def get_spectrogram(stft_signal):
        return stft_signal.real ** 2 + stft_signal.imag ** 2

    @staticmethod
    def maybe_add_channel(signal):
        if signal.ndim == 1:
            return np.expand_dims(signal, axis=0)
        elif signal.ndim == 2:
            return signal
        else:
            raise ValueError(signal.ndim, signal.shape)

    @staticmethod
    def activity_time_to_frequency(
            time_activity,
            stft_window_length,
            stft_shift,
            stft_pad
    ):
        assert np.asarray(time_activity).dtype != np.object, (
            type(time_activity), np.asarray(time_activity).dtype)
        time_activity = np.asarray(time_activity)

        return segment_axis(
            time_activity,
            length=stft_window_length,
            shift=stft_shift,
            end='pad' if stft_pad else 'cut'
        ).any(axis=-1)

    @staticmethod
    def activity_frequency_to_time(
            frequency_activity,
            stft_window_length,
            stft_shift,
    ):

        frequency_activity = np.asarray(frequency_activity)
        frequency_activity = np.broadcast_to(
            frequency_activity[..., None],
            (*frequency_activity.shape, stft_window_length)
        )

        time_activity = np.zeros(
            (*frequency_activity.shape[:-2],
             frequency_activity.shape[
                 -2] * stft_shift + stft_window_length - stft_shift)
        )

        # Get the correct view to time_signal
        time_signal_seg = segment_axis(
            time_activity, stft_window_length, stft_shift, end=None
        )
        time_signal_seg[frequency_activity > 0] = 1
        time_activity = time_activity != 0

        return time_activity != 0


class RadioProvider(Configurable):
    @classmethod
    def finalize_dogmatic_config(cls, config):
        config['collate'] = dict(
            factory=Padder,
            to_torch=False,
            sort_by_key=K.NUM_SAMPLES,
            padding=False,
            padding_keys=None
        )

    def __init__(
            self,
            database,
            transform,
            collate,
            audio_keys: tuple = (K.OBSERVATION,),
            shuffle: bool = True,
            batch_size: int = 1,
            batch_size_eval: int = None,
            num_workers: int = 4,
            buffer_size: int = 20,
            backend: str = 't',
            drop_last: bool = False,
            time_segments: int = 32000,
            sample_rate: int = 8000,
            min_speech_activity: int = 0,
            interference_activity: bool = False
    ):
        self.database = database
        self.transform = transform if transform is not None else lambda x: x
        self.collate = collate
        self.audio_keys = audio_keys
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.batch_size_eval = batch_size_eval
        self.num_workers = num_workers
        self.buffer_size = buffer_size
        self.backend = backend
        self.drop_last = drop_last
        self.time_segments = time_segments
        self.sample_rate = sample_rate
        self.min_speech_activity = min_speech_activity
        self.interference_activity = interference_activity

    @staticmethod
    def _example_id_to_rng(example_id):
        hash_value = md5(example_id.encode())
        hash_value = int(hash_value.hexdigest(), 16)
        hash_value = hash_value % 2 ** 32 - 1
        return np.random.RandomState(hash_value)

    def to_dict_structure(self, example):
        """Function to be mapped on an iterator."""
        out_dict = example[K.AUDIO_DATA]
        out_dict['audio_keys'] = list(example[K.AUDIO_DATA].keys())
        out_dict[K.EXAMPLE_ID] = example[K.EXAMPLE_ID]
        out_dict[K.NUM_SAMPLES] = example[K.NUM_SAMPLES]
        if isinstance(example[K.NUM_SAMPLES], dict):
            out_dict[K.NUM_SAMPLES] = example[K.NUM_SAMPLES][K.OBSERVATION]
        else:
            out_dict[K.NUM_SAMPLES] = example[K.NUM_SAMPLES]

        if K.ALIGNMENT_ACTIVITY in example:
            out_dict[K.TARGET_TIME_VAD] = ArrayInterval.from_str(
                *example[K.ALIGNMENT_ACTIVITY])[:]
        elif K.ACTIVITY:
            out_dict[K.TARGET_TIME_VAD] = ArrayInterval.from_str(
                *example[K.ACTIVITY])[:]

        if self.interference_activity:
            assert K.INTERFERENCE_ACTIVITY in example, example.keys()
            interference_activity = ArrayInterval.from_serializable(
                example[K.INTERFERENCE_ACTIVITY])[:]
            activity = out_dict[K.TARGET_TIME_VAD] + interference_activity
            activity[activity > 0] = 1
            out_dict[K.TARGET_TIME_VAD] = activity.astype(bool)

        out_dict['audio_keys'].append(K.TARGET_TIME_VAD)
        return out_dict

    def read_audio(self, example):
        """Function to be mapped on an iterator."""
        if K.START in example and K.END in example:
            example[K.AUDIO_DATA] = {
                key: pb.io.load_audio(
                    example[K.AUDIO_PATH][key], start=example['start'],
                    frames=example[K.NUM_SAMPLES]
                )
                for key in self.audio_keys
            }
        else:
            example[K.AUDIO_DATA] = {
                key: pb.io.load_audio(example[K.AUDIO_PATH][key])
                for key in self.audio_keys
            }
        if K.DELAY in example and K.SPEECH_SOURCE in self.audio_keys:
            delay = example[K.DELAY]
            source = example[K.AUDIO_DATA][K.SPEECH_SOURCE]
            if delay > 0:
                example[K.AUDIO_DATA][K.SPEECH_SOURCE] = np.concatenate(
                    [np.zeros_like(source)[:delay], source[:-delay]])
            elif delay < 0:
                example[K.AUDIO_DATA][K.SPEECH_SOURCE] = np.concatenate(
                    [source[-delay:], np.zeros_like(source)[:-delay]])
            else:
                example[K.AUDIO_DATA][K.SPEECH_SOURCE] = source
        return example

    def segment(self, example):
        segment_len = shift = self.time_segments
        audio_keys = example['audio_keys']
        num_samples = np.min([example[key].shape[-1] for key in audio_keys])
        for key in audio_keys:
            example[key] = segment_axis(
                example[key][..., :num_samples], segment_len,
                shift=shift, axis=-1, end='cut')
        lengths = ([example[key].shape[-2] for key in audio_keys])
        assert lengths.count(lengths[-2]) == len(lengths), {
            audio_keys[idx]: leng for idx, leng in enumerate(lengths)}
        length = lengths[0]
        if length == 0:
            print('was to short')
            raise FilterException
        out_list = list()
        example[K.NUM_SAMPLES] = self.time_segments
        for idx in range(length):
            new_example = deepcopy(example)
            for key in audio_keys:
                new_example[key] = new_example[key][..., idx, :]
            if np.sum(new_example[K.TARGET_TIME_VAD]) >= self.min_speech_activity:
                out_list.append(new_example)

        if len(out_list) == 0:
            print('This should not happen regularly')
            raise FilterException

        shuffle_fn(out_list)
        return out_list

    def get_map_iterator(self, iterator, batch_size=None,
                         prefetch=True, unbatch=False):
        iterator = iterator.map(self.transform)
        if prefetch:
            iterator = iterator.prefetch(
                self.num_workers, self.buffer_size,
                self.backend, catch_filter_exception=True
            )
        if unbatch:
            iterator = iterator.unbatch()
        if batch_size is not None:
            iterator = iterator.batch(batch_size, self.drop_last)
            iterator = iterator.map(self.collate)
        else:
            if self.batch_size is not None:
                iterator = iterator.batch(self.batch_size, self.drop_last)
                iterator = iterator.map(self.collate)
        return iterator

    def update_iterator(self, iterator):
        iterator = iterator.map(self.read_audio) \
            .map(self.database.add_num_samples)
        iterator = iterator.map(self.to_dict_structure)
        return iterator

    def get_train_iterator(self, time_segment=None):
        self.is_training = True
        self.transform.is_training = True
        iterator = self.database.get_dataset_train()
        iterator = self.update_iterator(iterator)
        unbatch = False
        if self.shuffle:
            iterator = iterator.shuffle(reshuffle=True)
        if self.time_segments is not None or time_segment is not None:
            assert not (self.time_segments and time_segment)
            iterator = iterator.map(partial(self.segment))
            unbatch = True
        return self.get_map_iterator(
            iterator, self.batch_size, unbatch=unbatch)

    def get_eval_iterator(self, num_examples=-1):
        self.is_training = False
        self.transform.is_training = False
        iterator = self.database.get_dataset_validation()
        iterator = self.update_iterator(iterator)
        iterator = iterator[:num_examples]
        return self.get_map_iterator(
            iterator, self.batch_size_eval, unbatch=False)

    def get_predict_iterator(self, num_examples=-1, dataset=None,
                             iterable_apply_fn=None, filter_fn=None):
        self.is_training = False
        self.transform.is_training = False
        if dataset is None:
            iterator = self.database.get_dataset_test()
        else:
            iterator = self.database.get_dataset(dataset)
        iterator = self.update_iterator(iterator)
        iterator = iterator[:num_examples]
        if iterable_apply_fn is not None:
            iterator = iterator.apply(iterable_apply_fn)
        iterator = self.get_map_iterator(iterator, prefetch=False,
                                         batch_size=self.batch_size_eval)
        if filter_fn is not None:
            iterator = iterator.filter(filter_fn)
        return iterator