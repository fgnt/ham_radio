import os

import numpy as np
import paderbox as pb
from pathlib import Path
from lazy_dataset.database import JsonDatabase

JSON_PATH = Path(os.environ['JSON_PATH'])


class HamRadioLibrispeech(JsonDatabase):
    def __init__(
            self,
            json_path: [str, Path] = JSON_PATH / 'ham_radio.json',
    ):
        super().__init__(json_path)

    @property
    def read_fn(self):
        def fn(x, *, start=0, frames=-1):

            x, sample_rate = pb.io.load_audio(
                x, frames=frames, start=start,
                return_sample_rate=True
            )
            x = x.astype(np.float32)

            if sample_rate == 8000:
                return x
            else:
                raise RuntimeError(
                    f'Unexpected file found: {x}\n'
                    f'x.shape: {x.shape}\n'
                    f'expected sample rate: 8000\n'
                )

        return fn

    def get_dataset_train(self):
        return self.get_dataset(['train'])

    def get_dataset_validation(self):
        return self.get_dataset(['validation'])

    def get_dataset_test(self):
        return self.get_dataset(['test'])

    def add_num_samples(self, example):
        if isinstance(example['num_samples'], dict):
            example['num_samples'] = example['num_samples']['observation']
        return example

