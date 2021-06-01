"""
Training script for a neural network based SAD system.
The following line starts the training:
export HAM_RADIO_JSON_PATH=/PATH/TO/HAM_RADIO_JSON & \
export STORAGE_ROOT=/PATH/TO/MODEL_DIR & \
python -m ham_radio.train with cnn
"""
import os
from copy import deepcopy
from pathlib import Path

import numpy as np
import sacred
from ham_radio import keys as K
from ham_radio.system.data import Transformer, RadioProvider
from ham_radio.system.model import SADModel
from ham_radio.system.module import CNN1d, CNN2d
from ham_radio.system.utils import Pool1d
from paderbox.io import dump_json
from paderbox.utils.nested import deflatten
from paderbox.io.new_subdir import get_new_subdir
from padertorch.configurable import class_to_str
from padertorch.configurable import config_to_instance
from padertorch.configurable import recursive_class_to_str
from padertorch.train.optimizer import Adam
from padertorch.train.trainer import Trainer
from sacred.utils import apply_backspaces_and_linefeeds

ex = sacred.Experiment('Train Voice Activity Detector')
ex.captured_out_filter = apply_backspaces_and_linefeeds


@ex.config
def config():
    max_it = int(2e5)
    model_dir = Path(os.environ['MODEL_DIR'])
    trainer_opts = deflatten({
        'model': {
            'factory': SADModel,
            'cnn_2d': {
                'factory': CNN2d,
                'in_channels': 1,
                'hidden_channels': (
                    np.array([16, 16, 32, 32, 64, 64])).tolist(),
                'pool_size': [1, (4, 1), 1, (8, 1), 1, (8, 1)],
                'num_layers': 6,
                'out_channels': None,
                'kernel_size': 3,
                'norm': 'batch',
                'activation': 'relu',
                'dropout': .0,
            },
            'cnn_1d': None,
            'pooling': {'factory': Pool1d, 'pooling': 'max', 'pool_size': 10,
                        'padding': None},
            'recall_weight': 1.,
            'input_norm': 'l2_norm',
        },
        'optimizer.factory': Adam,
        'stop_trigger': (int(1e5), 'iteration'),
        'summary_trigger': (500, 'iteration'),
        'checkpoint_trigger': (500, 'iteration'),
        'storage_dir': None,
        'virtual_minibatch_size': 1
    })
    stft_size = 256
    provider_opts = deflatten({
        'database': {
            'factory': 'ham_radio.database.HamRadioLibrispeech',
            'json_path': os.environ['HAM_RADIO_JSON_PATH']
        },
        'transform': {
            'factory': Transformer,
            'stft': dict(size=stft_size, shift=80, window_length=200),
        },
        'batch_size': 24,
        'batch_size_eval': 1,
        'buffer_size': 10,
        'audio_keys': [K.OBSERVATION],
        'collate': dict(
            sort_by_key=K.NUM_SAMPLES,
            padding=True,
            padding_keys=[K.SPEECH_FEATURES]
        ),
        'time_segments': 32000,
    })
    database_name = 'ham_radio'
    storage_dir = None
    add_name = None
    if storage_dir is None:
        model_name = class_to_str(trainer_opts['model']['factory'])
        assert isinstance(model_name, str), (model_name, type(model_name))
        ex_name = f'{model_name.split(".")[-1]}'
        if add_name is not None:
            ex_name += f'_{add_name}'
        storage_dir = get_new_subdir(model_dir / database_name,
                                     prefix=ex_name)
        sacred.observers.FileStorageObserver(storage_dir)
    else:
        sacred.observers.FileStorageObserver.create(storage_dir)
    trainer_opts['storage_dir'] = storage_dir

    trainer_opts = Trainer.get_config(
        trainer_opts
    )
    provider_opts = RadioProvider.get_config(
        provider_opts
    )
    debug = False
    validation_length = 40  # number of examples taken from the validation iterator
    validation_kwargs = dict(
        metric='loss', maximize=False, max_checkpoints=1, n_back_off=0,
        lr_update_factor=1 / 10, back_off_patience=None,
        early_stopping_patience=None
    )


@ex.named_config
def rnn():
    trainer_opts = dict(model=dict(
        rnn={
            'factory': 'torch.nn.GRU',
            'input_size': 64,
            'hidden_size': 256,
            'num_layers': 2,
            'dropout': 0.5,
            'bidirectional': True,
            'batch_first': False,
        },
        cnn_1d=None
    ))
    add_name = 'rnn'


@ex.named_config
def cnn():
    trainer_opts = dict(model=dict(
        cnn_1d={
            'factory': CNN1d,
            'in_channels': 64,
            'hidden_channels': 128,
            'out_channels': 10,
            'num_layers': 2,
            'kernel_size': [3, 1],
            'norm': 'batch',
            'activation': 'relu',
            'dropout': .0
        },
        rnn=None,
    ))


@ex.capture
def initialize_trainer_provider(task, trainer_opts, provider_opts, _run):
    assert 'database' in provider_opts, provider_opts
    assert 'factory' in provider_opts['database'], provider_opts
    storage_dir = Path(trainer_opts['storage_dir'])
    trainer_opts = deepcopy(trainer_opts)
    provider_opts = deepcopy(provider_opts)
    if (storage_dir / 'init.json').exists():
        assert task in ['restart', 'validate'], task
    elif task in ['train', 'create_checkpoint']:
        dump_json(dict(trainer_opts=recursive_class_to_str(trainer_opts),
                       provider_opts=recursive_class_to_str(provider_opts)),
                  storage_dir / 'init.json')
    else:
        raise ValueError(task, storage_dir)
    from paderbox.utils.pretty import pprint
    pprint('provider_opts:', provider_opts)
    pprint('trainer_opts:', trainer_opts)
    trainer = Trainer.from_config(trainer_opts)
    assert isinstance(trainer, Trainer)
    provider = config_to_instance(provider_opts)
    return trainer, provider


@ex.command
def restart(validation_length, validation_kwargs):
    trainer, provider = initialize_trainer_provider(task='restart')
    train_iterator = provider.get_train_iterator()
    validation_iterator = provider.get_eval_iterator(
        num_examples=validation_length
    )
    trainer.load_checkpoint()
    trainer.test_run(train_iterator, validation_iterator)
    trainer.register_validation_hook(validation_iterator, **validation_kwargs)
    trainer.train(train_iterator, resume=True)


@ex.automain
def train(debug, validation_kwargs, validation_length):
    trainer, provider = initialize_trainer_provider(task='train')
    train_iterator = provider.get_train_iterator()
    if debug:
        validation_iterator = provider.get_eval_iterator(
            num_examples=2
        )
    else:
        validation_iterator = provider.get_eval_iterator(
            num_examples=validation_length
        )
    trainer.register_validation_hook(validation_iterator, **validation_kwargs)
    trainer.test_run(train_iterator, validation_iterator)
    trainer.train(train_iterator)
