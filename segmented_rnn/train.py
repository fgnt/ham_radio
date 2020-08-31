"""

"""
import os
from pathlib import Path

import numpy as np
import sacred
from copy import deepcopy
from paderbox.io import dump_json
from paderbox.utils.nested import deflatten
from padertorch.configurable import config_to_instance
from padertorch.configurable import recursive_class_to_str
from padertorch.configurable import class_to_str
from padertorch.train.optimizer import Adam
from padertorch.train.trainer import Trainer
from sacred.utils import apply_backspaces_and_linefeeds
from segmented_rnn import keys as K

from segmented_rnn.system.model import BinomialClassifier
from segmented_rnn.system.module import CNN1d, CNN2d
from segmented_rnn.system.data import Transformer, RadioProvider
from segmented_rnn.system.utils import Pool1d

ex = sacred.Experiment('Train Voice Activity Detector')
ex.captured_out_filter = apply_backspaces_and_linefeeds

@ex.config
def config():
    max_it = int(2e5)
    model_dir = Path(os.environ['MODEL_DIR'])
    trainer_opts = deflatten({
        'model': {
            'factory': BinomialClassifier,
            'cnn_2d': {
                'factory': CNN2d,
                'in_channels': 1,
                'hidden_channels': (np.array([16, 16, 32, 32, 64, 64])).tolist(),
                'pool_size': [1, (4, 1), 1, (8, 1), 1, (8, 1)],
                'num_layers': 6,
                'out_channels': None,
                'kernel_size': 3,
                'norm': 'batch',
                'activation': 'relu',
                'dropout': .0,
            },
            'cnn_1d': None,
            'pooling': {'factory': Pool1d, 'pooling':'max', 'pool_size': 10,
                        'padding': None},
            'recall_weight': 1.,
            'input_norm': None
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
        'transform': {
            'factory': Transformer,
            'stft': dict(size=stft_size, shift=80, window_length=200),
            'mel': None
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
    database_name = None
    storage_dir = None
    add_name = None
    if storage_dir is None:
        model_name = class_to_str(trainer_opts['model']['factory'])
        assert isinstance(model_name, str), (model_name, type(model_name))
        ex_name = f'{model_name.split(".")[-1]}'
        if add_name is not None:
            ex_name += f'_{add_name}'
        observer = sacred.observers.FileStorageObserver(
            str(model_dir / database_name / ex_name))
        storage_dir = observer.basedir
    else:
        sacred.observers.FileStorageObserver.create(storage_dir)
    trainer_opts['storage_dir'] = storage_dir

    trainer_opts = Trainer.get_config(
        trainer_opts
    )
    provider_opts = RadioProvider.get_config(
        provider_opts
    )
    debug=False
    validate_checkpoint = 'ckpt_latest.pth'
    validation_length = 1000  # number of examples taken from the validation iterator
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
def segment_rnn():
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
        segmented_rnn=True,
        window_length=50,
        window_shift=25,
        cnn_1d=None
    ))
    add_name = 'segmented_rnn'


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


@ex.named_config
def fearless():
    provider_opts = {
        'database': {
            'factory': 'segmented_rnn.database.Fearless',
        }
    }
    database_name = 'fearless'
    trainer_opts = {
            'model': {'input_norm': None}
        }

@ex.named_config
def ham_radio():
    provider_opts = {
        'database': {
            'factory': 'segmented_rnn.database.HamRadioLibrispeech',
        }
    }
    trainer_opts = {
        'model': {'input_norm': 'l2_norm'}
    }
    database_name = 'ham_radio'


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
