import numpy as np
import paderbox as pb
from pathlib import Path

from paderbox.utils.mapping import Dispatcher
from paderbox.array.interval import ArrayInterval

from ham_radio import keys as K
from ham_radio.database.utils import dump_database_as_json
from ham_radio.database.utils import check_audio_files_exist
from sacred import Experiment

ex = Experiment('Create database json')

class HamRadioLibrispeechKeys:
    NOISE = 'noise'
    STATION = 'station'
    LIBRISPEECH_ID = 'librispeech_id'


dset2id_mapping = Dispatcher(
    train=(0, 140),
    dev=(176, 190),
    eval=(141, 175),
)

SAMPLE_RATE = 16000
HRL_K = HamRadioLibrispeechKeys


def get_example(audio: Path, org_json, transcription_json, dset_name,
                id_sub_fix=None):
    audio = Path(audio).expanduser().resolve()
    station, _id = audio.stem.split('__')
    num_samples = pb.io.audioread.audio_length(audio)

    ex = org_json[_id].copy()
    activity = ArrayInterval.from_str(*ex[K.ACTIVITY])
    assert np.isclose(activity.shape[-1], num_samples, 0.1), (
        activity.shape, num_samples)

    activity = ArrayInterval.from_str(*ex[K.ALIGNMENT_ACTIVITY])
    assert activity.shape[-1] == num_samples, (activity.shape, num_samples)
    # assert np.isclose(np.sum(ex['speech_length']), np.sum(activity), 0.1), (
    #     sum(ex['speech_length']), sum(activity))
    org_json[HRL_K.STATION] = station
    if int(_id) == 180:
        if audio.parent.name == '19_12_10_14_39_04__176_180':
            station += '_0'
        elif audio.parent.name == '19_12_10_14_56_58__180_185':
            station += '_1'
        else:
            raise ValueError()
    if id_sub_fix:
        station += '_' + id_sub_fix
    first_value = activity[0]
    assert first_value == 0
    clean_dir = audio.parents[3] / 'clean' / dset_name
    audio_dict = {
        K.SPEECH_SOURCE: clean_dir / f'clean_{_id}.wav',
        K.OBSERVATION: str(audio),
    }

    ex_id = station + '_' + _id

    ex[K.AUDIO_PATH] = audio_dict
    ex[K.ORIGINAL_TRANSCRIPTION] = ex['transcriptions']
    ex[K.TRANSCRIPTION] = transcription_json[_id]
    return ex_id, ex


def create_database(database_path):
    orig_json = pb.io.load_json(database_path / 'annotations.json')
    transcription_json = pb.io.load_json(database_path / 'transcription.json')
    db = {K.DATASETS: {}, K.ALIAS: {}}
    for dset in (database_path / 'noisy').glob('*'):
        dset_name = dset.name
        id_min, id_max = dset2id_mapping[dset_name]
        for dirs in dset.glob('*'):
            ex_dict = dict()
            for audio in dirs.glob('*.wav'):
                key, ex = get_example(audio, orig_json, transcription_json,
                                      dset_name)
                ex_dict[key] = ex
            ids = dirs.name.split('__')[1].split('_')
            assert id_min <= int(ids[0]) < id_max, (dirs.name, dset_name)
            assert id_min < int(ids[1]) <= id_max, (dirs.name, dset_name)
            db[K.DATASETS][f'{dset_name}_{"_".join(ids)}'] = ex_dict
        db[K.ALIAS].update({dset_name: [key for key in db[K.DATASETS].keys()
                                        if dset_name in key]})
        print(dset_name, 'includes the following ids', db[K.ALIAS][dset_name])
    return db


@ex.config
def config():
    database_path = None
    assert database_path is not None, 'You have to define a database path to' \
                                      'create a database json'
    assert Path(database_path).is_dir(), database_path
    json_path = None
    if json_path is None:
        import os
        assert 'HAM_RADIO_JSON' in os.environ, (
                'If json_path is not defined in the function call the global '
                'variable HAM_RADIO_JSON has to be set'
        )
        json_path = os.environ['HAM_RADIO_JSON']
    assert not Path(json_path).exists(), (
            'The specified json already exists. Please delete '
            'or rename the file'
    )

@ex.automain
def main(database_path, json_path):
    json = create_database(Path(database_path))
    print("Check that all wav files in the json exist.")
    check_audio_files_exist(json, speedup="thread")
    print("Finished check.")
    dump_database_as_json(json_path, json)