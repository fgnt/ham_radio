import click
import lazy_dataset
import jsonpickle
import numpy as np
import paderbox as pb

from padercontrib.database import keys as K
from paderbox.array.intervall import ArrayIntervall
from segmented_rnn.database.utils import click_common_options
from segmented_rnn.database.utils import click_convert_to_path
from segmented_rnn.database.utils import dump_database_as_json

class HamRadioLibrispeechKeys:
    NOISE = 'noise'
    STATION = 'station'
    LIBRISPEECH_ID = 'librispeech_id'


SAMPLE_RATE = 16000
HRL_K = HamRadioLibrispeechKeys


def get_example(audio, org_json):
    station, _id = audio.stem.split('__')
    ex = org_json[_id].copy()
    num_samples = pb.io.audio_length(audio)
    ex['silence_length'][0] += SAMPLE_RATE * 5
    ex['silence_length'][-1] += SAMPLE_RATE
    ex['silence_length'] = [l // 2 for l in ex['silence_length']]
    ex['speech_length'] = [l // 2 for l in ex['speech_length']]
    activity = jsonpickle.loads(ex['activity'])[:]
    activity_time = np.concatenate([
        np.zeros(SAMPLE_RATE * 5, activity.dtype),
        activity,
        np.zeros(SAMPLE_RATE, activity.dtype)
    ], axis=0)[::2]
    assert np.isclose(np.sum(activity) / 2, np.sum(activity_time), 0.1), (
    sum(activity) / 2, sum(activity_time))
    assert np.isclose(activity_time.shape[-1], num_samples, 0.1), (
    activity_time.shape, num_samples)
    ex['activity'] = jsonpickle.dumps(ArrayIntervall(
        activity_time))
    activity = jsonpickle.loads(ex['alignment_activity'])[:]
    activity_freq = np.concatenate([
        np.zeros(SAMPLE_RATE * 5, activity.dtype),
        activity,
        np.zeros(SAMPLE_RATE, activity.dtype)
    ], axis=0)[::2]
    assert np.isclose(np.sum(activity) // 2, np.sum(activity_freq), 0.1), (
    sum(activity) // 2, sum(activity_freq))
    assert np.isclose(activity_freq.shape[-1], num_samples, 1e-1), (
    activity_freq.shape, num_samples)
    ex['alignment_activity'] = jsonpickle.dumps(
        ArrayIntervall(activity_freq))
    assert np.isclose(np.sum(ex['speech_length']), np.sum(activity_time), 0.1), (
    sum(ex['speech_length']), sum(activity_time))
    audio_dict = {
        K.SPEECH_SOURCE: ex[K.AUDIO_PATH][K.OBSERVATION],
        K.OBSERVATION: str(audio),
    }
    ex_id = station + '_' + _id
    ex[K.NUM_SAMPLES] = num_samples
    ex[K.AUDIO_PATH] = audio_dict
    ex[HRL_K.STATION] = station
    ex[K.EXAMPLE_ID] = ex_id
    return ex_id, ex


def create_database(database_path, origin_path):
    org_json = pb.io.load_json(origin_path / 'information.json')
    db = {K.DATASETS: {
        'train':{}, 'validation': {}, 'test': {}, 'test_evening':{}}
    }
    for dirs in database_path.glob('*'):
        ex_dict = dict()
        for audio in dirs.glob('*.wav'):
            ex_id, ex = get_example(audio, org_json)
            ex_dict[ex_id] = ex
        if dirs.name.split('__')[1] == '141_145':
            db[K.DATASETS]['validation'].update(ex_dict)
            print('validation', dirs)
        elif 146 <= int(dirs.name.split('__')[1].split('_')[0]) < 180:
            db[K.DATASETS]['test'].update(ex_dict)
            print('test', dirs)
            db[K.DATASETS]['test'].update(ex_dict)
            print('test', dirs)
        else:
            db[K.DATASETS]['train'].update(ex_dict)
            print('train', dirs)
    lazy_ds = lazy_dataset.from_dict(db[K.DATASETS]['train'])
    assert len(lazy_ds) == len(set(lazy_ds.keys()))
    return db


@click.command()
@click_common_options('ham_radio_librispeech_v1.json', '/net/vol/ham/Cut')
@click.option(
    '--origin-path', default='/net/vol/jensheit/plath/data/clean_evaluation',
    help=f'Path with clean librispeech combinations.',
    type=click.Path(exists=True, dir_okay=True, writable=False),
    callback=click_convert_to_path,
)

def main(database_path, json_path, origin_path):
    json = create_database(database_path, origin_path)
    dump_database_as_json(json_path, json)


if __name__ == '__main__':
    main()
