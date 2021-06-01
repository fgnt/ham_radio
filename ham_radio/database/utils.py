import pathlib
from pathlib import Path
import collections


from paderbox.io.json_module import dump_json


# http://codereview.stackexchange.com/questions/21033/flatten-dictionary-in-python-functional-style
def flatten_with_key_paths(
        d, sep=None, flat_list=True, reverse_key_value=False, condition_fn=None
):
    """
    Example:
    >>> d = {'a': 1, 'c': {'a': 2, 'b': {'x': 5, 'y' : 10}}, 'd': [1, 2, 3]}
    >>> flatten_with_key_paths(d, flat_list=False)
    {('a',): 1, ('c', 'a'): 2, ('c', 'b', 'x'): 5, ('c', 'b', 'y'): 10, ('d',): [1, 2, 3]}
    >>> flatten_with_key_paths(d, sep='/', flat_list=False)
    {'a': 1, 'c/a': 2, 'c/b/x': 5, 'c/b/y': 10, 'd': [1, 2, 3]}
    >>> flatten_with_key_paths(d, sep='/', flat_list=True)
    {'a': 1, 'c/a': 2, 'c/b/x': 5, 'c/b/y': 10, 'd/0': 1, 'd/1': 2, 'd/2': 3}
    >>> flatten_with_key_paths(d, sep='/', flat_list=True, reverse_key_value=True)
    {1: 'd/0', 2: 'd/1', 5: 'c/b/x', 10: 'c/b/y', 3: 'd/2'}
    >>> flatten_with_key_paths(1, sep='/', flat_list=False)
    {'': 1}

    """

    res = {}

    def fetch(prefix, v0):
        if isinstance(v0, dict):
            for k, v in v0.items():
                fetch(prefix + (k,), v)
        elif flat_list and isinstance(v0, (tuple, list)):
            for k, v in enumerate(v0):
                fetch(prefix + (str(k),), v)
        else:
            key = prefix if sep is None else sep.join(prefix)
            if condition_fn is None or condition_fn(key, v0):
                if reverse_key_value:
                    res[v0] = key
                else:
                    res[key] = v0

    fetch((), d)
    return res


def check_audio_files_exist(
        database_dict,
        speedup=None,
):
    """
    No structure for the database_dict is assumed. It will just search for all
    string values ending with a certain file type (e.g. wav).

    >>> check_audio_files_exist({2: [1, '/net/db/timit/pcm/train/dr1/fcjf0/sa1.wav', 'abc.wav']})
    Traceback (most recent call last):
      ...
    AssertionError: ('abc.wav', (2, '2'))
    >>> check_audio_files_exist(1)
    Traceback (most recent call last):
      ...
    AssertionError: Expect at least one wav file. It is likely that the database folder is empty and the greps did not work. to_check: {}
    >>> check_audio_files_exist('abc.wav')
    Traceback (most recent call last):
      ...
    AssertionError: ('abc.wav', ())
    >>> check_audio_files_exist('/net/db/timit/pcm/train/dr1/fcjf0/sa1.wav')
    >>> check_audio_files_exist(1, speedup='thread')
    Traceback (most recent call last):
      ...
    AssertionError: Expect at least one wav file. It is likely that the database folder is empty and the greps did not work. to_check: {}
    >>> check_audio_files_exist('abc.wav', speedup='thread')
    Traceback (most recent call last):
      ...
    AssertionError: ('abc.wav', ())
    >>> check_audio_files_exist('/net/db/timit/pcm/train/dr1/fcjf0/sa1.wav', speedup='thread')
    """

    def path_exists(path):
        return Path(path).exists()

    def body(file_key_path):
        file, key_path = file_key_path
        assert path_exists(file), (file, key_path)

    def condition_fn(key_path, file):
        extensions = ('.wav', '.wv2', '.wv1', '.flac')
        return isinstance(file, (str, Path)) and str(file).endswith(extensions)

    # In case of CHiME5 flatten_with_key_paths is the bottleneck of this function
    to_check = flatten_with_key_paths(
        database_dict, reverse_key_value=True, condition_fn=condition_fn
    )

    assert len(to_check) > 0, (
        f'Expect at least one wav file. '
        f'It is likely that the database folder is empty '
        f'and the greps did not work. to_check: {to_check}'
    )

    if speedup and 'thread' == speedup:
        import os
        from multiprocessing.pool import ThreadPool

        # Use this number because ThreadPoolExecutor is often
        # used to overlap I/O instead of CPU work.
        # See: concurrent.futures.ThreadPoolExecutor
        # max_workers = (os.cpu_count() or 1) * 5

        # Not sufficiently benchmarked both, this is more conservative.
        max_workers = (os.cpu_count() or 1)

        with ThreadPool(max_workers) as pool:
            for _ in pool.imap_unordered(
                body,
                to_check.items()
            ):
                pass

    elif speedup is None:
        for file, key_path in to_check.items():
            assert path_exists(file), (file, key_path)
    else:
        raise ValueError(speedup, type(speedup))


def dump_database_as_json(filename, database_dict, *, indent=4):
    """
    Dumps a `database_dict` as json to `filename`. Ensures that filename has the
    extension '.json' and creates parent directories if necessary.
    """
    filename = pathlib.Path(filename)
    assert filename.suffix == '.json', f'Json file must end with ".json" and ' \
                                       f'not "{filename.suffix}"'
    # ToDo: Why ensure_ascii=False?
    dump_json(
        database_dict,
        filename,
        create_path=True,
        indent=indent,
        ensure_ascii=False,
    )
    print(f'Wrote {filename}')


def default_dict():
    """
    Defaultdict for json structure.
    """
    database = collections.defaultdict(
        lambda: collections.defaultdict(
            lambda: collections.defaultdict(
                lambda: collections.defaultdict(dict)
            )
        )
    )
    return database