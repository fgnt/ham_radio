"""A setuptools based setup module.

See:
https://packaging.python.org/en/latest/distributing.html
https://github.com/pypa/sampleproject
"""

# To use a consistent encoding
from codecs import open
# Always prefer setuptools over distutils
from distutils.core import setup
from os import path

from setuptools import find_packages

here = path.abspath(path.dirname(__file__))

# Get the long description from the relevant file

setup(
    name='segmented_rnn',

    # Versions should comply with PEP440.  For a discussion on single-sourcing
    # the version across setup.py and the project code, see
    # https://packaging.python.org/en/latest/single_source_version.html
    version='0.0.0',

    description='A collection of torch based speech separation networks',

    # The project's main homepage.
    url='https://ei.uni-paderborn.de/nt/',

    # Author details
    author='Jens Heitkaemper',
    author_email='j.heitkaemper@gmail.com',

    # Choose your license
    license='MIT',

    # See https://pypi.python.org/pypi?%3Aaction=list_classifiers
    classifiers=[
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 3 - Alpha',

        # Indicate who your project is intended for
        'Intended Audience :: Developers',
        'Topic :: Software Development :: Build Tools',

        # Pick your license as you wish (should match "license" above)
        'License :: OSI Approved :: MIT License',

        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        'Programming Language :: Python :: 3.6',
    ],

    # What does your project relate to?
    keywords='pytorch',

    # You can just specify the packages manually here if your project is
    # simple. Or you can use find_packages().
    packages=find_packages(exclude=['contrib', 'docs', 'tests*']),

    # List run-time dependencies here.  These will be installed by pip when
    # your project is installed. For an analysis of "install_requires" vs pip's
    # requirements files see:
    # https://packaging.python.org/en/latest/requirements.html
    extra_require={
        'all': [
            'numpy==1.18.1',
            'Cython==0.29.14',
            'paderbox @ https://github.com/fgnt/paderbox/tarball/7b3b4e9d00e07664596108f987292b8c78d846b1#egg',
            'padertorch @ https://github.com/fgnt/padertorch/tarball/3d0355edc9e3d7cdbcccb80e85107c5d78a47ccb#egg',
            'torch==1.4.0',
            'sacred==0.8.1',
            'lazy_dataset==0.0.9',
            'jsonpickle==1.2',
        ]
    },

    # Installation problems in a clean, new environment:
    # 1. `cython` and `scipy` must be installed manually before using
    # `pip install`
)
