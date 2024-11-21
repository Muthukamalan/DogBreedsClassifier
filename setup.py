#!/usr/bin/env python

from setuptools import find_packages, setup

setup(
    name='DogBreeds',
    version='1.0.0',
    description='helps to classify dog breeds',
    author='Muthu',
    author_email='muthukamalan98@gmail.com',
    url='https://github.com/Muthukamalan/DogBreedsClassifier',
    install_requires=['lightning', 'hydra-core', 'aws', 'dvc'],
    packages=find_packages(),
    # use this to customize global commands available in the terminal after installing the package
    entry_points={
        'console_scripts': [
            'train_command = src.train:main',
            'eval_command = src.eval:main',
        ]
    },
)
