import os
from setuptools import setup
from setuptools import find_packages

setup(
    name='slac.pytorch',
    version='1.0.0',
    author='Rody Haket',
    description=('SLAC in PyTorch'),
    license='',
    keywords='slac disentanglement drl',
    packages=find_packages(),
    install_requires=[
        'dmc2gym',
        'torch',
        'tensorboard',
        'gymnasium',
        'tdqm'
    ],
)