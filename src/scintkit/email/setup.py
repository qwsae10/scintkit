from setuptools import setup, find_packages

setup(
    name='scintkit',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'pandas',
        'matplotlib',
        'h5py'
    ],
    description='Library for ScintPi data availability scanning, plotting, and emailing.',
)