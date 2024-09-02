import sys
from setuptools import setup, find_packages
#sys.path.append('.')

with open('README.md','r') as fh:
    long_description = fh.read()

setup(
    name='Asparagus',
    version='0.2.2',
    description='Function Bundle from Sampling, Training to Application of NN Potentials',
    author='L.I.Vazquez-Salazar, Silvan Kaeser and Kai Toepfer',
    long_description=long_description,
    author_email='luisitza.vazquezsalazar@unibas.ch',
    packages=find_packages(include=['asparagus']),
    include_package_data=True,
    install_requires=[
        'ase==3.22.1', 
        'numpy<2.0', 
        'tensorboard', 
        'torch==1.12.0', 
        'torch-ema>=0.3',
        'tabulate',
        'h5py',
        'xtb',
        'pandas',
        'matplotlib',
        'seaborn',
        'scipy',
        'pytest',
    ]
    #TODO: Add more dependencies and option to be read from a file
)
