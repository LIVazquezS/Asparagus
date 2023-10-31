import sys
from setuptools import setup, find_packages
#sys.path.append('.')

with open('README.md','r') as fh:
    long_description = fh.read()

setup(
    name='Asparagus',
    version='0.1.0',
    description='PyTorch refined implementation of PhysNet',
    author='L.I.Vazquez-Salazar and Kai Toepfer',
    long_description=long_description,
    author_email='luisitza.vazquezsalazar@unibas.ch',
    packages=find_packages(include=['asparagus']),
    include_package_data=True,
    install_requires=[
        'ase', 
        'numpy', 
        'tensorboard', 
        'torch', 
        'torch-ema',
        'xtb']
    #TODO: Add more dependencies and option to be read from a file
)
