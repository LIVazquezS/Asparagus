from setuptools import setup, find_packages
import sys
sys.path.append('.')

with open('README.md','r') as fh:
    long_description = fh.read()
setup(
    name='PhysPack',
    version='0.1.0',
    description='PyTorch refined implementation of PhysNet',
    author='L.I.Vazquez-Salazar and Kai Toepfer',
    long_description=long_description,
    author_email='luisitza.vazquezsalazar@unibas.ch',
    packages=find_packages(include=['physpack']),
    include_package_data=True,
    install_requires=['numpy','torch','tensorboard','ase','torch-ema'] #TODO: Add more depenedencies and option to be read from a file
)