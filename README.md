# Project Asparagus

**Authors**: L.I. Vazquez-Salazar, K. Toepfer

## What is this?
 - A refined implementation of PhysNet NN (and other atomistic NN to come) in PyTorch. 
 - A Suit for the automatic construction of Potential Energy Surface (PES) since sampling to production.

## How to use? 

- Clone the repository
- Requirements:
  - Python <img src="https://latex.codecogs.com/svg.image?\geq&space;" title="\geq " /> 3.8 (I recommend to use 3.8.12, **DO NOT** use 3.9)
  - PyTorch <img src="https://latex.codecogs.com/svg.image?\geq&space;" title="\geq " /> 1.10
  - Torch-ema <img src="https://latex.codecogs.com/svg.image?\geq&space;" title="\geq " /> 0.3
  - TensorBoardX <img src="https://latex.codecogs.com/svg.image?\geq&space;" title="\geq " /> 2.4
  - Atomic Simulation Environment (ASE) <img src="https://latex.codecogs.com/svg.image?\geq&space;" title="\geq " /> 3.21
### Setting up the environment

We recommend to use [ Miniconda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/download.html) for the creation of a virtual environment. 

Once in miniconda, you can create a virtual enviroment called *physpack* 

``` 
conda env create physpack
```
 
To activate the virtual environment use the command:

```
conda activate physpack
```
### Installation
Installation must be done in the virtual environment through pip. It is important to mention that the path where you are
working will be added to the *PYTHONPATH*, so you can import the modules from anywhere.

Install via pip:
``` 
pip install -e .
```

**BEWARE**: With this command any modification that is done to the code in the folder *physpack* will be automatically reflected 
in the modules that you import.

**NOTE**: Everytime you want to import the module you must use the following command:

```
from physpack import PhysPack
```
Then PhysPack is a function that takes some arguments.

### Examples

Currently only Formaldehyde is available as an example. To run it, you only need to do:
 ```
python run.py
```

If you want to run something different, modifications to the `config.json` are needed. To create the database, required
you need to import the *datacontainer* module from the *physpack* package as follows:

```
from physpack.src.data import DataContainer
```

Then you can create the database as follows:

```
data = DataContainer(
    data_file='data/fad_set3.db',
    data_source=[
        'data/fad.set3.58069.qmmm.mp2.avtz.npz'],
        #'data/fad_set3_source.db'],
    data_load_properties=[
        'energy', 'force', 'total_charge', 'dipole'],
    data_unit_properties={
        'energy':   'eV',
        'forces':   'eV/Ang',
        'charge':   'e',
        'dipole':   'eAng'},
    data_alt_property_labels={
        'energy':   ['V', 'E']},
    data_overwrite=False)
```

Additional examples will be added in the future.


## What needs to be added?

- [ ] Add more NN architectures
- [ ] Read parameters from older PhysNet Versions (i.e. TF1 and TF2)
- [ ] Add sampling methods:
    - [ ] MD and MC with XTB
    - [ ] Normal Model Sampling
    - [ ] Umbrella Sampling
    - [ ] Methadynamics Sampling (Kai)
    - [ ] Adaptive Sampling
- [ ] Electronic structure calculations:
   - [ ] Automatic generation of input files for common used codes (e.g. Gaussian, Orca, MOLPRO, etc.)
   - [ ] Automatic extraction of information from output files
   - [ ] Preparation of training/ input files for the NN
- [ ] Finish automatic evaluation (Luis)
