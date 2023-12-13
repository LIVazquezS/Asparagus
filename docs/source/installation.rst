Installation
===================================


Asparagus depends on few dependencies. We recommend that you create a virtual environment to install Asparagus.


Setting up the environment
==========================

We recommend to use [Mamba](https://mamba.readthedocs.io/en/latest/user_guide/mamba.html) for the creation of a virtual environment.
Or in case [Miniconda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/download.html).

Once in mamba, you can create a virtual enviroment called *asparagus*

```
mamba create --name asparagus python=3.8
```

To activate the virtual environment use the command:

```
mamba activate asparagus
```
 **Note**: If you are using Miniconda, just replace `mamba` by `conda`.

Dependencies
============

Asparagus depends on the following packages:

   - Python <img src="https://latex.codecogs.com/svg.image?\geq&space;" title="\geq " /> 3.8 (We recommend to use 3.8.12, **DO NOT** (for the moment) use 3.9)
   - PyTorch <img src="https://latex.codecogs.com/svg.image?\geq&space;" title="\geq " /> 1.10
   - Torch-ema <img src="https://latex.codecogs.com/svg.image?\geq&space;" title="\geq " /> 0.3
   - TensorBoardX <img src="https://latex.codecogs.com/svg.image?\geq&space;" title="\geq " /> 2.4
   - Atomic Simulation Environment (ASE) <img src="https://latex.codecogs.com/svg.image?\geq&space;" title="\geq " /> 3.21
   - xtb

Installation
============

To install Asparagus first clone the repository:

```
git clone https://github.com/LIVazquezS/Asparagus
```

Then, go to the folder where you cloned the repository and install via pip:

```
pip install -e .
```

Alternatively, install via setup.py:

```
python setup.py install
```

**BEWARE**: With this command any modification that is done to the code in the folder *asparagus* will be automatically reflected
in the modules that you import.


