Usage
=====

.. _installation:

Installation
------------

- Clone the repository
- Requirements:
   - Python <img src="https://latex.codecogs.com/svg.image?\geq&space;" title="\geq " /> 3.8 (I recommend to use 3.8.12, **DO NOT** use 3.9)
   - PyTorch <img src="https://latex.codecogs.com/svg.image?\geq&space;" title="\geq " /> 1.10
   - Torch-ema <img src="https://latex.codecogs.com/svg.image?\geq&space;" title="\geq " /> 0.3
   - TensorBoardX <img src="https://latex.codecogs.com/svg.image?\geq&space;" title="\geq " /> 2.4
   - Atomic Simulation Environment (ASE) <img src="https://latex.codecogs.com/svg.image?\geq&space;" title="\geq " /> 3.21

Setting up the environment

We recommend to use [Miniconda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/download.html) for the creation of a virtual environment.

Once in miniconda, you can create a virtual enviroment called *asparagus*

.. code-block:: console

   conda env create asparagus

To activate the virtual environment use the command:

.. code-block:: console

   conda activate asparagus

Installation must be done in the virtual environment through pip. It is important to mention that the path where you are
working will be added to the *PYTHONPATH*, so you can import the modules from anywhere.

Install via pip:

.. code-block:: console

   pip install -e .

**BEWARE**: With this command any modification that is done to the code in the folder *physpack* will be automatically reflected
in the modules that you import.

**NOTE**: Everytime you want to import the module you must use the following command:

.. code-block:: python

   from physpack import PhysPack

Then PhysPack is a function that takes some arguments.





