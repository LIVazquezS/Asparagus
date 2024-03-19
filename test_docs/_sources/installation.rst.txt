Installation
===================================


Asparagus depends on few dependencies. We recommend that you create a virtual environment to install Asparagus.


Setting up the environment
--------------------------

We recommend to use `mamba`_ for the creation of a virtual environment.
Or in case `miniconda`_.

.. _mamba: https://mamba.readthedocs.io/en/latest/user_guide/mamba.html
.. _miniconda: https://docs.conda.io/projects/conda/en/latest/user-guide/install/download.html

Once in mamba, you can create a virtual enviroment called *asparagus*

.. code-block:: bash

      mamba create --name asparagus python=3.8

To activate the virtual environment use the command:

.. code-block:: bash

       mamba activate asparagus

**Note**: If you are using Miniconda, just replace ``mamba`` by ``conda``.

Dependencies
--------------

Asparagus depends on the following packages:

   - Python :math:`\geq` 3.8 (We recommend to use 3.8.12, **DO NOT** (for the moment) use 3.9)
   - PyTorch :math:`\geq` 1.10
   - Torch-ema :math:`\geq` 0.3
   - TensorBoard :math:`\geq` 2.4
   - Atomic Simulation Environment (ASE)  :math:`\geq` 3.21
   - xtb

Installation
-------------

To install Asparagus first clone the repository:

.. code-block:: bash

      git clone https://github.com/LIVazquezS/Asparagus

Then, go to the folder where you cloned the repository and install via pip:

.. code-block:: bash

      pip install -e .

Alternatively, install via ``setup.py``:

.. code-block:: bash

      python setup.py install

**BEWARE**: With this command any modification that is done to the code in the folder *asparagus* will be automatically reflected
in the modules that you import.


