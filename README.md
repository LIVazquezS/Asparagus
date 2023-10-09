# Project Asparagus

**Authors**: K. Toepfer,L.I. Vazquez-Salazar

## What is this?
 - A refined implementation of PhysNet NN (and other atomistic NN to come) in PyTorch. 
 - A Suit for the automatic construction of Potential Energy Surface (PES) from sampling to production.

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
conda create --name physpack python=3.8
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

**NOTE**: Everytime you want to import the module, you must use the following command:

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

If you want to evaluate the generated model, by default, the code will look for the 'best' checkpoint inside the 
folder generated after training. Remember that the folder name starts with the date and time. 

To do the evaluation, you need to add the following line to your run file:
 ```
 model.test_model()
 ```

By default, the code will show you the MAE and RMSE for the energy, forces and dipole. 

There are a few keywords that you can add to the function 'test_model' to change the behaviour of the evaluation.
If you add the option `plots=True`, the code will make a scatter plot with the predicted and the reference values.
The option `show_plots=True` will show the generated scatter plot. If you want to save it, you need to add the flag
`save_plots=True`. Other options for plotting are residuals and histograms that require the keywords `residual_plots=True,
show_residuals=True` and the same for histograms `histogram_plots=True, show_histograms=True`. It is important to mention
that by default, only plots of the energy are produced; however, if you want other properties, you can use the keyword:
`plots_to_show` and pass a list of the properties that you want to plot. The special keyword `all` will plot all the 
properties in the database.

If you want to save the generated data for later, you can do it as `.csv` or `.npz` files. To do so, you need to add
keyboard `save_csv=True` or `save_npz=True`. The default name for the files is `test_vals.csv` and `test_vals.npz` respectively.
By default, the code will save the files in the folder `test_results'.


Additional examples will be added in the future.


## What needs to be added?

- [ ] Add more NN architectures (Low priority)
- [x] Read parameters from older PhysNet Versions (i.e. TF1 and TF2) (Luis)
- [ ] Add sampling methods:
    - [x] MD with XTB
    - [x] MC with XTB
    - [x] Normal Model Sampling (Vanilla with random generation) 
    - [x] Normal Model Scanning 
    - [ ] Umbrella Sampling (Low priority)
    - [x] Metadynamics Sampling 
- [ ] Electronic structure calculations:
   - [x] ASE calculator (As good as it can be)
   - [ ] Automatic generation of input files for commonly used codes (e.g. Gaussian, Orca, MOLPRO, etc.)
   - [ ] Automatic extraction of information from output files
   - [ ] Preparation of training/ input files for the NN
- Trainer class:
  - [ ] Training of model ensemble 
- Tester class: 
  - [x] Finish automatic evaluation (Luis)
- Active learning
   - [ ] Adaptive Sampling
   - [ ] Uncertainty calculations (?)
     - [x] Model ensemble via ASE calculator  
- Tools class:
  - [x] Normal mode calculation (Luis)
  - [ ] Instantaneous normal mode calculation
  - [ ] Minimum energy path and Minimum dynamic path
  - [ ] Others(?)
- Production: 
  - [ ] PyCharmm 
  - [x] ASE calculator for dynamics
- Documentation:
  - [ ] Improve documentation
  - [ ] Add examples
  - [ ] Add tutorials
  
