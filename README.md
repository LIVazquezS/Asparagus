# Project Asparagus

**Authors**: K. Toepfer, L.I. Vazquez-Salazar

![alt text](https://github.com/LIVazquezS/Asparagus/blob/main/logo_low.png?raw=true)

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
  - xtb
### Setting up the environment

We recommend to use [Mamba](https://mamba.readthedocs.io/en/latest/user_guide/mamba.html) for the creation of a virtual environment. 

Once in mamba, you can create a virtual enviroment called *asparagus* 

``` 
mamba create --name asparagus python=3.8
```
 
To activate the virtual environment use the command:

```
mamba activate asparagus
```

### Installation
Installation must be done in the virtual environment through pip. It is important to mention that the path where you are
working will be added to the *PYTHONPATH*, so you can import the modules from anywhere.

Install via pip:
``` 
pip install -e .
```
Alternatively, install via setup.py:
``` 
python setup.py install
```

**BEWARE**: With this command any modification that is done to the code in the folder *asparagus* will be automatically reflected 
in the modules that you import.

**NOTE**: Everytime you want to import the module, you must use the following command:

```
from asparagus import Asparagus
```
Then Asparagus is a function that takes some arguments.

[//]: # (### Examples)

[//]: # ()
[//]: # (Currently only Formaldehyde is available as an example. To run it, you only need to do:)

[//]: # ( ```)

[//]: # (python run_formaldehyde.py)

[//]: # (```)

[//]: # ()
[//]: # (If you want to run something different, modifications to the `config.json` &#40;in this example `form_config.json`&#41; are needed. To create the database, required)

[//]: # (you need to import the *datacontainer* module from the *asparagus* package as follows:)

[//]: # ()
[//]: # (```)

[//]: # (from asparagus import DataContainer)

[//]: # (```)

[//]: # ()
[//]: # (Then you can create the database as follows:)

[//]: # ()
[//]: # (```)

[//]: # (data = DataContainer&#40;)

[//]: # (    data_file='data/fad_set3.db',)

[//]: # (    data_source=[)

[//]: # (        'data/fad.set3.58069.qmmm.mp2.avtz.npz'],)

[//]: # (        #'data/fad_set3_source.db'],)

[//]: # (    data_load_properties=[)

[//]: # (        'energy', 'force', 'total_charge', 'dipole'],)

[//]: # (    data_unit_properties={)

[//]: # (        'energy':   'eV',)

[//]: # (        'forces':   'eV/Ang',)

[//]: # (        'charge':   'e',)

[//]: # (        'dipole':   'eAng'},)

[//]: # (    data_alt_property_labels={)

[//]: # (        'energy':   ['V', 'E']},)

[//]: # (    data_overwrite=False&#41;)

[//]: # (```)

[//]: # ()
[//]: # (If you want to evaluate the generated model, by default, the code will look for the 'best' checkpoint inside the )

[//]: # (folder generated after training. Remember that the folder name starts with the date and time. )

[//]: # ()
[//]: # (To do the evaluation, you need to add the following line to your run file:)

[//]: # ( ```)

[//]: # ( model.test&#40;&#41;)

[//]: # ( ```)

[//]: # ()
[//]: # (By default, the code will show you the MAE and RMSE for the energy, forces and dipole. )

[//]: # ()
[//]: # (There are a few keywords that you can add to the function 'test' to change the behaviour of the evaluation.)

[//]: # ()
[//]: # (**OUTDATED**:)

[//]: # (If you add the option `plots=True`, the code will make a scatter plot with the predicted and the reference values.)

[//]: # (The option `show_plots=True` will show the generated scatter plot. If you want to save it, you need to add the flag)

[//]: # (`save_plots=True`. Other options for plotting are residuals and histograms that require the keywords `residual_plots=True,)

[//]: # (show_residuals=True` and the same for histograms `histogram_plots=True, show_histograms=True`. It is important to mention)

[//]: # (that by default, only plots of the energy are produced; however, if you want other properties, you can use the keyword:)

[//]: # (`plots_to_show` and pass a list of the properties that you want to plot. The special keyword `all` will plot all the )

[//]: # (properties in the database.)

[//]: # ()
[//]: # (If you want to save the generated data for later, you can do it as `.csv` or `.npz` files. To do so, you need to add)

[//]: # (keyboard `save_csv=True` or `save_npz=True`. The default name for the files is `test_vals.csv` and `test_vals.npz` respectively.)

[//]: # (By default, the code will save the files in the folder `test_results'.)

## Documentation

Please check our documentation [here](http://asparagus-bundle.readthedocs.io/en/latest/)

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
- Trainer class:
  - [ ] Training of model ensemble 
- Tester class: 
  - [x] Finish automatic evaluation 
- Active learning
   - [ ] Adaptive Sampling
   - [ ] Uncertainty calculations
     - [x] Model ensemble via ASE calculator 
     - [ ] Deep Evidential Regression (Low priority)
- Tools class:
  - [x] Normal mode calculation (Luis)
  - [x] Minimum energy path and Minimum dynamic path
  - [ ] Diffusion MonteCarlo
  - [ ] Others(?)
- Production: 
  - [x] PyCharmm 
  - [x] ASE calculator for dynamics
- Documentation:
  - [X] Improve documentation
  - [ ] Add examples
  - [ ] Add tutorials
  
