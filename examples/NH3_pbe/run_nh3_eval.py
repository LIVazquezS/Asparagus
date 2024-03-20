import os
import numpy as np

from ase import io
from ase.optimize import BFGS
from ase import vibrations
from ase.visualize import view

import matplotlib.pyplot as plt

import sys
sys.path.insert(0, '/home/toepfer/Documents/Project_PhysNet3/KaiAsparagus')
from asparagus import Asparagus
from asparagus.interface.orca_ase import ORCA_Dipole

# Asparagus Models
configs = [
    'nh3_md_orca/nh3_md.json', 
    'nh3_meta_orca/nh3_meta.json', 
    'nh3_nmscan_orca/nh3_nms.json']
labels = [
    'MD Sampling',
    'Meta-Dynamics Sampling',
    'Normal Mode Scanning']

# Iterate over models
for imodel, (config, label) in enumerate(zip(configs, labels)):
    
    # Read system
    system_model = io.read('../../data/nh3_c3v.xyz')
    system_ref = io.read('../../data/nh3_c3v.xyz')
    
    # Get calculators
    model = Asparagus(config=config)
    calc_model = model.get_ase_calculator()
    
    #calc_ref = ORCA_Dipole(
        #charge=0,
        #mult=1,
        #orcasimpleinput='RI PBE D3BJ def2-SVP def2/J TightSCF',
        #orcablocks='%pal nprocs 4 end',
        #directory='orca')

