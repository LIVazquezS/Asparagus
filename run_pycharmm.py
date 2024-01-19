# Test Script to import PhysNet as energy function in CHARMM via PyCHARMM

# Basics
import os
import sys
import ctypes
import pandas
import numpy as np

# ASE
from ase import Atoms
from ase import io
import ase.units as units

# PyCHARMM
import pycharmm
import pycharmm.generate as gen
import pycharmm.ic as ic
import pycharmm.coor as coor
import pycharmm.energy as energy
import pycharmm.dynamics as dyn
import pycharmm.nbonds as nbonds
import pycharmm.minimize as minimize
import pycharmm.crystal as crystal
import pycharmm.image as image
import pycharmm.psf as psf
import pycharmm.read as read
import pycharmm.write as write
import pycharmm.settings as settings
import pycharmm.lingo as stream
import pycharmm.select as select
import pycharmm.shake as shake
import pycharmm.cons_fix as cons_fix
import pycharmm.cons_harm as cons_harm
from pycharmm.lib import charmm as libcharmm
import pycharmm.lib as lib

# Asparagus
from asparagus import Asparagus

# Step 0: Load parameter files
#-----------------------------------------------------------

stream.charmm_script("""
! protein topology and parameter
open read card unit 10 name charmm_data/nh3_water.top
read  rtf card unit 10

open read card unit 20 name charmm_data/nh3_water.par
read para card unit 20 flex
""")

settings.set_bomb_level(-2)
settings.set_warn_level(-1)

# Step 1: Generate System
#-----------------------------------------------------------

solvation = True
setup = 1

if setup == 1:

    # Read ML system
    read.psf_card("charmm_data/ammonia.psf")
    read.pdb("charmm_data/ammonia.pdb")

    # Manipulate coordinates
    pos = coor.get_positions().to_numpy(dtype=np.float32)
    pos[:, 0] -= 10.0 # Shift ammonia to the cell border to test pbc
    pandas_pos = pandas.DataFrame(
        {'x': pos[:, 0], 'y': pos[:, 1], 'z': pos[:, 2]}
        )
    coor.set_positions(pandas_pos)

    # Read solvent
    if solvation:

        # Add TIP3P water residues
        read.sequence_pdb("charmm_data/water_box.pdb") 
        add_water = """
        generate WAT setup noang nodihe
        open read card unit 10 name charmm_data/water_box.pdb
        read coor pdb  unit 10 resid
        """
        stream.charmm_script(add_water)
        
        # Delete overlapping water molecules
        remove_water = """
        delete atom select ( .byres. ( (segid AMM1 .around. 2.0 ) -
            .and. (segid WAT .and. type OH2 ))) end
        """
        stream.charmm_script(remove_water)

elif setup == 2:

    # Read solvent
    if solvation:

        # Add TIP3P water residues
        add_water = """
        open read card unit 10 name charmm_data/water_box.pdb
        read sequence pdb unit 10
        generate WAT setup noang nodihe

        open read card unit 10 name charmm_data/water_box.pdb
        read coor pdb  unit 10 resid
        """
        stream.charmm_script(add_water)


    # Read ML system
    add_ammonia = """
    open read card unit 10 name charmm_data/ammonia.pdb
    read sequence pdb unit 10
    generate AMM1 setup warn first none last none

    open read card unit 10 name charmm_data/ammonia.pdb
    read coor pdb  unit 10 resid
    """
    stream.charmm_script(add_ammonia)

    # Manipulate coordinates
    pos = coor.get_positions().to_numpy(dtype=np.float32)
    pos[-4:, 0] -= 10.0 # Shift ammonia to the cell border to test pbc
    pandas_pos = pandas.DataFrame(
        {'x': pos[:, 0], 'y': pos[:, 1], 'z': pos[:, 2]}
        )
    coor.set_positions(pandas_pos)

    if solvation:
        
        # Delete overlapping water molecules
        remove_water = """
        delete atom select ( .byres. ( (segid AMM1 .around. 2.0 ) -
            .and. (segid WAT .and. type OH2 ))) end
        """
        stream.charmm_script(remove_water)



write.coor_pdb("charmm_data/ammonia_water.pdb", title="Ammonia solvated")
write.psf_card("charmm_data/ammonia_water.psf", title="Ammonia solvated")

# ASE atoms object (just to get atomic numbers)
ase_ammonia = io.read("charmm_data/ammonia.pdb", format="proteindatabank")

# Step 2: CHARMM Setup
#-----------------------------------------------------------

# Non-bonding parameter
dict_nbonds = {
    'atom': True,
    'vdw': True,
    'vswitch': True,
    'cutnb': 14,
    'ctofnb': 12,
    'ctonnb': 10,
    'cutim': 14,
    #'lrc': True,
    'inbfrq': -1,
    'imgfrq': -1
    }
nbond = pycharmm.NonBondedScript(**dict_nbonds)
nbond.run()

if solvation:
    
    # PBC box
    crystal.define_cubic(length=30.0)
    crystal.build(cutoff=14.0)

    stream.charmm_script('image byres xcen 0.0 ycen 0.0 zcen 0.0 sele all end')

    # H-bonds constraint
    #shake.on(bonh=True, tol=1e-7)
    stream.charmm_script('shake bonh para sele resname TIP3 end')

else:

    # Default Setup
    pass

# Energy
energy.show()

# Step 3: Asparagus Setup
#-----------------------------------------------------------

# Load Asparagus model
ml_model = Asparagus(config='nh3_config_nms.json')

# Get atomic number from ASE atoms object
ml_Z = ase_ammonia.get_atomic_numbers()

# Prepare PhysNet input parameter
ml_selection = pycharmm.SelectAtoms(seg_id='AMM1')

# Initialize PhysNet calculator
calc = pycharmm.MLpot(
    ml_model,
    ml_Z,
    ml_selection,
    ml_charge=0,
    ml_fq=True,
)

# Custom energy
energy.show()

# Step 4: Minimization
#-----------------------------------------------------------

if True:
    
    # Fix ML atoms
    cons_fix.setup(pycharmm.SelectAtoms(seg_id='AMM1'))

    # Optimization with PhysNet parameter
    minimize.run_sd(**{
        'nstep': 500,
        'nprint': 10,
        'tolenr': 1e-5,
        'tolgrd': 1e-5})

    # Unfix ML atoms
    cons_fix.turn_off()
    
    # Optimization with PhysNet parameter
    minimize.run_sd(**{
        'nstep': 500,
        'nprint': 10,
        'tolenr': 1e-5,
        'tolgrd': 1e-5})

    # Write pdb file
    write.coor_pdb("charmm_data/mini_ammonia.pdb", title="Mini SD")

else:
    
    # Read optimized coordinates
    read.pdb("charmm_data/mini_ammonia.pdb")
    

# Step 5: Heating - CHARMM, PhysNet
#-----------------------------------------------------------

if True:
    
    timestep = 0.0005   # 0.5 fs

    res_file = pycharmm.CharmmFile(
        file_name='charmm_data/heat.res', file_unit=2, 
        formatted=True, read_only=False)
    dcd_file = pycharmm.CharmmFile(
        file_name='charmm_data/heat.dcd', file_unit=1, 
        formatted=False, read_only=False)

    # Run some dynamics
    dynamics_dict = {
        'leap': False,
        'verlet': True,
        'cpt': False,
        'new': False,
        'langevin': False,
        'timestep': timestep,
        'start': True,
        'nstep': 10.*1./timestep,
        'nsavc': 0.01*1./timestep,
        'nsavv': 0,
        'inbfrq':-1,
        'ihbfrq': 50,
        'ilbfrq': 50,
        'imgfrq': 50,
        'ixtfrq': 1000,
        'iunrea':-1,
        'iunwri': res_file.file_unit,
        'iuncrd': dcd_file.file_unit,
        'nsavl':  0,  # frequency for saving lambda values in lamda-dynamics
        'iunldm':-1,
        'ilap': -1,
        'ilaf': -1,
        'nprint': 100, # Frequency to write to output
        'iprfrq': 500, # Frequency to calculate averages
        'isvfrq': 1000, # Frequency to save restart file
        'ntrfrq': 1000,
        'ihtfrq': 200,
        'ieqfrq': 1000,
        'firstt': 100,
        'finalt': 300,
        'tbath': 300,
        'iasors': 0,
        'iasvel': 1,
        'ichecw': 0,
        'iscale': 0,  # scale velocities on a restart
        'scale': 1,  # scaling factor for velocity scaling
        'echeck':-1}

    dyn_heat = pycharmm.DynamicsScript(**dynamics_dict)
    dyn_heat.run()
    
    res_file.close()
    dcd_file.close()

# Step 5: NVE - CHARMM, PhysNet
#-----------------------------------------------------------

if True:
    
    timestep = 0.00025   # 0.2 fs

    str_file = pycharmm.CharmmFile(
        file_name='charmm_data/heat.res', file_unit=3, 
        formatted=True, read_only=False)
    res_file = pycharmm.CharmmFile(
        file_name='charmm_data/nve.res', file_unit=2, 
        formatted=True, read_only=False)
    dcd_file = pycharmm.CharmmFile(
        file_name='charmm_data/nve.dcd', file_unit=1, 
        formatted=False, read_only=False)

    # Run some dynamics
    dynamics_dict = {
        'leap': False,
        'verlet': True,
        'cpt': False,
        'new': False,
        'langevin': False,
        'timestep': timestep,
        'start': False,
        'restart': True,
        'nstep': 50.*1./timestep,
        'nsavc': 0.001*1./timestep,
        'nsavv': 0,
        'inbfrq':-1,
        'ihbfrq': 50,
        'ilbfrq': 50,
        'imgfrq': 50,
        'ixtfrq': 1000,
        'iunrea': str_file.file_unit,
        'iunwri': res_file.file_unit,
        'iuncrd': dcd_file.file_unit,
        'nsavl':  0,  # frequency for saving lambda values in lamda-dynamics
        'iunldm':-1,
        'ilap': -1,
        'ilaf': -1,
        'nprint': 1, # Frequency to write to output
        'iprfrq': 500, # Frequency to calculate averages
        'isvfrq': 1000, # Frequency to save restart file
        'ntrfrq': 0,
        'ihtfrq': 0,
        'ieqfrq': 0,
        'firstt': 300,
        'finalt': 300,
        'tbath': 300,
        'iasors': 0,
        'iasvel': 1,
        'ichecw': 0,
        'iscale': 0,  # scale velocities on a restart
        'scale': 1,  # scaling factor for velocity scaling
        'echeck':-1}

    dyn_nve = pycharmm.DynamicsScript(**dynamics_dict)
    dyn_nve.run()

    str_file.close()
    res_file.close()
    dcd_file.close()
