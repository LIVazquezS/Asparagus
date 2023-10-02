import os
from asparagus import DataContainer
from asparagus import Asparagus

# From Asparagus dataset database
if False:

    if os.path.exists("config.json"): os.remove("config.json")
    data = DataContainer(
        data_file='data/diels_alder_c2h4_c4h6.db',
        data_source=[
            'dielsalder_samples/1_meta.db',
            'dielsalder_samples/4_meta.db',
            'dielsalder_samples/3_meta.db'],
        data_format=['db', 'db', 'db'],
        data_load_properties=[
            'energy', 'forces', 'charge', 'dipole'],
        data_unit_properties={
            'energy':   'eV',
            'forces':   'eV/Ang',
            'dipole':   'e*Ang'},
        data_overwrite=True)

    data = DataContainer(
        data_file='data/diels_alder_c2h4_c4h6.db')


# From ASE database
if False:

    if os.path.exists("config.json"): os.remove("config.json")
    data = DataContainer(
        data_file='data/from_ase_c4h6.db',
        data_source=[
            'data/c4h6.ase.db'],
        data_format=[
            'asedb'],
        data_load_properties=[
            'energy', 'forces', 'charge', 'dipole'],
        data_unit_properties={
            'positions':   'Bohr',
            'energy':   'eV',
            'forces':   'eV/Bohr',
            'charge':   'e',
            'dipole':   'e*Bohr'},
        data_overwrite=True)

# From NPZ database
if False:

    if os.path.exists("config.json"): os.remove("config.json")
    data = DataContainer(
        data_file='data/no2_1.db',
        data_source=[
            'data/data_NO2_1.npz'],
        data_load_properties=[
            'energy', 'charge', 'dipole'],
        data_unit_properties={
            'energy':   'eV',
            'charge':   'e',
            'dipole':   'e*Ang'},
        data_overwrite=True)