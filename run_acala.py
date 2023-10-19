from asparagus import DataContainer

from asparagus import Asparagus

import ase

if True:

    # Convert Numpy data file to Asparagus database format
    data = DataContainer(
        config='acala_config.json',
        data_file='data/acala.db',
        data_source=[
            'data/all_data_22000_acala.npz'],
        data_load_properties=[
            'energy', 'forces'],
        data_unit_properties={
            'positions':    'Ang',
            'energy':       'eV',
            'forces':       'eV/Ang',
            },
        data_alt_property_labels={
            'energy':       ['E'],
            'forces':       ['F'],
            'dipole':       ['F'],
            'charge':       ['Q'],
            'positions':    ['R'],
            'atoms_number': ['N'],
            'atomic_numbers':   ['Z'],
            },
        data_overwrite=False)
    
    # Write one sample as xyz format file for later use in ASE
    sample = data.train_set[0]
    ase_sample = ase.Atoms(
        sample['atomic_numbers'],
        positions=sample['positions']
        )
    ase.io.write("data/acala.xyz", ase_sample)
    
if True:
    
    # Initiate Asparagus and set parameters
    model = Asparagus(
        config='acala_config.json',
        data_file='data/acala.db',
        model_type='PhysNet',
        model_directory="model_acala",
        model_num_threads=8,
        model_electrostatic=False,
        model_dispersion=True,
        model_dispersion_trainable=False,
        model_interaction_cutoff=8.0,
        input_cutoff_descriptor=8.0,
        model_properties=['energy', 'forces'],
        data_num_train=2500,
        data_num_valid=250,
        data_num_test=250,
        data_train_batch_size=32,
        data_valid_batch_size=32,
        data_test_batch_size=32,
        trainer_max_epochs=10_000,
        trainer_properties_weights={'energy': 1., 'forces': 50.},
        trainer_validation_interval=1,
        )
    # Start training
    model.train()
    # Finally, apply test on all datasets (training, validation and test)
    model.test(
        test_datasets='all',
        test_directory=model.get('model_directory'))

if True:
    
    # Get the trained PhysNet model as an ASE calculator 
    model = Asparagus(
        config='acala_config.json',
        )
    calc = model.get_ase_calculator()
    
    # Read previously stored sample structure and assign ASE calculator
    acala_atoms = ase.io.read("data/acala.xyz")
    acala_atoms.calc = calc
    
    # Perform structure optimization and show results
    from ase.optimize import BFGS
    acala_atoms_list = [acala_atoms.copy()]
    dyn = BFGS(acala_atoms)
    dyn.run(fmax=0.01)
    acala_atoms_list.append(acala_atoms)
    
    from ase.visualize import view
    view(acala_atoms_list)
    
