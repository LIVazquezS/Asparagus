from asparagus import DataContainer

from asparagus import Asparagus

if False:

    data = DataContainer(
        config='acala_config.json',
        data_file='data/acala.db',
        data_source=[
            'data/all_data_22000_acala.npz'],
        data_train_batch_size=32,
        data_valid_batch_size=32,
        data_test_batch_size=32,
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
        data_overwrite=True)

if True:

    model = Asparagus(
        config='acala_config.json',
        data_file='data/acala.db',
        model_directory="model_acala",
        model_num_threads=8,
        model_electrostatic=False,
        model_interaction_cutoff=6.0,
        model_cutoff_width=1.0,
        input_cutoff_descriptor=6.0,
        model_properties=['energy', 'forces'],
        data_num_train=1000,
        data_num_valid=100,
        data_num_test=100,
        data_train_batch_size=32,
        data_valid_batch_size=32,
        data_test_batch_size=32,
        trainer_max_epochs=10_000,
        trainer_properties_weights={'energy': 1., 'forces': 50.},
        trainer_validation_interval=1,
        )
    model.train()
    model.test(
        test_datasets='test',
        test_directory=model.get('model_directory'))
