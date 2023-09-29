from asparagus import DataContainer

from asparagus import Asparagus

if True:

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

if True:

    model = Asparagus(
        config='acala_config.json',
        data_file='data/acala.db',
        model_directory="model_acala",
        model_interaction_cutoff=12.0,
        model_cutoff_width=2.0,
        input_cutoff_descriptor=8.0,
        model_properties=['energy', 'forces'],
        trainer_max_epochs=10_000,
        trainer_properties_weights={'energy': 1., 'force': 50.}
        )
    model.train()
    model.test(
        test_datasets='all',
        test_directory=model.get('model_directory'))
