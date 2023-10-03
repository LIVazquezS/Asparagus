from asparagus import DataContainer

from asparagus import Asparagus

if False:

    data = DataContainer(
        config='form_config.json',
        data_file='data/h2co_b3lyp.db',
        data_source=[
            'data/h2co_B3LYP_cc-pVDZ_4001.npz'],
        data_load_properties=[
            'energy', 'forces', 'charge', 'dipole'],
        data_unit_properties={
            'positions':    'Ang',
            'energy':       'eV',
            'forces':       'eV/Ang',
            'charge':       'e',
            'dipole':       'e*Ang'
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
        source_unit_properties={
            'positions':    'Ang',
            'energy':       'eV',
            'forces':       'eV/Ang',
            'charge':       'e',
            'dipole':       'e*Ang',
            },
        data_overwrite=True)

if True:

    model = Asparagus(
        config='form_config.json',
        data_file='data/h2co_b3lyp.db',
        model_directory="model_formaldehyde",
        model_interaction_cutoff=8.0,
        model_cutoff_width=2.0,
        input_cutoff_descriptor=8.0,
        model_properties=['energy', 'forces', 'atomic_charges', 'dipole'],
        trainer_max_epochs=50_000,
        trainer_properties_weights={
            'energy': 1., 'forces': 50., 'dipole': 100.}
        )
    model.train(
        trainer_validation_interval=10,
        )
    #model.test(
        #test_datasets='all',
        #test_directory=model.get('model_directory'))
    model.test(
        test_datasets='test',
        test_directory=model.get('model_directory'))

if False:

    model = Asparagus(
        config="form_config.json"
        )

    data = DataContainer(
        config='form_config.json')
    reference = data.dataset.get(1)

    import ase
    system = ase.Atoms(
        reference['atomic_numbers'],
        positions=reference['positions'])

    calc = model.get_ase_calculator()
    system.calc = calc
    results = calc.calculate()

    print("Model prediction:")
    print(results)
    print("Reference:")
    print(reference)

    from ase.optimize import BFGS
    form_list = [system.copy()]
    dyn = BFGS(system)
    dyn.run(fmax=0.01)
    form_list.append(system)
    from ase.visualize import view
    view(form_list)
