
from asparagus import DataContainer

from asparagus import MDSampler, NormalModeScanner, MetaSampler, ReCalculator

from asparagus import Asparagus

if False:

    data = DataContainer(
        data_file='data/no2_1.db',
        data_source=[
            'data/data_NO2_1.npz'],
        data_load_properties=[
            'energy', 'total_charge', 'dipole'],
        data_unit_properties={
            'energy':   'eV',
            'charge':   'e',
            'dipole':   'eAng'},
        data_alt_property_labels={
            'energy':   ['V', 'E']},
        data_overwrite=False)

if False:

    model = Asparagus(
        data_file='data/no2_1.db',
        model_interaction_cutoff=20.0,
        model_cutoff_width=5.0,
        input_cutoff_descriptor=14.0,
        model_properties=['energy', 'atomic_charges', 'dipole'],
        output_properties=['energy','atomic_charges', 'dipole'],
        trainer_max_epochs=100_000,
        trainer_properties_weights={'energy': 100., 'dipole': 1.}
        )
    #model.train()
    model.test(
        test_datasets='all',
        test_directory=model.get('model_directory'))
    
if True:
    
    import ase
    model = Asparagus(
        config="config.json"
        )

    #calc = model.get_ase_calculator()
    no2_atoms = ase.io.read("data/no2.xyz")
    #calc.calculate(no2_atoms)
    #calc.calculate(no2_atoms)

    #calc = model.get_ase_calculator()
    no2_atoms_copy = ase.io.read("data/no2.xyz")
    #calc.calculate([no2_atoms, no2_atoms_copy])
    
    #calc = model.get_ase_calculator(atoms=no2_atoms)
    #calc.calculate()
    #calc.calculate(no2_atoms)
    
    calc = model.get_ase_calculator(atoms=[no2_atoms, no2_atoms_copy])
    calc.calculate()
    calc.calculate([no2_atoms, no2_atoms_copy])
    
    
    
