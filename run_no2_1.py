
from asparagus import DataContainer

from asparagus import MDSampler, NormalModeScanner, MetaSampler, ReCalculator

from asparagus import Asparagus

if False:

    data = DataContainer(
        config='no2_config.json',
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
        data_overwrite=True)

if True:

    model = Asparagus(
        config='no2_config.json',
        data_file='data/no2_1.db',
        model_directory="model_NO2_1",
        model_interaction_cutoff=12.0,
        model_cutoff_width=2.0,
        input_cutoff_descriptor=6.0,
        model_properties=['energy', 'atomic_charges', 'dipole'],
        output_properties=['energy','atomic_charges', 'dipole'],
        trainer_max_epochs=100_000,
        trainer_properties_weights={'energy': 50., 'dipole': 1.}
        )
    model.train()
    model.test(
        test_datasets='all',
        test_directory=model.get('model_directory'))
if True:
    
    import ase
    model = Asparagus(
        config="model_NO2_1/config.json",
        model_directory="model_NO2_1"
        )

    print("\n\n1")
    calc = model.get_ase_calculator()
    no2_atoms = ase.io.read("data/no2.xyz")
    print(calc.calculate(no2_atoms))
    print(calc.calculate(no2_atoms))

    print("\n\n2")
    calc = model.get_ase_calculator()
    no2_atoms_copy = ase.io.read("data/no2.xyz")
    print(calc.calculate([no2_atoms, no2_atoms_copy]))

    print("\n\n3")
    calc = model.get_ase_calculator(atoms=no2_atoms)
    print(calc.calculate())
    print(calc.calculate(no2_atoms))

    print("\n\n4")
    calc = model.get_ase_calculator(atoms=no2_atoms)
    print(calc.calculate())
    print(calc.calculate([no2_atoms, no2_atoms_copy]))
    print(calc.calculate())

    print("\n\n5")
    no2_atoms = ase.io.read("data/no2.xyz")
    calc = model.get_ase_calculator(atoms_charge=0.0)
    print(calc)
    no2_atoms.calc = calc
    print(no2_atoms)
    print(calc.calculate())
    print(no2_atoms.get_potential_energy())
    print(no2_atoms.get_forces())
    from ase.optimize import BFGS
    no2_atoms_list = [no2_atoms.copy()]
    dyn = BFGS(no2_atoms)
    dyn.run(fmax=0.01)
    no2_atoms_list.append(no2_atoms)
    from ase.visualize import view
    view(no2_atoms_list)


