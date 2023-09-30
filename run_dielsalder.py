
from asparagus import DataContainer

from asparagus import MDSampler, NormalModeScanner, MetaSampler, ReCalculator

from asparagus import Asparagus

if False:

    from asparagus.src import utils

    utils.check_units("Debye", "e*Ang", verbose=True)
    utils.check_units("eV", "kcal/mol", verbose=True)
    utils.check_units("kcal/mol", "eV", verbose=True)
    utils.check_units("eV", "kJ/mol", verbose=True)
    utils.check_units("Hartree", "eV", verbose=True)
    utils.check_units("e*Ang", "e*Bohr", verbose=True)
    utils.check_units("kcal/mol/Ang**2", "kcal/mol/Bohr**2", verbose=True)
    utils.check_units("Hartree/Bohr**2", "kcal/mol/Ang**2", verbose=True)
    utils.check_units("Ang**3", "Bohr**3", verbose=True)
    utils.check_units("Bohr**0.5", "Ang**0.5", verbose=True)
    utils.check_units("Bohr**-0.5", "Ang**-0.5", verbose=True)
    utils.check_units("1/Bohr**0.5", "1/Ang**0.5", verbose=True)
    utils.check_units("3/2*Bohr**0.5", "3/2*Ang**0.5", verbose=True)
    utils.check_units("1.5*Bohr**0.5", "1.5*Ang**0.5", verbose=True)

if True:

    sampler = MetaSampler(
        config='diels_alder_config.json',
        sample_directory='dielsalder_samples',
        sample_systems='data/c2h4.xyz',
        sample_systems_format='xyz',
        sample_calculator='XTB',
        sample_properties=['energy', 'forces', 'dipole'],
        sample_systems_optimize=True,
        meta_cv=[[0, 1]],
        meta_hookean=[[0, 1, 4.0]],
        meta_gaussian_height=0.20,
        meta_gaussian_widths=0.2,
        meta_gaussian_interval=10,
        meta_time_step=1.0,
        meta_simulation_time=2000.0,
        meta_save_interval=10,
        meta_temperature=300,
        meta_langevin_friction=1.0,
        meta_initial_velocities=True,
        meta_initial_temperature=100.,
        )
    sampler.run()

    sampler = MetaSampler(
        config='diels_alder_config.json',
        sample_directory='dielsalder_samples',
        sample_systems='data/c4h6.xyz',
        sample_systems_format='xyz',
        sample_calculator='XTB',
        sample_properties=['energy', 'forces', 'dipole'],
        sample_systems_optimize=True,
        meta_cv=[[0, 1], [1, 2], [2, 3]],
        meta_hookean=[[0, 1, 2.0], [1, 2, 2.0], [2, 3, 2.0]],
        meta_gaussian_height=0.20,
        meta_gaussian_widths=0.2,
        meta_gaussian_interval=10,
        meta_time_step=1.0,
        meta_simulation_time=8000.0,
        meta_save_interval=20,
        meta_temperature=300,
        meta_langevin_friction=1.0,
        meta_initial_velocities=True,
        meta_initial_temperature=100.,
        )
    sampler.run()

    #sampler = MetaSampler(
        #config='diels_alder_config.json',
        #sample_directory='dielsalder_samples',
        #sample_systems='data/c6h10.xyz',
        #sample_systems_format='xyz',
        #sample_calculator='XTB',
        #sample_properties=['energy', 'forces', 'dipole'],
        #sample_systems_optimize=True,
        #meta_cv=[[0, 6], [1, 9]],
        #meta_hookean=[[0, 6, 6.0], [1, 9, 6.0]],
        #meta_gaussian_height=0.20,
        #meta_gaussian_widths=0.20,
        #meta_gaussian_interval=10,
        #meta_time_step=1.0,
        #meta_simulation_time=7990.0,
        #meta_save_interval=20,
        #meta_temperature=300,
        #meta_langevin_friction=1.0,
        #meta_initial_velocities=True,
        #meta_initial_temperature=100.,
        #)
    #sampler.run()

if True:

    data = DataContainer(
        config='diels_alder_config.json',
        data_file='data/diels_alder_c2h4_c4h6.db',
        data_source=[
            'dielsalder_samples/1_meta.db',
            'dielsalder_samples/2_meta.db',
            #'dielsalder_samples/3_meta.db'
            ],
        data_format=[
            'db',
            'db',
            #'db',
            ],
        data_load_properties=[
            'energy', 'forces', 'charge', 'dipole'],
        data_unit_properties={
            'energy':   'eV',
            'forces':   'eV/Ang',
            'dipole':   'e*Ang'},
        data_overwrite=True)

if True:

    model = Asparagus(
        config='diels_alder_config.json',
        data_file='data/diels_alder_c2h4_c4h6.db',
        model_directory='model_dielsalder',
        model_properties=['energy', 'forces', 'atomic_charges', 'dipole'],
        model_interaction_cutoff=12.0,
        model_cutoff_width=2.0,
        model_electrostatic=False,
        input_cutoff_descriptor=6.0,
        trainer_properties_weights={
            'energy': 1.,
            'forces': 20.,
            'dipole': 10.
            },
        trainer_optimizer_args={'lr': 0.001, 'weight_decay': 0.e-5},
        )
    model.train()
    model.test(
        test_datasets='all',
        test_directory=model.get('model_directory'))
