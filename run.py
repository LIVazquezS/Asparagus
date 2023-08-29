
from asparagus import DataContainer

from asparagus import NormalModeScanner, MDSampler, MetaSampler

from asparagus import Asparagus

if False:

    data = DataContainer(
        data_file='data/fad_set3.db',
        data_source=[
            'data/fad.set3.58069.qmmm.mp2.avtz.npz'],
        data_load_properties=[
            'energy', 'force', 'total_charge', 'dipole'],
        data_unit_properties={
            'energy':   'eV',
            'forces':   'eV/Ang',
            'charge':   'e',
            'dipole':   'eAng'},
        data_alt_property_labels={
            'energy':   ['V', 'E']},
        data_overwrite=False)

if True:

    #sampler = NormalModeScanner(
        #sample_directory='test_samples',
        #sample_systems='data/hono.xyz',
        #sample_systems_format='xyz',
        #sample_systems_optimize=True,
        #sample_systems_optimize_fmax=0.001,
        #)
    #sampler.run()

    #sampler = MDSampler(
        #sample_directory='test_samples',
        #sample_systems='data/hono.xyz',
        #sample_systems_format='xyz',
        #)
    #sampler.run()
    
    #sampler = MetaSampler(
        #sample_directory='test_samples',
        #sample_systems='data/hono.xyz',
        #sample_systems_format='xyz',
        #meta_cv=[[2, 1], [2, 3], [2, 0]],
        #meta_hookean=[[2, 1, 4.0], [2, 3, 4.0], [2, 0, 4.0]]
        #)
    #sampler.run()
    
    sampler = MetaSampler(
        sample_directory='test_samples',
        sample_systems='data/co2.xyz',
        sample_systems_format='xyz',
        meta_cv=[[0, 1, 2]],
        meta_gaussian_height=0.01,
        meta_gaussian_widths=0.05,
        meta_time_step=0.1,
        meta_simulation_time=1.0E5,
        meta_save_interval=100,
        meta_initial_velocities=True,
        meta_initial_temperature=300.
        )
    sampler.run()
    
if False:

    model = Asparagus(
        data_file='data/h2co_b3lyp.db',
        data_source=['data/h2co_B3LYP_cc_pVDZ_4001.npz'],
        data_load_properties=['energy', 'force', 'dipole'],
        model_properties=['energy', 'forces', 'atomic_charges', 'dipole'],
        output_properties=['energy', 'forces', 'atomic_charges', 'dipole'],
        trainer_optimizer_args={'lr': 0.0001},
        data_container=None)

    model.train()
