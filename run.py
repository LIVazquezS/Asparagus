
from asparagus import DataContainer

from asparagus import Sampler, NormalModeScanner

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

    sampler = NormalModeScanner(
        sample_directory='test_samples',
        sample_systems='data/hono.xyz',
        sample_systems_format='xyz',
        nms_systems_optimize=True,
        nms_systems_optimize_fmax=0.001,
        )
    sampler.run()

    #sampler = Sampler(
        #sample_directory='test_samples',
        #sample_systems='data/hono.xyz',
        #sample_systems_format='xyz',
        #sample_systems_optimize=True,
        #sample_systems_optimize_fmax=0.0001,
        #)
    #sampler.normal_mode_sampling()
    #sampler.normal_mode_sampling(
        #sample_file='nms1.traj',        # Trajectory or DB file to store samples
        #sample_step_size=1.0,           # Normal mode step size. Within harmonic
        ## approximation, step size 1 should increase potential by frequency level
        ## OR
        #sample_step_energy=0.5,         # Within harmonic approximation, scale
        ## normal mode, that step size leads to, e.g., 0.5eV energy difference
        #sample_potential_cut=5.0,       # Potential cut for mode elongation (eV)
        #sample_max_structures=1000,     # Maximum number of structures, sorted by E
        #sample_cross_modes=1,           # Combine different number of modes
        #)


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
