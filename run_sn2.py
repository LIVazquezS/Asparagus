from asparagus import DataContainer

from asparagus import MDSampler, NormalModeScanner, MetaSampler

from asparagus import Asparagus

if True:
# Sampling the TS, C-O and C-Br bond lengths are sampled.
    sampler = MetaSampler(
        config='sn2_config.json',
        sample_directory='sn2_samples',
        sample_systems=['data_sn2/fs_min.xyz'],
        sample_systems_format='xyz',
        sample_calculator='XTB',
        sample_properties=['energy', 'forces', 'dipole'],
        sample_systems_optimize=False,
        meta_cv=[[0,4],[0,5]],
        meta_hookean=[[0, 4, 8.0],[0,5,8.0]],
        meta_gaussian_height=0.05,
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

if False:

# This part samples the initial structures for the sn2 reaction, only methanol and bromomethane are sampled
#Methanol
    sampler_met = MDsampler(
        sample_directory='sn2_samples',
        sample_systems='data/sn2_methanol.xyz',
        sample_systems_format='xyz')

    sampler_met.run()

#Bromomethane
    sampler_init = MDsampler(
        sample_directory='sn2_samples',
        sample_systems='data/sn2_init_br.xyz',
        sample_systems_format='xyz')

    sampler_init.run()



