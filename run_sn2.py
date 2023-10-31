from asparagus import DataContainer

from asparagus import MDSampler, NormalModeScanner, MetaSampler

from asparagus import Asparagus

if True:

    model = Asparagus(config='sn2_config.json',data_file='sn2.db',data_source=['md.db','meta.db'])
    model.train()

# if False:
# # Sampling the TS, C-O and C-Br bond lengths are sampled.
# #
#     sampler = MetaSampler(
#         config='sn2_config.json',
#         sample_directory='sn2_samples',
#         sample_systems=['data_sn2/fs_min.xyz'],
#         sample_systems_format='xyz',
#         sample_calculator='XTB',
#         sample_properties=['energy', 'forces', 'dipole'],
#         sample_systems_optimize=False,
#         meta_cv=[[0,1],[0,5]],
#         meta_hookean=[[0, 1, 6.0],[0,5,6.0]],
#         meta_gaussian_height=0.10,
#         meta_gaussian_widths=0.2,
#         meta_gaussian_interval=10,
#         meta_time_step=1.0,
#         meta_simulation_time=3000.0,
#         meta_save_interval=10,
#         meta_temperature=300,
#         meta_langevin_friction=1.0,
#         meta_initial_velocities=True,
#         meta_initial_temperature=300.,
#     )
#     sampler.run()
    # sampler = MetaSampler(
    #     config='sn2_config.json',
    #     sample_directory='sn2_samples',
    #     sample_systems=['data_sn2/fs_min.xyz'],
    #     sample_systems_format='xyz',
    #     sample_calculator='XTB',
    #     sample_properties=['energy', 'forces', 'dipole'],
    #     sample_systems_optimize=False,
    #     meta_cv=[[0,1],[0,5]],
    #     meta_hookean=[[0, 1, 6.0],[0,5,6.0]],
    #     meta_gaussian_height=0.10,
    #     meta_gaussian_widths=0.2,
    #     meta_gaussian_interval=10,
    #     meta_time_step=1.0,
    #     meta_simulation_time=3000.0,
    #     meta_save_interval=10,
    #     meta_temperature=300,
    #     meta_langevin_friction=1.0,
    #     meta_initial_velocities=True,
    #     meta_initial_temperature=300.,
    # )
    # sampler.run()
# This part samples the initial structures for the sn2 reaction, only methanol and bromomethane are sampled
#Methanol
#     sampler_met = MDSampler(
#         sample_directory='sn2_samples',
#         sample_systems='data_sn2/prod.xyz',
#         sample_systems_format='xyz',
#         sample_calculator='XTB',
#         sample_systems_optimize=True,
#         md_temperature=300,
#         md_time_step=1.0,
#         md_simulation_time=1000.0,
#         md_save_interval=10,
#         md_initial_velocities=True,
#     )
#
#     sampler_met.run()
#
# #Bromomethane
#     sampler_init = MDSampler(
#         sample_directory='sn2_samples',
#         sample_systems='data_sn2/reac.xyz',
#         sample_systems_format='xyz',
#         sample_calculator='XTB',
#         sample_systems_optimize=True,
#         md_temperature=300,
#         md_time_step=1.0,
#         md_simulation_time=1000.0,
#         md_save_interval=10,
#         md_initial_velocities=True,
#     )
#
#     sampler_init.run()
# For angle between Br-C-O
# sampler = MetaSampler(
#     config='sn2_config.json',
#     sample_directory='sn2_samples',
#     sample_systems=['data_sn2/fs_min.xyz'],
#     sample_systems_format='xyz',
#     sample_calculator='XTB',
#     sample_properties=['energy', 'forces', 'dipole'],
#     sample_systems_optimize=False,
#     meta_cv=[[5,0, 1]],
#     meta_hookean=[[5,0,1, 6.0]],
#     meta_gaussian_height=0.10,
#     meta_gaussian_widths=0.2,
#     meta_gaussian_interval=10,
#     meta_time_step=1.0,
#     meta_simulation_time=5000.0,
#     meta_save_interval=10,
#     meta_temperature=300,
#     meta_langevin_friction=1.0,
#     meta_initial_velocities=True,
#     meta_initial_temperature=300.,
# )
# sampler.run()

if True:
    sampler = MetaSampler(
        config='sn2_config.json',
        sample_directory='sn2_samples',
        sample_systems=['data_sn2/pre_min.xyz'],
        sample_systems_format='xyz',
        sample_calculator='XTB',
        sample_properties=['energy', 'forces', 'dipole'],
        sample_systems_optimize=False,
        meta_cv=[['-',0,1,1,5]],
        meta_hookean=[[0, 5, 6.0]],
        meta_gaussian_height=0.10,
        meta_gaussian_widths=0.2,
        meta_gaussian_interval=10,
        meta_time_step=1.0,
        meta_simulation_time=5000.0,
        meta_save_interval=10,
        meta_temperature=300,
        meta_langevin_friction=1.0,
        meta_initial_velocities=True,
        meta_initial_temperature=100.,
    )
    sampler.run()




