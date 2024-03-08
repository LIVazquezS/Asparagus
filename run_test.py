
import asparagus

import torch
import numpy as np
import time

#==============================================================================
# Test Parameter
#==============================================================================

config_file = 'test/init.json'
config = {
    'config_file': config_file}
device = 'cpu'
dtype=torch.float32
    
#==============================================================================
# Test Asparagus Main Class Initialization
#==============================================================================

# Dictionary initialization
if False:

    model = asparagus.Asparagus(config)
    model = asparagus.Asparagus(config=config_file)
    model = asparagus.Asparagus(config_file=config_file)

# Global device and dtype setting
if False:

    model = asparagus.Asparagus(
        config,
        model_device=device,
        model_dtype=dtype)
    model = asparagus.Asparagus(
        config,
        model_device=device,
        model_dtype=dtype)

#==============================================================================
# Test Asparagus DataContainer Class Initialization
#==============================================================================

# SQL
if False:

    # Open DataBase file
    model = asparagus.Asparagus(
        config=config_file,
        data_file='data/nms_nh3.db',
        data_file_format='sql',
        )

    # Create new DataBase file
    model.set_DataContainer(
        config=config_file,
        data_file='test/nms_nh3_test.db',
        data_source='data/nms_nh3.db',
        data_overwrite=True,
    )

    # Add same source to DataBase file, should be skipped
    model.set_DataContainer(
        config=config_file,
        data_file='test/nms_nh3_test.db',
        data_source='data/nms_nh3.db',
        data_overwrite=False,
    )

    # Create new DataBase file with itself as source, should return error
    try:
        model.set_DataContainer(
            config=config_file,
            data_file='test/nms_nh3_test.db',
            data_source='test/nms_nh3_test.db',
            data_overwrite=True,
        )
    except SyntaxError:
        print("\nSyntaxError as expected\n")

    # Get DataContainer (Reset data source)
    model = asparagus.Asparagus(
        config=config_file,
        data_file='test/nms_nh3_test.db',
        data_file_format='sql',
        data_source='data/nms_nh3.db',
        )
    data = model.get_DataContainer()
    print("\nDatabase path: ", model.get_DataContainer(), "\n")
    print("\nDatabase entry '0': ", data[0])
    print("\nDatabase Train entry '1': ", data.get_train(1))
    print("\nDatabase Valid entry '2': ", data.get_valid(2))
    print("\nDatabase Test entry  '3': ", data.get_test(3))
    
    # Load Numpy .npz files
    model.set_DataContainer(
        config=config_file,
        data_file='test/h2co_test.db',
        data_source='data/h2co_B3LYP_cc-pVDZ_4001.npz',
        data_overwrite=True,
    )
    
    # Load multiple source files files
    model.set_DataContainer(
        config=config_file,
        data_file='test/nh3_h2co_test.db',
        data_source=['data/h2co_B3LYP_cc-pVDZ_4001.npz', 'data/nms_nh3.db'],
        data_overwrite=True,
    )
    
    # Check if repeated data source is skipped
    model.set_DataContainer(
        config=config_file,
        data_file='test/nh3_h2co_test.db',
        data_source=['data/nms_nh3.db'],
        data_overwrite=False,
    )
    
    # Load ASE trajectory file
    model.set_DataContainer(
        config=config_file,
        data_file='test/meta_nh3_test.db',
        data_source='data/meta_nh3.traj',
        data_overwrite=True,
    )

# HDF5
if False:

    # Create new DataBase file
    model.set_DataContainer(
        config=config_file,
        data_file='test/nms_nh3_test.db',
        data_file_format='hdf5',
        data_source='data/nms_nh3.db',
        data_source_format='sql',
        data_overwrite=True,
    )
    data = model.get_DataContainer()
    print(data[0])

#==============================================================================
# Test Asparagus Sampler Methods
#==============================================================================

# Sampler - with XTB and ORCA
# Mind: XTB is not thread safe when using with ASE modules such as Optimizer
# or Vibrations, but simple Calculator call works
if True:
    
    from asparagus.sample import Sampler
    
    sampler = Sampler(
        config='test/smpl_nh3.json',
        sample_directory='test',
        sample_data_file='test/smpl_nh3.db',
        sample_systems='data/nh3_c3v.xyz',
        sample_systems_format='xyz',
        sample_num_threads=1,
        )
    sampler.run()
    
    sampler = Sampler(
        config='test/smpl_nh3.json',
        sample_directory='test',
        sample_data_file='test/smpl_nh3.db',
        sample_systems=['data/nh3_c3v.xyz', 'data/nh3_d3h.xyz'],
        sample_systems_format='xyz',
        sample_num_threads=2,
        )
    sampler.run()

    sampler = Sampler(
        config='test/smpl_nh3.json',
        sample_directory='test',
        sample_data_file='test/smpl_nh3.db',
        sample_systems=['data/nh3_c3v.xyz', 'data/meta_nh3.traj'],
        sample_num_threads=1,
        )
    sampler.run()

    #from asparagus.sample import MCSampler

    #sampler = MCSampler(
        #config='test/mc_nh3.json',
        #sample_directory='test',
        #sample_data_file='test/mc_nh3.db',
        #sample_systems='data/nh3_c3v.xyz',
        #sample_systems_format='xyz',
        #sample_systems_optimize=True,
        #sample_systems_optimize_fmax=0.001,
        #mc_temperature=300.0,
        #mc_steps=100,
        #mc_max_displacement=0.1,
        #mc_save_interval=1,
        #)
    #sampler.run()

    #from asparagus.sample import MDSampler

    #sampler = MDSampler(
        #config='test/md_nh3.json',
        #sample_directory='test',
        #sample_data_file='test/md_nh3.db',
        #sample_systems='data/nh3_c3v.xyz',
        #sample_systems_format='xyz',
        #sample_systems_optimize=True,
        #sample_systems_optimize_fmax=0.001,
        #md_temperature=500,
        #md_time_step=1.0,
        #md_simulation_time=1000.0,
        #md_save_interval=10,
        #md_langevin_friction=0.01,
        #md_equilibration_time=0,
        #md_initial_velocities=False,
        #)
    #sampler.run()

    #from asparagus.sample import MetaSampler
    
    #sampler = MetaSampler(
        #config='test/meta_nh3.json',
        #sample_directory='test',
        #sample_data_file='test/meta_nh3.db',
        #sample_systems='data/nh3_c3v.xyz',
        #sample_systems_format='xyz',
        #sample_systems_optimize=True,
        #sample_systems_optimize_fmax=0.001,
        #meta_cv=[[0, 1], [0, 2], [0, 3]],
        #meta_gaussian_height=0.10,
        #meta_gaussian_widths=0.1,
        #meta_gaussian_interval=10,
        #meta_hookean=[[0, 1, 4.0], [0, 2, 4.0], [0, 3, 4.0]],
        #meta_temperature=500,
        #meta_time_step=1.0,
        #meta_simulation_time=10_0.0,
        #meta_save_interval=10,
        #)
    #sampler.run()

    from asparagus.sample import NormalModeScanner
    
    sampler = NormalModeScanner(
        config='test/nms_nh3.json',
        sample_directory='test',
        sample_data_file='test/nms_nh3.db',
        sample_systems='data/nh3_c3v.xyz',
        sample_systems_format='xyz',
        sample_calculator='XTB',   # Not thread save when using ASE modules
        sample_num_threads=1,
        sample_systems_optimize=True,
        sample_systems_optimize_fmax=0.001,
        nms_harmonic_energy_step=0.10,
        nms_energy_limits=1.00,
        nms_number_of_coupling=1,
        nms_limit_of_steps=10,
        nms_limit_com_shift=0.01,
        nms_save_displacements=True,
        )
    sampler.run()

    sampler = NormalModeScanner(
        config='test/nms_nh3.json',
        sample_directory='test',
        sample_data_file='test/nms_nh3.db',
        sample_systems=['data/nh3_c3v.xyz', 'data/nh3_d3h.xyz'],
        sample_systems_format='xyz',
        sample_calculator='ORCA',
        sample_calculator_args = {
            'charge': 0,
            'mult': 1,
            'orcasimpleinput': 'RI PBE D3BJ def2-SVP def2/J TightSCF',
            'orcablocks': '%pal nprocs 1 end',
            'directory': 'test/orca'},
        sample_num_threads=4,
        sample_systems_optimize=True,
        sample_systems_optimize_fmax=0.001,
        nms_harmonic_energy_step=0.10,
        nms_energy_limits=1.00,
        nms_number_of_coupling=1,
        nms_limit_of_steps=10,
        nms_limit_com_shift=0.01,
        nms_save_displacements=False,
        )
    sampler.run()

    from asparagus.sample import NormalModeSampler
    
    sampler = NormalModeSampler(
        config='test/nms_nh3.json',
        sample_directory='test',
        sample_data_file='test/nms_nh3.db',
        sample_systems='data/nh3_c3v.xyz',
        sample_systems_format='xyz',
        sample_calculator='XTB',   # Not thread save when using ASE modules
        sample_num_threads=1,
        sample_systems_optimize=True,
        sample_systems_optimize_fmax=0.001,
        nms_temperature=500.0,
        nms_nsamples=100,
        )
    sampler.run()
    
    
    sampler = NormalModeSampler(
        config='test/nms_nh3.json',
        sample_directory='test',
        sample_data_file='test/nms_nh3.db',
        sample_systems='data/nh3_d3h.xyz',
        sample_systems_format='xyz',
        sample_calculator='ORCA',
        sample_calculator_args = {
            'charge': 0,
            'mult': 1,
            'orcasimpleinput': 'RI PBE D3BJ def2-SVP def2/J TightSCF',
            'orcablocks': '%pal nprocs 1 end',
            'directory': 'test/orca'},
        sample_num_threads=4,
        sample_systems_optimize=True,
        sample_systems_optimize_fmax=0.001,
        nms_temperature=500.0,
        nms_nsamples=100,
        )
    sampler.run()

    pass

