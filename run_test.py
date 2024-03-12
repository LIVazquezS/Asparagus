
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

#==============================================================================
# Test Asparagus Sampler Methods
#==============================================================================

# Sampler - with XTB and ORCA
# Mind: XTB is not thread safe when using with ASE modules such as Optimizer
# or Vibrations, but simple Calculator call works
if False:
    
    from asparagus.sample import Sampler
    
    # Load single system from xyz file and compute properties using XTB default 
    # calculator
    sampler = Sampler(
        config='test/smpl_nh3.json',
        sample_directory='test',
        sample_data_file='test/smpl_nh3.db',
        sample_systems='data/nh3_c3v.xyz',
        sample_systems_format='xyz',
        sample_num_threads=1,
        )
    sampler.run()
    
    # Load two system from xyz file and compute properties using XTB default 
    # calculator in parallel (still works without using other ASE functions)
    sampler = Sampler(
        config='test/smpl_nh3.json',
        sample_directory='test',
        sample_data_file='test/smpl_nh3.db',
        sample_systems=['data/nh3_c3v.xyz', 'data/nh3_d3h.xyz'],
        sample_systems_format='xyz',
        sample_num_threads=2,
        )
    sampler.run()

    # Load two systems from xyz file and a ASE trajectory file and compute 
    # properties using XTB default calculator
    sampler = Sampler(
        config='test/smpl_nh3.json',
        sample_directory='test',
        sample_data_file='test/smpl_nh3.db',
        sample_systems=['data/nh3_c3v.xyz', 'data/meta_nh3.traj'],
        sample_num_threads=1,
        )
    sampler.run()
    
    # Load a selection of sample system from an Asparagus data file and compute
    # properties using XTB default calculator
    sampler = Sampler(
        config='test/smpl_nh3.json',
        sample_directory='test',
        sample_data_file='test/smpl_nh3.db',
        sample_systems='data/nms_nh3.db',
        sample_systems_format='db',
        sample_systems_indices=[0, 1, 2, 3, -4, -3, -2, -1],
        sample_num_threads=1,
        )
    sampler.run()

    from asparagus.sample import MCSampler

    # Sample a single system loaded from a xyz file using the Monte-Carlo
    # sampling method with the XTB calculator
    sampler = MCSampler(
        config='test/mc_nh3.json',
        sample_directory='test',
        sample_data_file='test/mc_nh3.db',
        sample_systems='data/nh3_c3v.xyz',
        sample_systems_format='xyz',
        sample_calculator='XTB',   # Not thread save when using ASE modules
        sample_num_threads=1,
        sample_systems_optimize=True,
        sample_systems_optimize_fmax=0.001,
        mc_temperature=300.0,
        mc_steps=100,
        mc_max_displacement=0.1,
        mc_save_interval=1,
        )
    sampler.run()
    
    # Sample two systems loaded from a xyz files in parallel using the
    # Monte-Carlo sampling method and the ORCA calculator (thread safe)
    sampler = MCSampler(
        config='test/mc_nh3.json',
        sample_directory='test',
        sample_data_file='test/mc_nh3.db',
        sample_systems=['data/nh3_c3v.xyz', 'data/nh3_d3h.xyz'],
        sample_systems_format='xyz',
        sample_calculator='ORCA',
        sample_calculator_args = {
            'charge': 0,
            'mult': 1,
            'orcasimpleinput': 'RI PBE D3BJ def2-SVP def2/J TightSCF',
            'orcablocks': '%pal nprocs 1 end',
            'directory': 'test/orca'},
        sample_save_trajectory=True,
        sample_num_threads=2,
        sample_systems_optimize=True,
        sample_systems_optimize_fmax=0.001,
        mc_temperature=300.0,
        mc_steps=10,
        mc_max_displacement=0.1,
        mc_save_interval=1,
        )
    sampler.run()

    from asparagus.sample import MDSampler

    # Sample a single system loaded from a xyz file using the Molecular 
    # Dynamics sampling method with the XTB calculator
    sampler = MDSampler(
        config='test/md_nh3.json',
        sample_directory='test',
        sample_data_file='test/md_nh3.db',
        sample_systems='data/nh3_c3v.xyz',
        sample_systems_format='xyz',
        sample_calculator='XTB',   # Not thread save when using ASE modules
        sample_num_threads=1,
        sample_systems_optimize=True,
        sample_systems_optimize_fmax=0.001,
        md_temperature=500,
        md_time_step=1.0,
        md_simulation_time=100.0,
        md_save_interval=10,
        md_langevin_friction=0.01,
        md_equilibration_time=0,
        md_initial_velocities=False,
        )
    sampler.run()

    # Sample two systems loaded from a xyz files in parallel using the 
    # Molecular Dynamics sampling method and the ORCA calculator (thread safe)
    sampler = MDSampler(
        config='test/md_nh3.json',
        sample_directory='test',
        sample_data_file='test/md_nh3.db',
        sample_systems=['data/nh3_c3v.xyz', 'data/nh3_d3h.xyz'],
        sample_systems_format='xyz',
        sample_calculator='ORCA',
        sample_calculator_args = {
            'charge': 0,
            'mult': 1,
            'orcasimpleinput': 'RI PBE D3BJ def2-SVP def2/J TightSCF',
            'orcablocks': '%pal nprocs 1 end',
            'directory': 'test/orca'},
        sample_save_trajectory=True,
        sample_num_threads=2,
        sample_systems_optimize=True,
        sample_systems_optimize_fmax=0.001,
        md_temperature=500,
        md_time_step=1.0,
        md_simulation_time=20.0,
        md_save_interval=10,
        md_langevin_friction=0.01,
        md_equilibration_time=0,
        md_initial_velocities=True,
        md_initial_temperature=300,
        )
    sampler.run()

    from asparagus.sample import MetaSampler
    
    # Sample a single system loaded from a xyz file using the Meta Dynamics
    # sampling method with the XTB calculator
    sampler = MetaSampler(
        config='test/meta_nh3.json',
        sample_directory='test',
        sample_data_file='test/meta_nh3.db',
        sample_systems='data/nh3_c3v.xyz',
        sample_systems_format='xyz',
        sample_calculator='XTB',   # Not thread save when using ASE modules
        sample_save_trajectory=True,
        sample_num_threads=1,
        sample_systems_optimize=True,
        sample_systems_optimize_fmax=0.001,
        meta_cv=[[0, 1], [0, 2], [0, 3]],
        meta_gaussian_height=0.10,
        meta_gaussian_widths=0.1,
        meta_gaussian_interval=10,
        meta_hookean=[[0, 1, 4.0], [0, 2, 4.0], [0, 3, 4.0]],
        meta_temperature=500,
        meta_time_step=1.0,
        meta_simulation_time=10_0.0,
        meta_save_interval=10,
        )
    sampler.run()

    
    # Sample a system loaded from a xyz files in parallel using the Meta
    # Dynamics sampling method and the ORCA calculator (thread safe)
    # Here each of the multiple runs store the Gaussian add-potentials into 
    # in the same list, affecting the other runs as well and decrease the
    # sample steps to reach higher potential areas.
    # Not yet working as planned
    sampler = MetaSampler(
        config='test/meta_nh3.json',
        sample_directory='test',
        sample_data_file='test/meta_nh3.db',
        sample_systems='data/nh3_c3v.xyz',
        sample_systems_format='xyz',
        sample_calculator='ORCA',
        sample_calculator_args = {
            'charge': 0,
            'mult': 1,
            'orcasimpleinput': 'RI PBE D3BJ def2-SVP def2/J TightSCF',
            'orcablocks': '%pal nprocs 1 end',
            'directory': 'test/orca'},
        sample_save_trajectory=True,
        sample_num_threads=4,
        sample_systems_optimize=True,
        sample_systems_optimize_fmax=0.001,
        meta_cv=[[0, 1], [0, 2], [0, 3]],
        meta_gaussian_height=0.10,
        meta_gaussian_widths=0.1,
        meta_gaussian_interval=10,
        meta_hookean=[[0, 1, 4.0], [0, 2, 4.0], [0, 3, 4.0]],
        meta_temperature=500,
        meta_time_step=1.0,
        meta_simulation_time=20.0,
        meta_save_interval=10,
        meta_parallel=True,
        )
    sampler.run()

    # Sample two system loaded from a xyz files in parallel using the Meta
    # Dynamics sampling method and the ORCA calculator (thread safe)
    sampler = MetaSampler(
        config='test/meta_nh3.json',
        sample_directory='test',
        sample_data_file='test/meta_nh3.db',
        sample_systems=['data/nh3_c3v.xyz', 'data/nh3_d3h.xyz'],
        sample_systems_format='xyz',
        sample_calculator='ORCA',
        sample_calculator_args = {
            'charge': 0,
            'mult': 1,
            'orcasimpleinput': 'RI PBE D3BJ def2-SVP def2/J TightSCF',
            'orcablocks': '%pal nprocs 1 end',
            'directory': 'test/orca'},
        sample_save_trajectory=True,
        sample_num_threads=2,
        sample_systems_optimize=True,
        sample_systems_optimize_fmax=0.001,
        meta_cv=[[0, 1], [0, 2], [0, 3]],
        meta_gaussian_height=0.10,
        meta_gaussian_widths=0.1,
        meta_gaussian_interval=10,
        meta_hookean=[[0, 1, 4.0], [0, 2, 4.0], [0, 3, 4.0]],
        meta_temperature=500,
        meta_time_step=1.0,
        meta_simulation_time=20.0,
        meta_save_interval=10,
        )
    sampler.run()

    from asparagus.sample import NormalModeScanner
    
    # Sample a single system loaded from a xyz file using the Normal Mode
    # Scanner sampling method with the XTB calculator
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
        nms_energy_limits=0.50,
        nms_number_of_coupling=1,
        nms_limit_of_steps=10,
        nms_limit_com_shift=0.01,
        nms_save_displacements=True,
        )
    sampler.run()

    # Sample two systems loaded from a xyz file using the Normal Mode
    # Scanner sampling method with the ORCA calculator (thread safe).
    # Here it parallelize over the (1) system optimizations, (2) atom
    # displacement calculations for numeric normal mode analysis and (3) the
    # scans along single or combinations of normal modes. Step (2) and (3) will
    # run in serial for each sample system.
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
        nms_energy_limits=0.50,
        nms_number_of_coupling=1,
        nms_limit_of_steps=10,
        nms_limit_com_shift=0.01,
        nms_save_displacements=False,
        )
    sampler.run()

    from asparagus.sample import NormalModeSampler
    
    # Sample a single system loaded from a xyz file using the Normal Mode
    # Sampler sampling method with the XTB calculator
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
    
    # Sample a system loaded from a xyz file using the Normal Mode
    # Sampler sampling method with the ORCA calculator (thread safe).
    # Here it parallelize over the (1) system optimizations, (2) atom
    # displacement calculations for numeric normal mode analysis and (3) the
    # number of randomly sampled system conformations. Step (2) and (3) will
    # run in serial for each potential sample system.
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
        nms_nsamples=10,
        )
    sampler.run()

#==============================================================================
# Test Asparagus Calculators
#==============================================================================

# Shell Calculator
if False:
    
    from asparagus.sample import Sampler
    
    # Calculate properties of a sample system with multiple conformations
    # using the Shell calculator with template files for an ORCA calculation.
    sampler = Sampler(
        config='test/calc_nh3.json',
        sample_directory='test',
        sample_data_file='test/smpl_nh3.db',
        sample_systems='data/nh3_c3v.xyz',
        sample_systems_format='xyz',
        sample_calculator='shell',
        sample_calculator_args = {
            'files': [
                'data/template/run_orca.sh',
                'data/template/run_orca.inp',
                'data/template/run_orca.py',
                ],
            'files_replace': {
                '%xyz%': '$xyz',
                '%charge%': '$charge',
                '%multiplicity%': '$multiplicity',
                },
            'execute_file': 'run_orca.sh',  # or 'data/template/run_orca.sh'
            'charge': 0,
            'multiplicity': 1,
            'directory': 'test/shell',
            'result_properties': ['energy', 'forces', 'dipole']
            },
        sample_num_threads=1,
        )
    sampler.run()

    # Calculate properties of a sample system with multiple conformations
    # using the Shell calculator with template files for an ORCA calculation
    # and in parallel.
    sampler = Sampler(
        config='test/calc_nh3.json',
        sample_directory='test',
        sample_data_file='test/smpl_nh3.db',
        sample_systems='data/meta_nh3.traj',
        sample_calculator='shell',
        sample_calculator_args = {
            'files': [
                'data/template/shell/run_orca.sh',
                'data/template/shell/run_orca.inp',
                'data/template/shell/run_orca.py',
                ],
            'files_replace': {
                '%xyz%': '$xyz',
                '%charge%': '$charge',
                '%multiplicity%': '$multiplicity',
                },
            'execute_file': 'data/template/shell/run_orca.sh',
            'charge': 0,
            'multiplicity': 1,
            'directory': 'test/shell',
            'result_properties': ['energy', 'forces', 'dipole']
            },
        sample_num_threads=4,
        sample_save_trajectory=True,
        )
    sampler.run()

# Slurm Calculator
if True:
    
    from asparagus.sample import Sampler
    
    # Calculate properties of a sample system with multiple conformations
    # using the Slurm calculator with template files for a MOLPRO calculation.
    sampler = Sampler(
        config='test/calc_nh3.json',
        sample_directory='test',
        sample_data_file='test/smpl_nh3.db',
        sample_systems='data/nh3_c3v.xyz',
        sample_systems_format='xyz',
        sample_calculator='slurm',
        sample_calculator_args = {
            'files': [
                'data/template/slurm/run_molpro.sh',
                'data/template/slurm/run_molpro.inp',
                'data/template/slurm/run_molpro.py',
                ],
            'files_replace': {
                '%xyz%': '$xyz',
                '%charge%': '$charge',
                '%spin2%': '$spin2',
                },
            'execute_file': 'run_molpro.sh',
            'charge': 0,
            'multiplicity': 1,
            'directory': 'test/slurm',
            'result_properties': ['energy', 'forces', 'dipole']
            },
        sample_num_threads=1,
        )
    sampler.run()

    # Calculate properties of a sample system with multiple conformations
    # using the Slurm calculator with template files for a MOLPRO calculation.
    # Here, define own slurm task id catch and check function
    
    def catch_id(stdout):
        """
        Catch slurm task id from the output when running:
          subrocess.run([command, execute_file], capture_output=True)
          (here [command, execute_file] -> 'sbatch run_molpro.sh')
        
        Parameters
        ----------
        stdout: str
            Decoded output line (e.g. 'Submitted batch job 10937679')
        
        Return
        ------
        int
            Task id
        """
        return int(proc.stdout.decode().split()[-1])
    
    import subprocess
    def check_id(slurm_id):
        """
        Check slurm task id with e.g. task id list extracted from squeue
        
        Parameters
        ----------
        slurm_id: int
            Slurm task id of the submitted job
        
        Return
        ------
        bool
            Answer if task is done:
            False, if task is still running (task id is found in squeue)
            True, if task is done (task id not found in squeue)
        """
        proc = subprocess.run(
            ['squeue', '-u', os.environ['USER']],
            capture_output=True)
        active_id = [
            int(tasks.split()[0])
            for tasks in proc.stdout.decode().split('\n')[1:-1]]
        return not slurm_id in active_id

    sampler = Sampler(
        config='test/calc_nh3.json',
        sample_directory='test',
        sample_data_file='test/smpl_nh3.db',
        sample_systems='data/nh3_c3v.xyz',
        sample_systems_format='xyz',
        sample_calculator='slurm',
        sample_calculator_args = {
            'files': [
                'data/template/slurm/run_molpro.sh',
                'data/template/slurm/run_molpro.inp',
                'data/template/slurm/run_molpro.py',
                ],
            'files_replace': {
                '%xyz%': '$xyz',
                '%charge%': '$charge',
                '%spin2%': '$spin2',
                },
            'execute_file': 'run_molpro.sh',
            'charge': 0,
            'multiplicity': 1,
            'directory': 'test/slurm',
            'result_properties': ['energy', 'forces', 'dipole']
            },
        sample_num_threads=1,
        scan_interval=1,
        scan_catch_id=catch_id,
        scan_check_id=check_id,
        )
    sampler.run()
