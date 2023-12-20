import os
import numpy as np

from ase import io
from ase import units
from ase.optimize import BFGS

import matplotlib.pyplot as plt

# Sampling
if True:
    
    from asparagus import MDSampler
    
    sampler = MDSampler(
        config='nh3_config_md.json',
        sample_directory='sampling_nh3_md',
        sample_data_file='sampling_nh3_md/md_nh3.db',
        sample_systems=['data/nh3_c3v.xyz'],
        sample_systems_format='xyz',
        sample_systems_optimize=True,
        sample_systems_optimize_fmax=0.001,
        md_temperature=500,
        md_time_step=1.0,
        md_simulation_time=10_000.0,
        md_save_interval=10,
        md_langevin_friction=0.01,
        md_equilibration_time=0,
        md_initial_velocities=False
        )
    sampler.run()

# Train
if True:
    
    from asparagus import Asparagus
    
    model = Asparagus(
        config='nh3_config_md.json',
        data_file='sampling_nh3_md/md_nh3.db',
        model_directory='sampling_nh3_md',
        model_properties=['energy', 'forces', 'dipole'],
        model_interaction_cutoff=8.0,
        trainer_properties_weights={
            'energy': 1.,
            'forces': 50.,
            'dipole': 25.
            },
        trainer_max_epochs=1_000,
        )
    model.train()
    model.test(
        test_datasets='all',
        test_directory=model.get('model_directory'))

if False:
    
    from xtb.ase.calculator import XTB
    from ase import vibrations
    
    from asparagus import Asparagus
    
    # Read system
    system_model = io.read('data/nh3_c3v.xyz')
    system_ref = io.read('data/nh3_c3v.xyz')
    
    # Get calculators
    model = Asparagus(config='nh3_config_md.json')
    calc_model = model.get_ase_calculator()
    calc_ref = XTB()
    
    # Assign calculators
    system_model.calc = calc_model
    system_ref.calc = calc_ref
    
    # Optimize systems
    dyn = BFGS(system_model)
    dyn.run(fmax=0.001)
    dyn = BFGS(system_ref)
    dyn.run(fmax=0.001)
    
    system_model_nhmin = system_model.get_distance(0, 1)
    system_ref_nhmin = system_ref.get_distance(0, 1)
    
    print("Model: N-H equilibrium distance: ", system_model_nhmin)
    print("Ref.: N-H equilibrium distance: ", system_ref_nhmin)

    # Vibration
    vibrations_model = vibrations.Vibrations(
        system_model,
        name="vibfiles")
    vibrations_model.clean()
    vibrations_model.run()
    vibrations_model.summary()
    vibrations_ref = vibrations.Vibrations(
        system_ref,
        name="vibfiles")
    vibrations_ref.clean()
    vibrations_ref.run()
    vibrations_ref.summary()
    
    # Set model system position to reference system 
    system_ref.set_positions(system_model.get_positions())
    
    # Get energies for model minimum conformation
    energy_min_model = system_model.get_potential_energy()
    energy_min_model_ref = system_ref.get_potential_energy()
    
    # Scan N-H bond length
    drange = np.arange(0.7, 3.0, 0.02)
    energy_model = np.zeros_like(drange)
    energy_ref = np.zeros_like(drange)
    for idx, di in enumerate(drange):
        
        system_model.set_distance(0, 1, di, fix=0)
        energy_model[idx] = system_model.get_potential_energy()
        system_ref.set_distance(0, 1, di, fix=0)
        energy_ref[idx] = system_ref.get_potential_energy()
    
    # Read sample trajectory
    trajectory = io.Trajectory('sampling_nh3_md/1_md.traj')
    distances = []
    for frame in trajectory:
        distances.append(frame.get_distance(0, 1))
        distances.append(frame.get_distance(0, 2))
        distances.append(frame.get_distance(0, 3))

    # Get distance histogram
    histogram_distances, _ = np.histogram(distances, bins=drange)
    histogram_distances = (
        histogram_distances.astype(float)
        / float(np.sum(histogram_distances))
        / (drange[1] - drange[0]))
    histogram_distances = (
        histogram_distances.astype(float)
        / np.max(histogram_distances)
        * 4.0)
    dcenter = drange[1:] + (drange[1] - drange[0])/2.

    # Plot property: Fontsize
    SMALL_SIZE = 12
    MEDIUM_SIZE = 14
    BIGGER_SIZE = 16
    plt.rc('font', size=SMALL_SIZE, weight='bold')
    plt.rc('xtick', labelsize=SMALL_SIZE)
    plt.rc('ytick', labelsize=SMALL_SIZE)
    plt.rc('legend', fontsize=SMALL_SIZE)
    plt.rc('axes', titlesize=MEDIUM_SIZE)
    plt.rc('axes', labelsize=MEDIUM_SIZE)
    plt.rc('figure', titlesize=BIGGER_SIZE)

    # Plot property: Figure size and arrangement
    figsize = (6, 6)
    sfig = float(figsize[0])/float(figsize[1])
    left = 0.15
    bottom = 0.15
    column = [0.80, 0.00]
    row = [column[0]*sfig]

    # Initialize figure
    fig = plt.figure(figsize=figsize)
    axs1 = fig.add_axes(
        [left + 0.*np.sum(column), bottom, column[0], row[0]])

    # Plot distance histogram
    axs1.bar(
        dcenter, histogram_distances, 
        width=(drange[1] - drange[0]), 
        edgecolor='grey', linewidth=1, fc='None')
    
    # Plot results
    axs1.plot(
        drange, energy_model - energy_min_model, 
        color='red', ls='-', 
        label=(
            'Model - MD\n' + r'(N-H)$_\mathrm{eq}$'
            + f' = {system_model_nhmin:.3f} ' + r'$\mathrm{\AA}$')
        )
    axs1.plot(
        drange, energy_ref - energy_min_model_ref, 
        color='black', ls='--', 
        label=(
            'XTB\n' + r'(N-H)$_\mathrm{eq}$'
            + f' = {system_ref_nhmin:.3f} ' + r'$\mathrm{\AA}$')
        )
    
    axs1.set_xlim(drange[0], drange[-1])
    refmin = np.min(energy_ref - energy_min_model_ref)
    refmax = np.max(energy_ref - energy_min_model_ref)
    refdif = refmax - refmin
    axs1.set_ylim(refmin - .1*refdif, refmax + .1*refdif)
    
    axs1.set_xlabel(r"N-H distance ($\mathrm{\AA}$)", fontweight='bold')
    axs1.set_ylabel("Potential (eV)", fontweight='bold')
    axs1.get_yaxis().set_label_coords(-0.10, 0.50)
    
    axs1.legend(loc='upper left')
    
    plt.savefig(
        os.path.join(model.get('model_directory'), 'eval_md.png'),
        format='png', dpi=200)
    #plt.show()
    plt.close()
    
if True:
    
    from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
    from ase.md.verlet import VelocityVerlet
    from ase.md.langevin import Langevin
    
    # Read system
    system_model = io.read('data/nh3_c3v.xyz')
    
    # Assign calculator
    model = Asparagus(config='nh3_config_md.json')
    calc_model = model.get_ase_calculator()
    system_model.calc = calc_model
    
    # Optimize systems
    dyn = BFGS(system_model)
    dyn.run(fmax=0.001)
    
    # Assign velocities
    MaxwellBoltzmannDistribution(system_model, temperature_K=300)
    
    # Assign Propagator
    dyn = Langevin(
        system_model, 0.2*units.fs, 300*units.kB, 0.001,
        logfile=os.path.join(model.get('model_directory'), 'vib_md.log')
        )
    
    # Prepare trajectory
    trajectory = io.Trajectory(
        os.path.join(model.get('model_directory'), 'vib_md.traj'),
        atoms=system_model, 
        mode='w', properties=['energy', 'forces', 'dipole'])
    dyn.attach(trajectory.write, interval=1)
    
    dyn.run(5000)
