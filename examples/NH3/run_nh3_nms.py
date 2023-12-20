import os
import numpy as np

from ase import Atoms
from ase import io
from ase import units

import matplotlib.pyplot as plt

# Sampling
if True:
    
    from asparagus import NormalModeScanner
    
    sampler = NormalModeScanner(
        config='nh3_config_nms.json',
        sample_directory='sampling_nh3_nms',
        sample_data_file='sampling_nh3_nms/nms_nh3.db',
        sample_systems=['data/nh3_c3v.xyz'],
        sample_systems_format='xyz',
        sample_systems_optimize=True,
        sample_systems_optimize_fmax=0.001,
        nms_harmonic_energy_step=0.03,
        nms_energy_limits=1.00,
        nms_number_of_coupling=2,
        )
    sampler.run(
        nms_exclude_modes=np.arange(6))


# Train
if True:

    from asparagus import Asparagus

    model = Asparagus(
        config='nh3_config_nms.json',
        data_file='sampling_nh3_nms/nms_nh3.db',
        model_directory='sampling_nh3_nms',
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
    
    from ase.optimize import BFGS
    from xtb.ase.calculator import XTB
    from ase import vibrations

    from asparagus import Asparagus

    # Read system
    system_model = io.read('data/nh3_c3v.xyz')
    system_ref = io.read('data/nh3_c3v.xyz')
    
    # Get calculators
    model = Asparagus(config='nh3_config_nms.json')
    calc_model = model.get_ase_calculator()
    calc_ref = XTB(accuracy=.1)
    
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
    trajectory = io.Trajectory('sampling_nh3_nms/1_nmscan.traj')
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
        color='green', ls='-', 
        label=(
            'Model - NMS\n' + r'(N-H)$_\mathrm{eq}$'
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
        os.path.join(model.get('model_directory'), 'eval_nms.png'),
        format='png', dpi=200)
    #plt.show()
    plt.close()

if True:
    
    from ase.optimize import BFGS
    from xtb.ase.calculator import XTB
    from ase import vibrations
    
    # Generate System
    system = Atoms("N2", positions=[[.0, .0, .0], [.0, .0, 1.0]])
    
    # Get calculators
    calc = XTB()
    
    # Assign calculators
    system.calc = calc
    
    # Optimize systems
    dyn = BFGS(system)
    dyn.run(fmax=0.001)
    
    #io.write("n2_eq.xyz", system, format='xyz')
    
    system_emin = system.get_potential_energy()
    system_dmin = system.get_distance(0, 1)
    
    # Get Vibrations
    vibrations_model = vibrations.Vibrations(
        system,
        name="vibfiles")
    vibrations_model.clean()
    vibrations_model.run()
    vibrations_model.summary()
    
    # Vibrational mode
    vibm = np.array(
        vibrations_model.get_mode(-1).reshape(2, 3)
        / np.sqrt(np.sum(vibrations_model.get_mode(-1)**2)))
    
    # Vibrational frequency in cm-1
    vibf = vibrations_model.get_frequencies()[-1]
    
    # Force constant
    redm = 0.5*system.get_masses()[0]
    vibk = (
        4.0*np.pi**2*(np.abs(vibf)*1.e2*units._c)**2
        * redm*units._amu*units.J*1.e-20)
    print(vibk)

    vstep = 0.25
    vibs = np.sqrt(3*vstep/vibk)
    print(vibs)
    
    # Scan
    drange = np.concatenate(
        (np.arange(system_dmin, 0.96, -0.002)[::-1], 
         np.arange(system_dmin, 1.45, 0.002)[1:]))
    energy = np.zeros_like(drange)
    for idx, di in enumerate(drange):
        
        system.set_distance(0, 1, di)
        energy[idx] = system.get_potential_energy()
    energy -= system_emin
    
    # Harmonic energy
    energy_harm = 0.5*vibk*((drange - system_dmin))**2
    
    # Get closest idx
    idx_sort = np.argsort(np.abs(energy_harm - vstep))
    for idx in idx_sort:
        if drange[idx] > system_dmin:
            idx_step = idx
            break
    print(idx_step, drange[idx_step], drange[idx_step] - system_dmin)
    
    # Get actual step energy
    system.set_distance(0, 1, system_dmin + 1.0*vibs)
    distance_1vibs = system_dmin + 1.0*vibs
    energy_1vibs = 0.5*vibk*(1.0*vibs)**2
    
    # Refernce energy per step
    energy_step = np.zeros(5, dtype=float)
    distance_step = np.zeros(5, dtype=float)
    for istep in range(1, 5):
        
        system.set_distance(0, 1, system_dmin + istep*vibs)
        distance_step[istep] = system_dmin + istep*vibs
        energy_step[istep] = system.get_potential_energy()
    energy_step -= system_emin
    
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
    
    axs1.plot(drange, energy, '-k', label='Ref.')
    
    print(energy_step)
    axs1.plot(
        [0, distance_step[1]], [energy_step[1], energy_step[1]], 
        ':', color='black')
    axs1.plot(
        [distance_step[1], distance_step[1]], [0, energy_step[1]], 
        ':', color='black')
    
    axs1.plot(
        [0, distance_step[2]], [energy_step[2], energy_step[2]], 
        ':', color='black')
    axs1.plot(
        [distance_step[2], distance_step[2]], [0, energy_step[2]], 
        ':', color='black')
    
    axs1.plot(
        [0, distance_step[3]], [energy_step[3], energy_step[3]], 
        ':', color='black')
    axs1.plot(
        [distance_step[3], distance_step[3]], [0, energy_step[3]], 
        ':', color='black')
    
    axs1.plot(drange, energy_harm, '-', color='orange', label='Harm.')
    
    axs1.plot([0, drange[idx_step]], [vstep, vstep], '--', color='orange')
    axs1.plot(
        [drange[idx_step], drange[idx_step]], [0, vstep], '--', color='orange')
    
    print(energy_1vibs)
    axs1.plot(
        [0, distance_1vibs], [energy_1vibs, energy_1vibs], '--', color='red')
    axs1.plot(
        [distance_1vibs, distance_1vibs], [0, energy_1vibs], '--', color='red')
    
    
    axs1.set_xlim(drange[0], drange[-1])
    axs1.set_ylim(0.0, 2.0)
    
    axs1.set_xlabel(r"N-N distance ($\mathrm{\AA}$)", fontweight='bold')
    axs1.set_ylabel("Potential (eV)", fontweight='bold')
    axs1.get_yaxis().set_label_coords(-0.12, 0.50)
    
    axs1.legend(loc='upper left')
    
    #plt.savefig('theory_nms.png', format='png', dpi=200)
    plt.show()
    plt.close()

    
    plt.show()
    
    
    
