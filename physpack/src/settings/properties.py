#======================================
# Model type labels
#======================================

# Valid model type labels with known build up
_valid_model_type = [
    'PhysNet',
]

# Valid input model type labels
_valid_input_model_type = [
    'PhysNetRBF',
]

# Valid graph model type labels
_valid_graph_model_type = [
    'PhysNetMP',
]

# Valid output model type labels
_valid_output_model_type = [
    'PhysNetOut',
]

#======================================
# Cutoff function labels
#======================================

# Valid cutoff function labels
_valid_cutoff_fn = [
    'Poly6',
]

#======================================
# Property labels
#======================================

# Valid property labels
_valid_properties = [
    'positions',
    'energy',
    'atomic_energies',
    'forces',
    'hessian',
    'charge',
    'atomic_charges',
    'dipole',
]

# Alternative property label dictionary 
# (keys:    internally used property labels
#  items:   possible externally used property labels)
# Comparison always between lowercase form
_alt_property_labels = {
    'atoms_number':     ['N'],
    'atomic_numbers':   ['Z', 'atomic_number'],
    'positions':        ['R', 'position'],
    'cell':             ['unit_cell'],
    'pbc':              ['periodicity'],
    'energy':           ['E', 'energies', 'U0', 'V'],
    'atomic_energies':  [
        'Ea', 'Ei', 'atom_energy', 'atoms_energy', 'atomic_energy',
        'atom_energies', 'atoms_energies'],
    'forces':           ['F', 'force'],
    'hessian':          ['H', 'hessians'],
    'charge':           ['Q', 'charges', 'total_charge', 'total_charges'],
    'atomic_charges':    [
        'Qa', 'Qi', 'atom_charge', 'atom_charges', 
        'atomic_charge', 'atomic_charges'],
    'dipole':           ['D', 'dipoles'],
    }
    

# Default property units
_default_units = {
    'atoms_number':     '',
    'atomic_numbers':   '',
    'positions':        'Ang',
    'energy':           'eV',
    'atomic_energies':  'eV',
    'forces':           'eV/Ang',
    'hessian':          'eV/Ang/Ang',
    'charge':           'e',
    'atomic_charges':   'e',
    'dipole':           'eAng',
    }

