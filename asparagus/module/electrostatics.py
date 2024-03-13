from typing import Optional, List, Dict, Tuple, Union, Any

import torch

from .. import layers
from .. import utils
from .. import settings

# ======================================
#  Point Charge Electrostatics
# ======================================


class PC_shielded_electrostatics(torch.nn.Module):
    """
    Torch implementation of a shielded point charge electrostatic model that
    avoids singularities at very close atom pair distances.

    Parameters
    ----------
    split_distance: bool
        If True, the electrostatic potential is split into ordinary and
        shielded contributions. If False, only ordinary contributions are
        used.
    short_range_cutoff: float
        Short range cutoff distance in Angstrom.
    long_range_cutoff: float
        Long range cutoff distance in Angstrom.
    unit_properties: dict, optional, default None
        Dictionary of unit properties.
    switch_fn: (str, callable), optional, default None
        Switch function for the short range cutoff.
    device: str, optional, default 'cpu'
        Device type for model variable allocation
    dtype: dtype object, optional, default 'torch.float64'
        Model variables data type
    **kwargs
        Additional keyword arguments.

    """

    def __init__(
        self,
        split_distance: bool,
        short_range_cutoff: float,
        long_range_cutoff: float,
        unit_properties: Optional[Dict[str, str]] = None,
        switch_fn: Optional[Union[str, object]] = None,
        device: Optional[str] = 'cpu',
        dtype: Optional[object] = torch.float64,
        **kwargs
    ):

        super(PC_shielded_electrostatics, self).__init__()

        # Assign variables
        self.split_distance = split_distance
        self.short_range_cutoff = short_range_cutoff
        self.long_range_cutoff = long_range_cutoff
        self.long_range_cutoff_squared = long_range_cutoff**2
        self.dtype = dtype
        self.device = device

        # Assign switch_fn
        switch_class = layers.get_cutoff_fn(switch_fn)
        self.switch_fn = switch_class(short_range_cutoff)

        # Set property units for parameter scaling
        self.set_unit_properties(unit_properties)

        return

    def set_unit_properties(
        self,
        unit_properties: Dict[str, str],
    ):
        
        # Get conversion factors
        if unit_properties is None:
            unit_energy = settings._default_units.get('energy')
            unit_positions = settings._default_units.get('positions')
            unit_charge = settings._default_units.get('charge')
            factor_energy, _ = utils.check_units(unit_energy)
            factor_positions, _ = utils.check_units(unit_positions)
            factor_charge, _ = utils.check_units(unit_charge)
        else:
            factor_energy, _ = utils.check_units(
                unit_properties.get('energy'))
            factor_positions, _ = utils.check_units(
                unit_properties.get('positions'))
            factor_charge, _ = utils.check_units(
                unit_properties.get('charge'))
    
        # Convert 1/(2*4*pi*epsilon) from e**2/eV/Ang to model units
        kehalf_ase = 7.199822675975274
        self.register_buffer(
            'kehalf',
            torch.tensor(
                [kehalf_ase*factor_charge**2/factor_energy/factor_positions],
                dtype=self.dtype)
            )

        return

    def forward(
        self,
        atomic_charges: torch.Tensor,
        distances: torch.Tensor,
        idx_i: torch.Tensor,
        idx_j: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute shielded electrostatic interaction between atom center point 
        charges.

        Parameters
        ----------
        atomic_charges : torch.Tensor
            Atomic charges
        distances : torch.Tensor
            Distances between all atom pairs in the batch.
        idx_i : torch.Tensor
            Indices of the first atom of each pair.
        idx_j : torch.Tensor
            Indices of the second atom of each pair.

        Returns
        -------
        torch.Tensor
            Dispersion atom energy contribution
        
        """

        # Gather atomic charges
        atomic_charges_i = torch.gather(atomic_charges, 0, idx_i)
        atomic_charges_j = torch.gather(atomic_charges, 0, idx_j)

        # Compute shielded distances
        distances_shielded = torch.sqrt(distances**2 + 1.0)

        # Compute switch weights
        switch_off_weights = self.switch_fn(distances)
        switch_on_weights = 1.0 - switch_off_weights

        # Compute electrostatic potential
        if self.split_distance:

            # Compute ordinary (unshielded) and shielded contributions
            E_ordinary = (
                1.0/distances
                + distances/self.long_range_cutoff_squared
                - 2.0/self.long_range_cutoff)
            E_shielded = (
                1.0/distances_shielded
                + distances_shielded/self.long_range_cutoff_squared
                - 2.0/self.long_range_cutoff)

            # Combine electrostatic contributions
            E = (
                self.kehalf*atomic_charges_i*atomic_charges_j*(
                    switch_off_weights*E_shielded
                    + switch_on_weights*E_ordinary))
            
        else:

            # Compute ordinary (unshielded) and shielded contributions
            E_ordinary = 1.0/distances
            E_shielded = 1.0/distances_shielded
            
            ## Compute ordinary (unshielded) and shielded contributions
            #E_ordinary = (
                #1.0/distances
                #+ distances/self.long_range_cutoff_squared
                #- 2.0/self.long_range_cutoff)
            #E_shielded = (
                #1.0/distances_shielded
                #+ distances_shielded/self.long_range_cutoff_squared
                #- 2.0/self.long_range_cutoff)

            # Combine electrostatic contributions
            E = (
                self.kehalf*atomic_charges_i*atomic_charges_j*(
                    switch_off_weights*E_shielded
                    + switch_on_weights*E_ordinary))

        # Apply interaction cutoff
        E = torch.where(
            distances <= self.long_range_cutoff,
            E,                      # distance <= cutoff
            torch.zeros_like(E))    # distance > cutoff

        # Sum up electrostatic atom pair contribution of each atom
        return utils.segment_sum(E, idx_i, device=self.device)
