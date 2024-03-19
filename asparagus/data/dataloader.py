import os
import logging
import numpy as np
from typing import Optional, List, Dict, Tuple, Union

import torch

from .. import data
from .. import utils

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

__all__ = ['DataLoader']

class DataLoader(torch.utils.data.DataLoader):
    """
    Data loader class from a dataset
    
    Parameters
    ----------
    dataset: (data.DataSet, data.DataSubSet)
        DataSet or DataSubSet instance of reference data
    data_batch_size: int
        Number of atomic systems per batch
    data_shuffle: bool
        Shuffle batch compilation after each epoch
    data_num_workers: int
        Number of parallel workers for collecting data
    data_collate_fn: callable, optional, default None
        Callable function that prepare and return batch data
    data_pin_memory: bool, optional, default False
        If True data are loaded to GPU

    """

    def __init__(
        self,
        dataset: Union[data.DataSet, data.DataSubSet],
        data_batch_size: int,
        data_shuffle: bool,
        data_num_workers: int,
        data_collate_fn: Optional[object] = None,
        data_pin_memory: Optional[bool] = False,
        **kwargs
    ):
        """
        Initialize data loader.
        """
        
        # Check collate function
        if data_collate_fn is None:
            data_collate_fn = self.data_collate_fn

        # Assign reference dataset as class parameter for additions by the 
        # neighbor list functions
        self.dataset = dataset
        self.data_batch_size = data_batch_size
        
        # Initiate DataLoader
        super(DataLoader, self).__init__(
            dataset=dataset,
            batch_size=data_batch_size,
            shuffle=data_shuffle,
            num_workers=data_num_workers,
            collate_fn=self.data_collate_fn,
            pin_memory=data_pin_memory,
            **kwargs
        )
        
        # Initialize neighbor list function class parameter
        self.neighbor_list = None

        return

    def init_neighbor_list(
        self,
        cutoff: Optional[float] = np.inf,
        store: Optional[bool] = False,
        func_neighbor_list: Optional[str] = 'ase',
    ):
        """
        Initialize neighbor list function
        
        Parameters
        ----------
        cutoff: float, optional, default infinity
            Neighbor list cutoff range equal to max interaction range
        store: bool, optional, default False
            Pre-compute neighbor list and store in the reference dataset
        func_neighbor_list: str, optional, default 'ase'
            Function type to compute the neighbor list

        """
        
        # Initialize neighbor list creator
        if func_neighbor_list.lower() == 'ase':
            self.neighbor_list = utils.ASENeighborList(cutoff=cutoff)
        elif func_neighbor_list.lower() == 'torch':
            self.neighbor_list = utils.TorchNeighborList(cutoff=cutoff)
        else:
            raise NotImplementedError(
                "Neighbor list functions other than from ASE ot Torch are not "
                + "implemented, yet!")

        # If demanded, apply neighbor list computation and store results in the
        # reference data set
        if store:
            self.store_neighbor_list()

        return

    def store_neighbor_list(
        self,
    ):
        """
        Compute neighbor list and store results in 

        """

        # Check cutoff range
        metadata = self.dataset.get_metadata()
        if metadata.get('cutoff') is None:
            recompute = True
        elif metadata.get('cutoff') == self.neighbor_list.cutoff:
            recompute = False
        else:
            recompute = True

        # Iterate over reference dataset
        Ndata = len(self.dataset)
        data_inds = []
        data_subset = []
        for idata in range(Ndata):

            # Get reference data 
            datai = self.dataset.get(idata)

            # Compute neighbor list data if necessary because of different
            # cutoff range or missing data
            if recompute or datai.get('idx_i') is None:
                datai = self.neighbor_list(datai)
            
                # Collect neighbor list properties
                data_subseti = {
                    prop: datai[prop] 
                    for prop in ['idx_i', 'idx_j', 'pbc_offset']}
                data_inds.append(idata)
                data_subset.append(data_subseti)

            # Add to reference dataset every batch size
            if not (idata + 1)%self.data_batch_size:

                self.dataset.update_properties(
                    idx=data_inds,
                    properties=data_subset)
                data_inds = []
                data_subset = []

        # Add to reference dataset remaining
        if len(data_inds):
            self.dataset.update_properties(
                idx=data_inds,
                properties=data_subset)

        # Update cutoff range in metadata
        if recompute:
            metadata = self.dataset.get_metadata()
            metadata['cutoff'] = self.neighbor_list.cutoff
            self.dataset.set_metadata(metadata=metadata)

        return

    def data_collate_fn(
        self,
        batch: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Prepare batch properties from a dataset such as pair indices and
        return with system properties
        
        Parameters
        ----------
        batch: dict
            Data batch
        
        Returns
        -------
        dict
            Collated data batch with additional properties
            Properties:
                atoms_number: (Natoms,) torch.Tensor
                atomic_numbers: (Natoms,) torch.Tensor
                positions: (Natoms, 3) torch.Tensor
                forces: (Natoms, 3) torch.Tensor
                energy: (Natoms,) torch.Tensor
                charge: (Natoms,) torch.Tensor
                idx_i: (Npairs,) torch.Tensor
                idx_j: (Npairs,) torch.Tensor
                pbc_offset: (Npairs, 3) torch.Tensor
                sys_i: (Natoms,) torch.Tensor
                sys_ij: (Npairs,) torch.Tensor

        """

        # Collected batch system properties
        coll_batch = {}

        # Get batch size a.k.a. number of systems
        Nsys = len(batch)

        # Get atoms number per system segment
        coll_batch['atoms_number'] = torch.tensor(
            [b['atoms_number'] for b in batch])

        # System segment index of atom i
        coll_batch['sys_i'] = torch.repeat_interleave(
            torch.arange(Nsys, dtype=torch.int64), 
            repeats=coll_batch['atoms_number'], dim=0)

        # Atomic numbers properties
        coll_batch['atomic_numbers'] = torch.cat(
            [b['atomic_numbers'] for b in batch], 0).to(torch.int64)

        # Periodic boundary conditions
        coll_batch['pbc'] = torch.cat(
            [b['pbc'] for b in batch], 0).to(torch.bool).reshape(Nsys, 3)

        # Unit cell sizes
        coll_batch['cell'] = torch.cat(
            [b['cell'] for b in batch], 0).reshape(Nsys, -1)

        # Compute the cumulative segment size number
        atomic_numbers_cumsum = torch.cat(
            [
                torch.zeros((1,), dtype=coll_batch['sys_i'].dtype),
                torch.cumsum(coll_batch["atoms_number"][:-1], dim=0)
            ],
            dim=0)
        
        # Prepare neighbor list calculator
        if self.neighbor_list is None:
            self.init_neighbor_list()

        # Iterate over batch properties
        skip_props = ['atoms_number', 'atomic_numbers', 'pbc', 'cell']
        for prop_i in batch[0]:
            
            # Skip previous parameter and None
            if prop_i in skip_props or batch[0].get(prop_i) is None:

                continue

            # Pair indices (integer data)
            elif prop_i in ['idx_i', 'idx_j']:

                try:
                    coll_batch[prop_i] = torch.cat(
                        [
                            b[prop_i].to(torch.int64) + off 
                            for b, off in zip(batch, atomic_numbers_cumsum)
                        ],
                        dim=0).to(torch.int64)
                except AttributeError:
                    coll_batch = self.neighbor_list(
                        coll_batch,
                        atomic_numbers_cumsum=atomic_numbers_cumsum)
                    logger.warning(
                        "Most likely, pair indices of one data frame is None, "
                        + "because the data seed or dataset has changed after "
                        + "storing pair indices just for a sub set of data "
                        + "like training or validation.")

            # Properties (float data)
            else:

                # Concatenate tensor data
                if batch[0][prop_i].size():
                    coll_batch[prop_i] = torch.cat(
                        [b[prop_i] for b in batch], 0).to(torch.float64)

                # Concatenate numeric data
                else:
                    coll_batch[prop_i] = torch.tensor(
                        [b[prop_i] for b in batch]).to(torch.float64)

        # Compute pair indices and position offsets if not available
        if batch[0].get('idx_i') is None:
            coll_batch = self.neighbor_list(
                coll_batch,
                atomic_numbers_cumsum=atomic_numbers_cumsum)

        # Get number of atom pairs per system segment
        Npairs = torch.tensor(coll_batch['idx_i'].shape[0])

        # System segment index of atom pair i,j
        sys_ij = torch.repeat_interleave(
            torch.arange(Nsys, dtype=torch.int64), repeats=Npairs, dim=0)

        # Add atom pairs segment index to collective batch
        coll_batch['sys_ij'] = sys_ij

        return coll_batch

