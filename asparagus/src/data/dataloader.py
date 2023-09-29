import os
import logging
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
    dataset: DataSet object
        DataSet or DataSubSet instance
    data_batch_size: int
        Number of atomic systems per batch
    data_shuffle: bool
        Shuffle batch compilation after each go-through 
    data_num_workers: int
        Number of parallel workers for collecting data
    data_collate_fn: callable
        Callable function that prepare and return batch data
    data_pin_memory: bool, optional False
        If True data are loaded to GPU
    
    Returns
    -------
        DataLoader object
            DataLoader object iterating over dataset batches
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
        
        # Check collate function
        if data_collate_fn is None:
            data_collate_fn = self.data_collate_fn
        
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

    def init_neighbor_list(
        self,
        cutoff: Optional[float] = 100.0,
        func_neighbor_list: Optional[str] = 'ase',
    ):
        """
        Initialize neighbor list function
        """
        
        # Initialize neighbor list creator
        if func_neighbor_list.lower() == 'ase':
            self.neighbor_list = utils.ASENeighborList(cutoff=cutoff)
        else:
            raise NotImplementedError(
                "Neighbor list functions other than from ASE are not"
                + "implemented, yet!")

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
                    atoms_number: (N,) torch.Tensor
                    atomic_numbers: (N,) torch.Tensor
                    positions: (N,3) torch.Tensor
                    forces: (N,3) torch.Tensor
                    energy: (N,) torch.Tensor
                    charge: (N,) torch.Tensor
                    atoms_seg: (N,) torch.Tensor
                    pairs_seg: (N,) torch.Tensor
                    idx_i: (N,) torch.Tensor
                    idx_j: (N,) torch.Tensor

        """

        # Collected batch system properties
        coll_batch = {}

        # Get batch batch_size
        Nbatch = len(batch)

        # Get atoms number per system segment
        coll_batch["atoms_number"] = torch.tensor(
            [b["atoms_number"] for b in batch])

        # System segment index of atom i
        coll_batch["atoms_seg"] = torch.repeat_interleave(
            torch.arange(Nbatch, dtype=torch.int64), 
            repeats=coll_batch["atoms_number"], dim=0)

        # Iterate over batch properties
        for prop_i in batch[0]:

            # Pass pair index properties for now
            if prop_i in ["idx_i", "idx_j", "pbc_offset"]:
                
                pass

            # Atomic numbers properties
            elif prop_i == "atomic_numbers":
                coll_batch[prop_i] = torch.cat([b[prop_i] for b in batch], 0).to(
                    torch.int64)

            else:

                # Continue if reference data are not available
                if batch[0][prop_i] is None:
                    continue

                # Concatenate tensor data
                elif batch[0][prop_i].size():
                    coll_batch[prop_i] = torch.cat(
                        [b[prop_i] for b in batch], 0).to(torch.float64)

                # Concatenate numeric data
                else:
                    coll_batch[prop_i] = torch.tensor(
                        [b[prop_i] for b in batch]).to(torch.float64)

        # Check pair index properties
        if batch[0].get("idx_i") is None:
            
            # Compute pair indices and position offsets
            if self.neighbor_list is None:
                self.init_neighbor_list
            coll_batch = self.neighbor_list(coll_batch)
        
        # Get number of atom pairs per system segment
        Npairs = torch.tensor(len(coll_batch["idx_i"]))

        # System segment index of atom pair i,j
        idx_seg = torch.repeat_interleave(
            torch.arange(Nbatch, dtype=torch.int64), repeats=Npairs, dim=0)

        # Add atom pairs segment index to collective batch
        coll_batch["pairs_seg"] = idx_seg
        
        return coll_batch

