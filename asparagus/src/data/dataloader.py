import os
import logging
from typing import Optional, List, Dict, Tuple, Union

import torch
#from torch.utils.data.dataloader import _collate_fn_t, T_co

from .. import data

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

__all__ = ['DataLoader']


def data_collate_fn(batch):
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
    Natoms = torch.tensor([b["atoms_number"] for b in batch])

    # Get segment size cumulative sum
    Natoms_cumsum = torch.cat(
        [torch.zeros((1,), dtype=Natoms.dtype), torch.cumsum(Natoms, dim=0)], 
        dim=0)

    # System segment index of atom i
    idx_seg = torch.repeat_interleave(
        torch.arange(Nbatch, dtype=torch.int64), repeats=Natoms, dim=0)

    # Add atoms segment index to collective batch
    coll_batch["atoms_seg"] = idx_seg

    # Get number of atom pairs per system segment
    Npairs = torch.tensor([len(b["idx_i"]) for b in batch])

    # System segment index of atom pair i,j
    idx_seg = torch.repeat_interleave(
        torch.arange(Nbatch, dtype=torch.int64), repeats=Npairs, dim=0)

    # Add atom pairs segment index to collective batch
    coll_batch["pairs_seg"] = idx_seg

    # Iterate over batch properties
    for prop_i in batch[0]:

        # For pair index properties
        if prop_i in ["idx_i", "idx_j"]:

            coll_batch[prop_i] = torch.cat(
                [b[prop_i] + off for b, off in zip(batch, Natoms_cumsum)],
                0).to(torch.int64)

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

    return coll_batch


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
        If True data are loaded to GPU (TODO)
    
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
        data_collate_fn: object = data_collate_fn,
        data_pin_memory: bool = False,
        **kwargs
    ):
        
        super(DataLoader, self).__init__(
            dataset=dataset,
            batch_size=data_batch_size,
            shuffle=data_shuffle,
            #sampler=sampler,
            #batch_sampler=batch_sampler,
            num_workers=data_num_workers,
            collate_fn=data_collate_fn,
            pin_memory=data_pin_memory,
            **kwargs
        )
