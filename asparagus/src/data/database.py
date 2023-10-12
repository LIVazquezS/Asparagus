import os
import sys
import json
import logging
from typing import Optional, List, Dict, Tuple, Union, Any

from ase.parallel import world, DummyMPI, parallel_function, parallel_generator

import numpy as np

import torch
torch.multiprocessing.set_sharing_strategy('file_system')

from .. import data
from .. import utils

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

__all__ = ['DataBase', 'connect']

def connect(data_file, data_lock_file=True):
    """
    Create connection to database.

    Parameters
    ----------
        data_file: str
            Database file path
        data_lock_file: bool
            Use a lock file
            
    Returns
    -------
        object
            Database interface object
    """
    return data.DataBase_SQLite3(
        data_file, data_lock_file)


class DataBase:
    """
    Base class for the database
    """
    
    def __init__(
        self, 
        data_file: str,
    ):
        """
        DataBase object that contain reference data.
        This is a condensed version of the ASE Database class:
        https://gitlab.com/ase/ase/-/blob/master/ase/db/core.py

        Parameters
        ----------
        data_file: str
            Reference database file
                
        Returns
        -------
            object
                DataBase for data storing
        """
        
        # DataBase file name
        if utils.is_string(data_file):
            data_file = os.path.expanduser(data_file)
        self.data_file = data_file
        
    def set_metadata(self, metadata):
        """
        Set database metadata.
        
        Parameters
        ----------
            metadata: dict
                Database metadata
        
        Returns
        -------
            dict
                Metadata stored in Database
                If Database is new output is same as input.
        """
        
        return self._set_metadata(metadata)

    def _set_metadata(self, metadata):
        raise NotImplementedError

    @property
    def metadata(self) -> Dict[str, Any]:
        self._get_metadata()

    def get_metadata(self):
        """
        Get the database metadata dictionary

        Returns
        -------
            dict
                Metadata stored in Database
                If Database is new output is same as input.
        """

        return self._get_metadata()

    def _get_metadata(self):
        raise NotImplementedError

    def init_systems(self):
        """
        Initialize systems column in database according to metadata
        """
        return self._init_systems()

    def _init_systems(self):
        raise NotImplementedError

    def write(self, properties={}, row_id=None, **kwargs):
        """
        Write reference data to database.
        
        Parameters
        ----------
            properties: dict
                Reference data
            row_id: int
                Overwrite existing row.

        Returns
        -------
            int
                Returns integer id of the new row.
        """

        row_id = self._write(properties, row_id)
        
        return row_id

    def _write(self, properties, row_id=None):        
        return 1

    @parallel_function
    def update(self, row_id, properties={}, **kwargs):
        """
        Update reference data of certain properties in database.
        
        Parameters
        ----------
            properties: dict
                Reference data
            row_id: int
                Overwrite existing row.

        Returns
        -------
            int
                Returns integer id of the new row.
        """

        row_id = self._update(row_id, properties)

        return row_id

    def _update(self, row_id, properties):
        return 1

    def __delitem__(self, rwo_id):
        self._delete([rwo_id])

    def __getitem__(self, selection):
        return self.get(selection)

    def get(self, selection=None, **kwargs):
        """
        Select a single or multiple rows and return it as a dictionary.
        
        Parameters
        ----------
        selection: (int, list(int))
            See the select() method.
        
        Returns
        -------
        dict
            Returns entry of the selection.
        """
        
        return self._get(selection, **kwargs)

    def _get(self, selection, **kwargs):
        raise NotImplementedError

    def count(self, selection=None, **kwargs):
        """
        Count rows in DataBase
        """
        return self._count(selection, **kwargs)
        
    def _count(self, selection, **kwargs):
        raise NotImplementedError

    def __len__(self):
        return self.count()

    def delete(self, row_ids):
        """
        Delete entry from the database.
        """
        if utils.is_integer(row_ids):
            row_ids = [row_ids]
        self._delete(row_ids)

    def _delete(self, row_ids):
        raise NotImplementedError

    @parallel_function
    def reserve(self):
        """
        Write empty row if not already present.
        """

        # Write empty row
        row_id = self._write({}, None)

        return row_id

    def _get_metadata(self):
        raise NotImplementedError
