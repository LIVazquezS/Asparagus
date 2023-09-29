import os
import sys
import json
import logging
import functools
from typing import Optional, List, Dict, Tuple, Union, Any

from ase.parallel import world, DummyMPI, parallel_function, parallel_generator
from ase.utils import Lock

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
    

def lock(method):
    """
    Decorator for using a lock-file.
    """
    @functools.wraps(method)
    def new_method(self, *args, **kwargs):
        if self.lock is None:
            return method(self, *args, **kwargs)
        else:
            with self.lock:
                return method(self, *args, **kwargs)
    return new_method



class DataBase:
    """
    Base class for the database
    """
    
    def __init__(
        self, 
        data_file: str,
        data_lock_file: bool,
    ):
        """
        DataBase object that contain reference data.
        This is a condensed version of the ASE Database class:
        https://gitlab.com/ase/ase/-/blob/master/ase/db/core.py

        Parameters
        ----------
            data_file: str
                Reference database file
            data_lock_file: bool
                Use a lock file when manipulating the database to prevent
                parallel manipulation by multiple processes.
                
        Returns
        -------
            object
                DataBase for data storing
        """
        
        # DataBase file name
        if utils.is_string(data_file):
            data_file = os.path.expanduser(data_file)
        self.data_file = data_file
        
        # Prepare data locker
        if data_lock_file and utils.is_string(data_file):
            self.lock = Lock(data_file + '.lock', world=DummyMPI())
        else:
            self.lock = None


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
        
    
    @parallel_function
    @lock
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
    

    def __delitem__(self, rwo_id):
        self.delete([rwo_id])
        
    
    def __getitem__(self, selection):
        return self.get(selection)
    

    def get(self, selection=None, **kwargs):
        """
        Select a single row and return it as a dictionary.
        
        Parameters
        ----------
            selection: int, str or list
                See the select() method.
        
        Returns
        -------
            dict
                Returns entry of the selection.
        """
        
        # Get row of selection
        row = list(self.select(selection, **kwargs))
        
        # Check selection results
        if row is None:
            raise KeyError('no match')
        
        return row
    
    
    def parse_selection(self, selection, **kwargs):
        """
        Interpret the row selection
        """
        
        if selection is None or selection == '':
            cmps = []
        elif utils.is_integer(selection):
            cmps = [('id', '=', selection)]
        elif utils.is_integer_array(selection):
            cmps = [('id', '=', selection_i) for selection_i in selection]
        else:
            raise ValueError(
                f"Database selection '{selection}' is not a valid input!\n" +
                f"Provide either an index or list of indices.")
        
        return cmps
    
    @parallel_generator
    def select(
        self, 
        selection=None, 
        selection_filter=None, 
        **kwargs):
        """
        Select rows.

        Return AtomsRow iterator with results.  Selection is done
        using key-value pairs.
        
        Parameters
        ----------
            selection: int or list of int
                Row index or list of indices
                
            selection_filter: function
                A function that takes as input a row and returns True or False.
                
        Returns
        -------
            dict
                Returns entry of the selection.
        """
        
        # Check and interpret selection
        cmps = self.parse_selection(selection)
        
        # Iterate over selection
        #row = list(self._select(cmps))
        for row in self._select(cmps):
        
            # Apply potential reference data filter or not 
            if selection_filter is None or selection_filter(row):
                    
                yield row
                
        
    def _select(self, cmps):
        
        raise NotImplementedError
    
    
    def count(self, selection=None, **kwargs):
        """
        Count rows in DataBase
        """
        
        # Check selection
        cmps = self.parse_selection(selection)
        
        # Count rows
        return self._count(cmps)


    def _count(self, cmps):
        
        n = 0
        for row in self._select(cmps):
            n += 1
        return n


    def __len__(self):
        return self.count()


    def delete(self, row_ids):
        
        raise NotImplementedError
    
    
    @parallel_function
    @lock
    def reserve(self):
        """
        Write empty row if not already present.
        """
        
        # Write empty row
        row_id = self._write({}, None)
        
        return row_id
    
    
    def get_metadata(self):
        """
        Return metadata of the database
        """
        return self._get_metadata()
    
    
    def _get_metadata(self):
        raise NotImplementedError


def object_to_bytes(
    obj: Any
    ) -> bytes:
    """
    Serialize Python object to bytes.
    """
    
    parts = [b'12345678']
    obj = o2b(obj, parts)
    offset = sum(len(part) for part in parts)
    x = np.array(offset, np.int64)
    if not np.little_endian:
        x.byteswap(True)
    parts[0] = x.tobytes()
    parts.append(json.dumps(obj, separators=(',', ':')).encode())
    return b''.join(parts)


def bytes_to_object(
    b: bytes
    ) -> Any:
    """
    Deserialize bytes to Python object.
    """
    
    x = np.frombuffer(b[:8], np.int64)
    if not np.little_endian:
        x = x.byteswap()
    offset = x.item()
    obj = json.loads(b[offset:].decode())
    return b2o(obj, b)


def o2b(
    obj: Any, 
    parts: List[bytes]
    ):
    
    if (
        obj is None 
        or utils.is_numeric(obj) 
        or utils.is_bool(obj) 
        or utils.is_string(obj)
        ):
        return obj
    
    if utils.is_dictionary(obj):
        return {key: o2b(value, parts) for key, value in obj.items()}
    
    if isinstance(obj, (list, tuple)):
        return [o2b(value, parts) for value in obj]
    
    if isinstance(obj, np.ndarray):
        
        assert obj.dtype != object, \
            'Cannot convert ndarray of type "object" to bytes.'
        offset = sum(len(part) for part in parts)
        if not np.little_endian:
            obj = obj.byteswap()
        parts.append(obj.tobytes())
        return {'__ndarray__': [obj.shape, obj.dtype.name, offset]}
    
    if isinstance(obj, complex):
        return {'__complex__': [obj.real, obj.imag]}
    
    objtype = type(obj)
    raise ValueError(
        f"Objects of type {objtype} not allowed")


def b2o(
    obj: Any, 
    b: bytes
    ) -> Any:
    
    if isinstance(obj, (int, float, bool, str, type(None))):
        return obj

    if isinstance(obj, list):
        return [b2o(value, b) for value in obj]

    assert isinstance(obj, dict)

    x = obj.get('__complex__')
    if x is not None:
        return complex(*x)

    x = obj.get('__ndarray__')
    if x is not None:
        shape, name, offset = x
        dtype = np.dtype(name)
        size = dtype.itemsize * np.prod(shape).astype(int)
        a = np.frombuffer(b[offset:offset + size], dtype)
        a.shape = shape
        if not np.little_endian:
            a = a.byteswap()
        return a

    dct = {key: b2o(value, b) for key, value in obj.items()}
    objtype = dct.pop('__ase_objtype__', None)
    if objtype is None:
        return dct
    return create_ase_object(objtype, dct)
