import os
import time
import json
import logging
import functools
from typing import Optional, Union, List, Dict, Tuple, Callable, Any

from ase.parallel import DummyMPI, parallel_function, parallel_generator
from ase.utils import Lock

import numpy as np

import torch

from .. import data
from .. import utils

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

__all__ = ['DataBase_npz']

# Current npz database version
VERSION = 0

# Structural property labels and dtypes
structure_properties_dtype = {
    'atoms_number':     np.int32,
    'atomic_numbers':   np.int32,
    'positions':        np.float32,
    'charge':           np.float32,
    'cell':             np.float32,
    'pbc':              np.bool_,
    'idx_i':            np.int32,
    'idx_j':            np.int32,
    'pbc_offset':       np.float32,
}
properties_numpy_dtype = np.float64
reference_properties_torch_dtype = torch.float64

# Structural property labels and array shape
structure_properties_shape = {
    'atoms_number':     (-1,),
    'atomic_numbers':   (-1,),
    'positions':        (-1, 3,),
    'charge':           (-1,),
    'cell':             (-1,),
    'pbc':              (1, 3,),
    'idx_i':            (-1,),
    'idx_j':            (-1,),
    'pbc_offset':       (-1, 3,),
}
reference_properties_shape = {
    # 'energy':           (-1,),
    # 'atomic_energies':  (-1,),
    'forces':           (-1, 3,),
    # 'hessian':          (-1,),
    # 'atomic_charge':    (-1,),
    # 'dipole':           (3,),
    # 'atomic_dipoles':   (-1,),
    'polarizability':   (3, 3,),
    }


def connect(
    data_file,
    lock_file: Optional[bool] = True,
    **kwargs,
) -> object:
    """
    Create connection to database.

    Parameters
    ----------
    data_file: str
        Database file path
    lock_file: bool, optional, default True
        Use a lock file

    Returns
    -------
    data.Database_npz
        Numpy npz database interface object

    """
    return DataBase_npz(data_file, lock_file)


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


class DataBase_npz(data.DataBase):
    """
    Numpy npz data base class
    """

    _metadata = {}

    def __init__(
        self,
        data_file: str,
        lock_file: bool,
    ):
        """
        Numpy Database object that contain reference data.

        Parameters
        ----------
        data_file: str
            Reference database file
        lock_file: bool
            Use a lock file when manipulating the database to prevent
            parallel manipulation by multiple processes.

        """

        # Inherit from DataBase base class
        super().__init__(data_file)

        # Check for .npz suffix
        if self.data_file.split('.')[-1].lower != 'npz':
            self.data_file = f"{self.data_file:s}.npz"

        # Prepare data locker
        if lock_file and utils.is_string(data_file):
            self.lock = Lock(self.data_file + '.lock', world=DummyMPI())
        else:
            self.lock = None

        # Prepare metadata file path
        self.metadata_file = os.path.join(f"{self.file_name:s}.json")
        self.metadata_indent = 2

        return

    def _load(self):
        if os.path.exists(self.data_file):
            self.data = np.load(self.data_file)
            self.next_id = self.data['id'][-1] + 1
        else:
            self.data = {}
            self.next_id = 1
        self.data_new = {}
        return

    def _save(self):
        if self.data_new:
            self._merge()
            np.savez(self.data_file, **self.data)
        return

    def _merge(self):
        for prop in self.data:
            if self.data_new.get(prop) is None:
                continue
            self.data[prop] = np.concatenate(
                (self.data[prop], *self.data_new[prop]),
                axis=0)
            self.data_new = {}
        return

    def __enter__(self):
        self._load()
        return

    def __exit__(self, exc_type, exc_value, tb):
        if exc_type is not None:
            raise exc_type
        else:
            self._save()
        return

    @lock
    def _set_metadata(self, metadata):

        # Store metadata to file
        json.dumps(
            self.metadata_file,
            metadata,
            indent=self.metadata_indent,
            default=str)

        # Store metadata as class variable
        self._metadata = metadata

        return self._metadata

    def _get_metadata(self):

        # Read metadata
        metadata = json.load(self.metadata_file)

        return metadata

    def _init_systems(self):
        return

    def _init_data(self, data):

        # Initialize current datatime and User name
        data['mtime'] = []
        data['username'] = []

        # Initialize structural properties
        for prop_i in structure_properties_dtype:
            data[prop_i] = []

        # Initialize reference properties
        for prop_i in self.metadata.get('load_properties'):
            if prop_i not in structure_properties_dtype:
                data[prop_i] = []

        return data

    def _reset(self):

        # Reset stored metadata dictionary
        self._metadata = {}
        return

    @lock
    def _write(self, properties, row_id):

        # Initialize new data dictionary
        if not self.data_new:
            self.data_new = self.init_data(self.data_new)

        # Current datatime and User name
        self.data_new['mtime'].append(time.ctime())
        self.data_new['username'].append(os.getenv('USER'))

        # Structural properties
        for prop_i, dtype_i in structure_properties_dtype.items():
            if properties.get(prop_i) is None:
                self.data_new[prop_i].append(None)
            else:
                self.data_new[prop_i].append(
                    np.array([properties.get(prop_i)], dtype=dtype_i))

        # Reference properties
        for prop_i in self.metadata.get('load_properties'):
            if prop_i not in structure_properties_dtype:
                if properties.get(prop_i) is None:
                    self.data_new[prop_i].append(None)
                else:
                    self.data_new[prop_i].append(
                        np.array([properties.get(prop_i)], dtype=dtype_i))

        # Add or update database values
        if row_id is None:
            row_id = self.next_id
        row_id = self._update(row_id, data_new=self.data_new)

        return row_id

    def _update(self, row_id, properties=None):

        # Initialize new data dictionary
        if not self.data_new:
            data_new = self.init_data(data_new)

        # Current datatime and User name
        self.data_new['mtime'].append(time.ctime())
        self.data_new['username'].append(os.getenv('USER'))

        # Structural properties
        for prop_i, dtype_i in structure_properties_dtype.items():
            if properties.get(prop_i) is None:
                self.data_new[prop_i].append(None)
            else:
                self.data_new[prop_i].append(
                    np.array([properties.get(prop_i)], dtype=dtype_i))

        # Reference properties
        for prop_i in self.metadata.get('load_properties'):
            if prop_i not in structure_properties_dtype:
                if properties.get(prop_i) is None:
                    self.data_new[prop_i].append(None)
                else:
                    self.data_new[prop_i].append(
                        np.array([properties.get(prop_i)], dtype=dtype_i))

        if not new_data:

            raise SyntaxError(
                "At least one input 'ref_data' or 'properties' should "
                + "contain reference data!")

        elif ref_data is None:

            row_id = self._write(properties, row_id)

        else:

            # Add or update database values
            with self.managed_connection() as data:

                key_id = f"id{row_id:d}"
                if not isinstance(self.data, dict):
                    self.data = dict(self.data)
                self.data[key_id] = ref_data

        return row_id

    def get_last_id(self):

        data_files = self.data.files

        # Get last row id
        if 'last_id' in data_files:
            row_id = self.data['last_id']
            next_key_id = f"id{row_id+1:d}"
        else:
            row_id = 0
            next_key_id = f"id{row_id+1:d}"

        # Check last row id
        while next_key_id in data_files:
            row_id += 1
            next_key_id = f"id{row_id+1:d}"

        return row_id

    def _select(self, cmps, verbose=False):

        sql, args = self.create_select_statement(cmps)
        metadata = self._get_metadata()

        with self.managed_connection() as con:

            # Execute SQL request
            cur = con.cursor()
            cur.execute(sql, args)

            for row in cur.fetchall():

                yield self.convert_row(row, metadata, verbose=verbose)

    def _count(self, cmps):

        sql, args = self.create_select_statement(cmps, what='COUNT(*)')

        with self.managed_connection() as con:
            cur = con.cursor()
            try:
                cur.execute(sql, args)
                return cur.fetchone()[0]
            except sqlite3.OperationalError:
                return 0

    def delete(self, row_ids):
        """
        Delete rows.

        Parameters
        ----------
            row_ids: int or list of int
                Row index or list of indices to delete
        """

        if len(row_ids) == 0:
            return

        self._delete(row_ids)

    def _delete(self, row_ids):
        raise NotImplementedError()

    @property
    def metadata(self):
        return self._get_metadata()
