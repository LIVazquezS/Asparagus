import os
import sys
import time
import json
import sqlite3
import logging
from typing import Optional, List, Dict, Tuple, Union, Any
from contextlib import contextmanager

from ase.parallel import world, DummyMPI, parallel_function, parallel_generator
from ase.utils import Lock

import numpy as np

import torch

from .. import data
from .. import utils
from .. import settings

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

__all__ = ['DataBase_SQLlite3']

# Current SQLite3 database version
VERSION = 0

all_tables = ['systems']


# Initial SQL statement lines
init_systems = [
    """CREATE TABLE systems (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    mtime TEXT,
    username TEXT,
    atoms_number BLOB,
    atomic_numbers BLOB,
    positions BLOB,
    charge BLOB,
    cell BLOB,
    pbc BLOB,
    idx_i BLOB,
    idx_j BLOB,
    pbc_offset BLOB,
    """]

init_information = [
    """CREATE TABLE information (
    name TEXT,
    value TEXT)""",

    "INSERT INTO information VALUES ('version', '{}')".format(VERSION)]

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
properties_numpy_dtype = np.float32
properties_torch_dtype = torch.float32

# Structural property labels and array shape
structure_properties_shape = {
    'atoms_number':     (),
    'atomic_numbers':   (-1,),
    'positions':        (-1, 3,),
    'charge':           (-1,),
    'cell':             (1, 3,),
    'pbc':              (1, 3,),
    'idx_i':            (-1,),
    'idx_j':            (-1,),
    'pbc_offset':       (-1, 3,),
}


class DataBase_SQLite3(data.DataBase):
    """
    SQL lite 3 data base class
    """
    
    # Initialize connection interface
    connection = None
    _metadata = None
    
    # Used for autoincrement id
    default = 'NULL'
    
    #def __init__(
        #self, 
        #data_file, 
        #data_lock_file
    #):
        #"""
        #SQLite3 dataBase object that contain reference data.
        #This is a condensed version of the ASE Database class:
        #https://gitlab.com/ase/ase/-/blob/master/ase/db/sqlite.py

        #Parameters
        #----------
            #data_file: str
                #Reference database file
            #data_lock_file: bool
                #Use a lock file when manipulating the database to prevent
                #parallel manipulation by multiple processes.
                
        #Returns
        #-------
            #object
                #SQLite3 dataBase for data storing
        #"""
        
        ## Inherit from DataBase base class
        #super().__init__(data_file, data_lock_file)
        
        #return
    
    def _connect(self):
        return sqlite3.connect(self.data_file, timeout=20)


    def __enter__(self):
        assert self.connection is None
        self.change_count = 0
        self.connection = self._connect()
        return self


    def __exit__(self, exc_type, exc_value, tb):
        if exc_type is None:
            self.connection.commit()
        else:
            self.connection.rollback()
        self.connection.close()
        self.connection = None
    
    
    @contextmanager
    def managed_connection(self, commit_frequency=5000):
        try:
            con = self.connection or self._connect()
            yield con
        except ValueError as exc:
            if self.connection is None:
                con.close()
            raise exc
        else:
            if self.connection is None:
                con.commit()
                con.close()
            else:
                self.change_count += 1
                if self.change_count % commit_frequency == 0:
                    con.commit()


    def _set_metadata(self, metadata):
        
        # Convert metadata dictionary
        md = json.dumps(metadata)
        
        with self.managed_connection() as con:
            
            # Select any data in the database
            cur = con.execute(
                "SELECT COUNT(*) FROM sqlite_master WHERE name='information'")
            
            if cur.fetchone()[0]:
                
                # Update metadata if existing
                cur.execute(
                    "UPDATE information SET value=? WHERE name='metadata'", 
                    [md])
                
            else:
                
                # Initialize data columns 
                for statement in init_information:
                    con.execute(statement)
                con.commit()
                
                # Write metadata 
                cur.execute(
                    "INSERT INTO information VALUES (?, ?)", ('metadata', md))
    
    
    def _get_metadata(self):
        
        # Read metadata if not in memory
        if self._metadata is None:
            
            with self.managed_connection() as con:
                
                cur = con.execute(
                    'SELECT value FROM information WHERE name="metadata"')
                results = cur.fetchall()
                if results:
                    self._metadata = json.loads(results[0][0])
        
        return self._metadata
    
    
    def _init_systems(self):
        
        # Get metadata
        metadata = self._get_metadata()
        
        with self.managed_connection() as con:
            
            # Select any data in the database
            cur = con.execute(
                "SELECT COUNT(*) FROM sqlite_master WHERE name='systems'")
            
            # If no system in database
            if cur.fetchone()[0] == 0:
                
                # Update initial statements with properties to load
                init_systems_execute = init_systems[:]
                for prop_i in metadata.get('load_properties'):
                    if prop_i not in structure_properties_dtype.keys():
                        init_systems_execute[0] += f"{prop_i} BLOB,\n"
                init_systems_execute[0] = init_systems_execute[0][:-2] + ")"
                
                # Initialize data columns 
                for statement in init_systems_execute:
                    con.execute(statement)
                con.commit()
                
                self.version = VERSION
            
            # Else get information from database
            else:
                
                cur = con.execute(
                    'SELECT value FROM information WHERE name="version"')
                self.version = int(cur.fetchone()[0])
            
        # Check version compatibility
        if self.version > VERSION:
            raise IOError(
                f"Can not read newer version of the database format "
                f"(version {self.version}).")
        
        return
    
    
    def encode(self, obj):
        return data.object_to_bytes(obj)
        
        
    def decode(self, txt):
        return data.bytes_to_object(txt)
        

    def blob(self, item):
        """
        Convert an item to blob/buffer object if it is an array.
        """

        if item is None:
            return None
        elif utils.is_integer(item):
            return item
        elif utils.is_numeric(item):
            return item
        
        if item.dtype == np.int64:
            item = item.astype(np.int32)
        if item.dtype == torch.int64:
            item = item.astype(torch.int32)
        if not np.little_endian:
            item = item.byteswap()
        return memoryview(np.ascontiguousarray(item))
        

    def deblob(self, buf, dtype=torch.float, shape=None):
        """
        Convert blob/buffer object to ndarray of correct dtype and shape.
        (without creating an extra view).
        """
        
        if buf is None:
            return None
        
        if len(buf) == 0:
            item = np.zeros(0, dtype)
        else:
            item = np.frombuffer(buf, dtype)
            if not np.little_endian:
                item = item.byteswap()
        
        if shape is not None:
            item = item.reshape(shape)
            
        return item


    def _write(self, properties, row_id):
        
        # Reference data list
        columns = []
        values = []
        
        # Current datatime and User name
        columns += ['mtime', 'username']
        values += [time.ctime(), os.getenv('USER')]

        # Structural properties
        structure_values = []
        for prop_i, dtype_i in structure_properties_dtype.items():

            columns += [prop_i]
            if properties.get(prop_i) is None:
                values += [None]
            elif utils.is_array_like(properties.get(prop_i)):
                values += [self.blob(
                    np.array(properties.get(prop_i), dtype=dtype_i))]
            else:
                values += [dtype_i(properties.get(prop_i))]
        
        # Reference properties
        for prop_i in self.metadata.get('load_properties'):

            if prop_i not in structure_properties_dtype.keys():
                
                columns += [prop_i]
                if properties.get(prop_i) is None:
                    values += [None]
                elif utils.is_array_like(properties.get(prop_i)):
                    values += [self.blob(
                        np.array(
                            properties.get(prop_i),
                            dtype=properties_numpy_dtype))]
                else:
                    values += [properties_numpy_dtype(properties.get(prop_i))]

        # Convert values to tuple
        columns = tuple(columns)
        values = tuple(values)
        
        # Add or update database values
        with self.managed_connection() as con:
            
            # Add values to database
            if row_id is None:
                
                # Get SQL cursor
                cur = con.cursor()
                
                # Add to database
                q = self.default + ', ' + ', '.join('?' * len(values))
                cur.execute(
                    f"INSERT INTO systems VALUES ({q})", values)
                row_id = self.get_last_id(cur)
                
            else:

                row_id = self._update(row_id, values=values, columns=columns)
                
        return row_id


    def update(self, row_id, properties):

        # Reference data list
        columns = []
        values = []

        # Current datatime and User name
        columns += ['mtime', 'username']
        values += [time.ctime(), os.getenv('USER')]

        # Structural properties
        for prop_i, dtype_i in structure_properties_dtype.items():

            if prop_i in properties:
                columns += [prop_i]
                if properties.get(prop_i) is None:
                    values += [None]
                elif utils.is_array_like(properties.get(prop_i)):
                    values += [self.blob(
                        np.array(properties.get(prop_i), dtype=dtype_i))]
                else:
                    values += [dtype_i(properties.get(prop_i))]

        # Reference properties
        for prop_i in self.metadata.get('load_properties'):

            if (
                    prop_i in properties
                    and prop_i not in structure_properties_dtype
            ):
                
                columns += [prop_i]
                if properties.get(prop_i) is None:
                    values += [None]
                elif utils.is_array_like(properties.get(prop_i)):
                    values += [self.blob(
                        np.array(
                            properties.get(prop_i),
                            dtype=properties_numpy_dtype))]
                else:
                    values += [properties_numpy_dtype(properties.get(prop_i))]
        
        # Convert values to tuple
        columns = tuple(columns)
        values = tuple(values)

        # Add or update database values
        with self.managed_connection() as con:
            
            row_id = self._update(row_id, values=values, columns=columns)

        return row_id


    def _update(self, row_id, values=None, columns=None, properties=None):
        
        if values is None and properties is None:
            
            raise SyntaxError(
                "At least one input 'values' or 'properties' should "
                + "contain reference data!")
        
        elif values is None:
            
            row_id = self._write(properties, row_id)
            
        else:
        
            # Add or update database values
            with self.managed_connection() as con:
                
                # Get SQL cursor
                cur = con.cursor()
                
                # Update values in database
                q = ', '.join([f'{column:s} = ?' for column in columns])
                cur.execute(
                    f"UPDATE systems SET {q} WHERE id=?",
                    values + (row_id,))
            
        return row_id


    def get_last_id(self, cur):
        
        # Select last seqeuence  number from database
        cur.execute('SELECT seq FROM sqlite_sequence WHERE name="systems"')
        
        # Get next row id
        result = cur.fetchone()
        if result is not None:
            row_id = result[0]
            return row_id
        else:
            return 0
    
    
    def _select(self, cmps, verbose=False):
        
        sql, args = self.create_select_statement(cmps)
        metadata = self._get_metadata()
        
        with self.managed_connection() as con:
            
            # Execute SQL request
            cur = con.cursor()
            cur.execute(sql, args)
            
            for row in cur.fetchall():
                
                yield self.convert_row(row, metadata, verbose=verbose)
                
    
    def create_select_statement(self, cmps, what='systems.*'):
        """
        Translate selection to SQL statement.
        """

        tables = ['systems']
        where = []
        args = []
        
        # Prepare SQL statement
        for key, op, value in cmps:
            where.append('systems.{}{}?'.format(key, op))
            args.append(value)
        
        # Create SQL statement
        sql = "SELECT {} FROM\n  ".format(what) + ", ".join(tables)
        if where:
            sql += "\n  WHERE\n  " + " AND\n  ".join(where)
            
        return sql, args


    def convert_row(self, row, metadata, verbose=False):
        
        # Convert reference properties to a dictionary
        properties = {}
        
        # Add database information
        if verbose:
            
            # Get row id
            properties["row_id"] = row[0]
            
            # Get modification date
            properties["mtime"] = row[1]
            
            # Get username
            properties["user"] = row[2]
        
        # Structural properties
        Np = 3
        for prop_i, dtype_i in structure_properties_dtype.items():
            
            if row[Np] is None:
                properties[prop_i] = None
            elif isinstance(row[Np], bytes):
                properties[prop_i] = torch.from_numpy(
                    self.deblob(
                        row[Np], dtype=dtype_i, 
                        shape=structure_properties_shape[prop_i]).copy())
            else:
                properties[prop_i] = torch.reshape(
                    torch.tensor(row[Np], dtype=dtype_i),
                    structure_properties_shape[prop_i])
            
            #if prop_i == "positions":
                #properties[prop_i] = properties[prop_i].reshape(-1, 3)
                
            Np += 1
            
            
        for prop_i in metadata.get('load_properties'):
            
            if prop_i not in structure_properties_dtype.keys():
                
                if row[Np] is None:
                    properties[prop_i] = None
                elif isinstance(row[Np], bytes):
                    properties[prop_i] = torch.from_numpy(
                        self.deblob(
                            row[Np], dtype=properties_numpy_dtype).copy()
                        ).to(properties_torch_dtype)
                else:
                    properties[prop_i] = torch.tensor(
                        row[Np], dtype=properties_torch_dtype)
                
                if prop_i == "forces":
                    properties[prop_i] = properties[prop_i].reshape(-1, 3)
            
                
                Np += 1
        
        return properties
    
    
    @parallel_function
    def _count(self, cmps):
        
        sql, args = self.create_select_statement(cmps, what='COUNT(*)')

        with self.managed_connection() as con:
            cur = con.cursor()
            try:
                cur.execute(sql, args)
                return cur.fetchone()[0]
            except sqlite3.OperationalError:
                return 0

    
    @parallel_function
    @data.lock
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
            
        self.vacuum()


    def _delete(self, row_ids):
        
        with self.managed_connection() as con:
            cur = con.cursor()
            selection = ', '.join([str(row_id) for row_id in row_ids])
            for table in all_tables[::-1]:
                cur.execute(
                    f"DELETE FROM {table} WHERE id in ({selection});")
    
    
    def vacuum(self):
        """
        Execute SQL command 'Vacuum' (?)
        """
        
        with self.managed_connection() as con:
            con.commit()
            con.cursor().execute("VACUUM")


    @property
    def metadata(self):
        return self._get_metadata()
        

    
