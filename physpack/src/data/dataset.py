import os
import logging
from typing import Optional, List, Dict, Tuple, Union, Any

import numpy as np 

from ase import Atoms
import ase.db as ase_db
#from ase.db import  connect
from ase.neighborlist import neighbor_list

import torch

from .. import data
from .. import utils
from .. import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

__all__ = ['DataSet', 'DataSubSet']

class DataSet():
    """
    DataSet class containing and loading reference data from files
    """
    
    def __init__(
        self,
        data_file: str,
        unit_positions: str,
        load_properties: List[str],
        unit_properties: Dict[str, str],
        data_overwrite: bool,
    ):
        """
        Parameters
        ----------
            
            data_file: str
                Reference ASE database file
            unit_positions: str
                Unit of the atom positions ('Ang' or 'Bohr') and other unit 
                cell information
            load_properties: List(str)
                Subset of properties to load
            unit_properties: dictionary
                Dictionary from properties (keys) to corresponding unit as a 
                string (item), e.g.:
                    {property: unit}: { 'energy', 'eV',
                                        'force', 'eV/Ang', ...}
                If 'load_properties is None, all properties defined in 
                'unit_properties' are loaded.
            data_overwrite: bool
                True: Overwrite 'data_file' (if exist) with 'data_source' data.
                False: Add data to 'data_file' (if exist) from 'data_source'.
            
        
        Returns
        -------
            object
                DataSet for providing data from reference DataBase
        """
        
        # Assign arguments
        self.data_file = data_file
        self.unit_positions = unit_positions
        self.load_properties = load_properties
        self.unit_properties = unit_properties
        self.data_overwrite = data_overwrite
        
        # Initialize data path format list
        self.data_sources = []
        self.data_format = []
        
        # Iterate over args
        for arg, item in locals().items():
            match = utils.check_input_dtype(
                arg, item, settings._dtypes_args, raise_error=True)
            
        # Remove data_file if overwrite is requested
        if os.path.exists(self.data_file) and self.data_overwrite:
            os.remove(self.data_file)
        
        # Collect metadata
        self.metadata = {
            'unit_positions': self.unit_positions,
            'load_properties': self.load_properties,
            'unit_properties': self.unit_properties,
            'data_sources': self.data_sources,
            'data_format': self.data_format,
            'data_property_scaling': {},
            'data_uptodate_property_scaling': False,
            }
        
        # Initialize or check database
        self.metadata = self.init_database(self.data_file, self.metadata)
        
        return
    
    
    def __len__(
        self,
    ) -> int:
        
        with data.connect(self.data_file) as db:
            return db.count()
    
    
    def __getitem__(
        self,
        idx: int,
    ) -> Dict[str, torch.tensor]:
        
        return self._get_properties(idx)
        
    
    def get_properties(
        self,
        idx: int,
    ) -> Dict[str, torch.tensor]:
        
        return self._get_properties(idx)


    def _get_properties(
        self,
        idx: int,
    ):
        
        with data.connect(self.data_file) as db:
            row = db.get(idx + 1)[0]
        
        return row
        
    
    def init_database(
        self,
        data_file: Optional[str] = None,
        metadata: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """
        Initialize database according to metadata or check with metadata
        in existing database.
        
        Returns
        -------
            dict
                Metadata of the database
        """
        
        # Assign optional parameters
        if data_file is None:
            data_file = self.data_file
            
        if metadata is None:
            metadata = self.metadata
        
        # Check or initialize database path
        if os.path.exists(data_file):
            metadata = self.check_metadata(data_file, metadata)
        else:
            with data.connect(data_file) as db:
                db.set_metadata(metadata)
                db.init_systems()
        
        return metadata
    
    
    def get_metadata(
        self,
        data_file: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Get metadata from database
        
        Returns
        -------
            dict
                Metadata of the database
        """
        
        # Assign optional parameters
        if data_file is None:
            data_file = self.data_file
        
        # Read metadata from database file
        with data.connect(data_file) as db:
            return db.get_metadata()
    
    
    def set_metadata(
        self,
        data_file: Optional[str] = None,
        metadata: Optional[Dict] = None,
    ):
        """
        Add metadata to the ASE database file
        """
        
        # Check for custom data file path
        if data_file is None:
            data_file = self.data_file
        
        # Check for custom metadata
        if metadata is None:
            metadata = self.metadata

        # Set metadata
        with data.connect(data_file) as db:
            db.set_metadata(metadata)


    def check_metadata(
        self,
        data_file: Optional[str] = None,
        metadata: Optional[Dict] = None,
        update_metadata: Optional[bool] = True,
        overwrite_metadata: Optional[bool] = False,
    ) -> Dict[str, Any]:
        """
        Check (and update) metadata from the database file with current one
        
        Parameters
        ----------
            
            update_metadata: bool, optional, default True
                Update metadata in the database with a merged version of 
                if the stored metadata and new one if no significant conflict 
                arise between them.
            overwrite_metadata: bool, optional, default False
                Overwrite metadata in the database with the new one without
                performing compatibility checks.
                
        Returns
        -------
            dict
                Metadata of the database
        """
        
        # Skip and set metadata if to be overwritten
        if overwrite_metadata:
            self.set_metadata(data_file, metadata)
            return
        
        # Check for custom data file path
        if data_file is None:
            data_file = self.data_file
        
        # Check for custom metadata
        if metadata is None:
            metadata = self.metadata
            
        # Read metadata from database file
        db_metadata = self.get_metadata(data_file)
        
        # Check metadata spatial unit
        if metadata['unit_positions'] != db_metadata['unit_positions']:
            raise ValueError(
                f"Existing database file '{data_file}'" + 
                f"has different 'unit_positions' entry "
                f"'{db_metadata['unit_positions']}' " +
                f"then current 'unit_positions' with " +
                f"{metadata['unit_positions']}!")

        # Check metadata loaded properties
        comp_properties = np.logical_not(np.array([
            prop in db_metadata['load_properties'] 
            for prop in metadata['load_properties']], dtype=bool))
        if any(comp_properties):
            raise ValueError(
                f"Existing database file '{data_file}' " + 
                f"does not include properties "
                f"{np.array(metadata['load_properties'])[comp_properties]}, "
                f"which are requested by 'load_properties'!")

        # Check metadata loaded properties units
        comp_units = []
        for key, item in metadata['unit_properties'].items():
            if db_metadata['unit_properties'][key] != item:
                comp_units.append(True)
                logger.warning(
                    f"WARNING:\nDeviation in property unit for '{key}'!\n" +
                    f" database file: " + 
                    f"'{db_metadata['unit_properties'][key]}'\n" +
                    f" Current input: '{item}'")
            else:
                comp_units.append(False)
        if any(comp_units):
            raise ValueError(
                f"Existing database file '{data_file}' " + 
                f"deviates from current input of 'unit_properties'!")
        
        # Update metadata with database metadata
        if update_metadata:
            metadata.update(db_metadata)
            self.set_metadata(data_file, metadata)

        return metadata
            
    def load(
        self, 
        data_source: str, 
        data_format: str,
        alt_property_labels: dict,
        **kwargs
    ):
        """
        Load data from reference data file
        """
        
        # Detect data file extension if not given
        if data_format is None:
            data_format = data_source.split('.')[-1]
        
        # Check if data_source already loaded
        with data.connect(self.data_file) as db:
            metadata = db.metadata
            if data_source in db.metadata['data_sources']:
                logger.warning(
                    f"WARNING:\nData source '{data_source}' already " + 
                    f"written to dataset '{self.data_file}'! " +
                    f"Loading data source is skipped.\n")
                return
            metadata['data_sources'].append(data_source)
            metadata['data_format'].append(data_format)
            metadata['data_uptodate_property_scaling'] = False
            db.set_metadata(metadata)
        
        # Load data file with case sensitive function by file format
        # 'db': Own data base format
        if data_format == 'db':
            self._load_db(data_source, alt_property_labels, **kwargs)
        # 'asedb': ASE data base format
        elif data_format == 'asedb':
            self._load_ase_db(data_source, alt_property_labels, **kwargs)
        # 'npz': Numpy npz file format
        elif data_format == 'npz':
            self._load_npz(data_source, alt_property_labels, **kwargs)
        else:
            raise TypeError(
                f"Data file format '{data_format}' is not supported!")
        
        return
    
    
    def _load_db(
        self,
        data_source: str,
        alt_property_labels: dict,
        **kwargs
    ):
        """
        Load data from own database files
        """
        
        # Get data sample to compare property labels
        with data.connect(data_source) as db:
            data_sample = db.get(1)
            Ndata = db.count()
        
        # Assign custom property labels to valid labels:
        assigned_properties = {}
        for custom_label in data_sample.keys():
            match, modified, valid_label = utils.check_property_label(
                custom_label, settings._valid_properties, alt_property_labels)
            if match:
                assigned_properties[valid_label] = custom_label
            elif modified:
                logger.warning(
                    f"WARNING:\nProperty key '{custom_label}' in " +
                    f"database '{data_source}' is not a valid label!\n" +
                    f"Property key '{custom_label}' is assigned as " +
                    f"'{valid_label}'.\n")
            else:
                logger.warning(
                    f"WARNING:\nUnknown property '{custom_label}' in ASE " + 
                    f"database '{data_source}'!\nProperty ignored.\n")
        
        # Check if all properties in 'load_properties' are found
        found_properties = [
            prop in assigned_properties.keys()
            for prop in self.load_properties]
        for ip, prop in enumerate(self.load_properties):
            if not found_properties[ip]:
                logger.error(
                    f"ERROR:\nRequested property '{prop}' in " + 
                    f"'load_properties' is not found in " + 
                    f"database '{data_source}'!\n")
        if not all(found_properties):
            raise ValueError(
                f"Not all properties in 'load_properties' are found " +
                f"in database '{data_source}'!\n")
        
        # Show assigned property information
        message = (
            f"INFO:\nAssignment of property labels in database " +
            f"'{data_source}'!\n" +
            f"Valid Label         Data File Label\n" +
            f"-----------------------------------\n")
        for key, item in assigned_properties.items():
            message += f"{key:<19} {item:<20}\n"
        logger.info(message)
        
        # Add atom systems to database
        atoms_properties = {}
        db_source = data.connect(data_source)
        with data.connect(self.data_file) as db:
            
            logger.info(
                f"INFO:\nWriting '{data_source}' to database " + 
                f"'{self.data_file}'!\n" +
                f"{Ndata} data point will be added.\n")
            
            for idx in range(Ndata):
                
                # Get atoms object and property data
                data = db_source.get(idx + 1)
                
                # Collect properties
                for ip, prop in enumerate(self.load_properties):
                    atoms_properties[prop] = data[assigned_properties[prop]]
                
                # Check for atom pair indices
                if not (
                    'idx_i' in atoms_properties.keys()
                    or 'idx_j' in atoms_properties.keys()
                    or 'pbc_offset' in atoms_properties.keys()):
                    
                    # Create atom object
                    atoms = Atoms(
                        numbers=atoms_properties['atomic_numbers'],
                        positions=atoms_properties['positions'],
                        cell=atoms_properties['cell'],
                        pbc=atoms_properties['pbc'])
                    
                    # Create atom pair indices
                    idx_i, idx_j, pbc_offset = neighbor_list(
                        'ijS',
                        atoms,
                        100.0,
                        self_interaction=False)
                    
                    atoms_properties['idx_i'] = idx_i
                    atoms_properties['idx_j'] = idx_j
                    atoms_properties['pbc_offset'] = pbc_offset
                    
                # Write to ASE database file
                db.write(properties=atoms_properties)
        
        # Print completion message
        logger.info(
            f"INFO:\nLoading from database '{data_source}' complete!\n")
        
        return
    
    
    def _load_ase_db(
        self,
        data_source: str,
        alt_property_labels: dict,
        **kwargs
    ):
        """
        Load data from ASE database files
        """
        
        # Get data sample to compare property labels
        with ase_db.connect(data_source) as db:
            data_sample = db.get(1).data
            Ndata = db.count()
        
        # Assign custom property labels to valid labels:
        assigned_properties = {}
        for custom_label in data_sample.keys():
            match, modified, valid_label = utils.check_property_label(
                custom_label, settings._valid_properties, alt_property_labels)
            if match:
                assigned_properties[valid_label] = custom_label
            elif modified:
                logger.warning(
                    f"WARNING:\nProperty key '{custom_label}' in ASE " +
                    f"database '{data_source}' is not a valid label!\n" +
                    f"Property key '{custom_label}' is assigned as " +
                    f"'{valid_label}'.\n")
            else:
                logger.warning(
                    f"WARNING:\nUnknown property '{custom_label}' in ASE " + 
                    f"database '{data_source}'!\nProperty ignored.\n")
        
        # Check if all properties in 'load_properties' are found
        found_properties = [
            prop in assigned_properties.keys()
            for prop in self.load_properties]
        for ip, prop in enumerate(self.load_properties):
            if not found_properties[ip]:
                logger.error(
                    f"ERROR:\nRequested property '{prop}' in " + 
                    f"'load_properties' is not found in ASE " + 
                    f"database '{data_source}'!\n")
        if not all(found_properties):
            raise ValueError(
                f"Not all properties in 'load_properties' are found " +
                f"in ASE database '{data_source}'!\n")
        
        # Show assigned property information
        message = (
            f"INFO:\nAssignment of property labels in ASE database " +
            f"'{data_source}'!\n" +
            f"Valid Label         Data File Label\n" +
            f"-----------------------------------\n")
        for key, item in assigned_properties.items():
            message += f"{key:<19} {item:<20}\n"
        logger.info(message)
        
        # Add atom systems to database
        atoms_properties = {}
        db_source = ase_db.connect(data_source)
        with connect(self.data_file) as db:
            
            logger.info(
                f"INFO:\nWriting '{data_source}' to database " + 
                f"{self.data_file}!\n" +
                f"{Ndata} data point will be added.\n")
            
            for idx in range(Ndata):
                
                # Get atoms object and property data
                atoms, data = db_source.get(idx + 1)
                
                # Collect properties
                for ip, prop in enumerate(self.load_properties):
                    atoms_properties[prop] = data[prop]
                
                # Check for atom pair indices
                if not (
                    'idx_i' in atoms_properties.keys()
                    or 'idx_j' in atoms_properties.keys()
                    or 'pbc_offset' in atoms_properties.keys()):
                    
                    # Create atom pair indices
                    idx_i, idx_j, pbc_offset = neighbor_list(
                        'ijS',
                        atoms,
                        100.0,
                        self_interaction=False)
                    
                    atoms_properties['idx_i'] = idx_i
                    atoms_properties['idx_j'] = idx_j
                    atoms_properties['pbc_offset'] = pbc_offset
                    
                # Write to ASE database file
                db.write(properties=atoms_properties)
        
        # Print completion message
        logger.info(
            f"INFO:\nLoading from ASE database {data_source} complete!\n")
        
        return
    
    
    def _load_npz(
        self,
        data_source: str,
        alt_property_labels: dict,
        **kwargs
    ):
        """
        Load data from Numpy 'npz' files to ASE database file
        """
        
        # Open npz file
        data_npz = np.load(data_source)
        
        # Assign custom property labels to valid labels:
        assigned_properties = {}
        for custom_label in data_npz.keys():
            match, modified, valid_label = utils.check_property_label(
                custom_label, settings._valid_properties, alt_property_labels)
            if match:
                assigned_properties[valid_label] = custom_label
            elif modified:
                logger.warning(
                    f"WARNING:\nProperty key '{custom_label}' in Numpy " +
                    f"dataset '{data_source}' is not a valid label!\n" +
                    f"Property key '{custom_label}' is assigned as " +
                    f"'{valid_label}'.\n")                    
            else:
                logger.warning(
                    f"WARNING:\nUnknown property '{custom_label}' in Numpy " + 
                    f"dataset '{data_source}'!\nProperty ignored.\n")
        
        # Atom numbers
        if 'atoms_number' in assigned_properties.keys():
            atoms_number = data_npz[assigned_properties['atoms_number']]
            Ndata = atoms_number.shape[0]
        else:
            raise ValueError(
                f"Property 'atoms_number' not found in Numpy dataset " + 
                f"'{self.data_file}'!\n")
        
        # Atomic number
        if 'atomic_numbers' in assigned_properties.keys():
            atomic_numbers = data_npz[assigned_properties['atomic_numbers']]
        else:
            raise ValueError(
                f"Property 'atomic_numbers' not found in Numpy dataset " + 
                f"'{self.data_file}'!\n")
        
        # Atom positions
        if 'positions' in assigned_properties.keys():
            positions = data_npz[assigned_properties['positions']]
        else:
            raise ValueError(
                f"Property 'positions' not found in Numpy dataset " + 
                f"'{self.data_file}'!\n")
        
        # Total atoms charge
        if 'charge' in assigned_properties.keys():
            charge = data_npz[assigned_properties['charge']]
        else:
            raise ValueError(
                f"Property 'charge' not found in Numpy dataset " + 
                f"'{self.data_file}'!\n")
        
        # Cell information
        if 'cell' in assigned_properties.keys():
            cell = data_npz[assigned_properties['cell']]
        else:
            cell = np.zeros((Ndata, 3), dtype=float)
            logger.info(
                f"INFO:\nNo cell information in Numpy dataset " +
                f"'{self.data_file}'!\n")
        
        # PBC information
        if 'pbc' in assigned_properties.keys():
            pbc = data_npz[assigned_properties['pbc']]
        else:
            pbc = np.zeros((Ndata, 3), dtype=bool)
            logger.info(
                f"INFO:\nNo pbc information in Numpy dataset " +
                f"'{self.data_file}'!\n")
        
        # Check if all properties in 'load_properties' are found
        found_properties = [
            prop in assigned_properties.keys()
            for prop in self.load_properties]
        for ip, prop in enumerate(self.load_properties):
            if not found_properties[ip]:
                logger.error(
                    f"ERROR:\nRequested property '{prop}' in " + 
                    f"'load_properties' is not found in Numpy " + 
                    f"dataset '{data_source}'!\n")
        if not all(found_properties):
            raise ValueError(
                f"Not all properties in 'load_properties' are found " +
                f"in Numpy dataset '{data_source}'!\n")
        
        # Check if properties are atom number dependent
        atomwise_properties = {}
        for prop in self.load_properties:
            if (
                len(data_npz[assigned_properties[prop]].shape) >= 2 and
                data_npz[assigned_properties[prop]].shape[0] == Ndata):
                    atomwise_properties[prop] = True
            else:
                    atomwise_properties[prop] = False
                    
        # Show assigned property information
        message = (
            f"INFO:\nAssignment of property labels in Numpy file " +
            f"'{data_source}'!\n" +
            f"Valid Label         Data File Label\n" +
            f"-----------------------------------\n")
        for key, item in assigned_properties.items():
            message += f"{key:<19} {item:<20}\n"
        logger.info(message)
        
        # Property information
        data_properties = {}
        for ip, prop in enumerate(self.load_properties):
            data_properties[prop] = data_npz[assigned_properties[prop]]
        
        # Add atom systems to ASE database
        logger.info(
            f"INFO:\nWriting '{data_source}' to database " + 
            f"{self.data_file}!\n" +
            f"{len(atoms_number)} data point will be added.\n")
        atoms_properties = {}
        with data.connect(self.data_file) as db:
            
            for isys, Nsys in enumerate(atoms_number):
                
                # Create atom object
                atoms = Atoms(
                    numbers=atomic_numbers[isys, :atoms_number[isys]],
                    positions=positions[isys, :atoms_number[isys]],
                    cell=cell[isys],
                    pbc=pbc[isys])
                
                # Create atom pair indices
                idx_i, idx_j, pbc_offset = neighbor_list(
                    'ijS',
                    atoms,
                    100.0,
                    self_interaction=False)
                
                # Collect atoms properties
                atoms_properties['atoms_number'] = (
                    atoms_number[isys])
                atoms_properties['atomic_numbers'] = (
                    atomic_numbers[isys, :atoms_number[isys]])
                atoms_properties['positions'] = (
                    positions[isys, :atoms_number[isys]])
                atoms_properties['charge'] = (
                    charge[isys])
                atoms_properties['cell'] = cell[isys]
                atoms_properties['pbc'] = pbc[isys]
                atoms_properties['idx_i'] = idx_i
                atoms_properties['idx_j'] = idx_j
                atoms_properties['pbc_offset'] = pbc_offset
                
                # Collect properties
                for ip, prop in enumerate(self.load_properties):
                    if atomwise_properties[prop]:
                        atoms_properties[prop] = (
                            data_properties[prop][isys][:atoms_number[isys]])
                    else:
                        atoms_properties[prop] = (
                            data_properties[prop][isys])
                
                # Write to ASE Database file
                row_id = db.write(properties=atoms_properties)
                        
        # Print completion message
        logger.info(
            f"INFO:\nLoading from Numpy file {data_source} complete!\n")
        
        return


class DataSubSet(DataSet):
    """
    DataSubSet class iterating and returning over a subset of DataSet.
    """
    
    def __init__(
        self, 
        data_file: str, 
        subset_idx: List[int],
        unit_positions: str,
        load_properties: List[str],
        unit_properties: Dict[str, str],
        data_overwrite: bool,
    ):
        """
        Parameters
        ----------
            
            data_file: str
                Reference ASE database file
            subset_idx: List(int)
                List of reference data indices of this subset.
        
        Returns
        -------
            object
                DataSubSet to present training, validation or testing
        """
        
        # Inherit from DataSet base class
        super().__init__(
            data_file, 
            unit_positions, 
            load_properties, 
            unit_properties,
            data_overwrite)
        
        # Assign arguments
        self.data_file = data_file
        self.subset_idx = [int(idx) for idx in subset_idx]
        
        # Iterate over args
        for arg, item in locals().items():
            match = utils.check_input_dtype(
                arg, item, settings._dtypes_args, raise_error=True)
        
        # Check database file
        if not os.path.exists(self.data_file):
            raise ValueError(
                f"File {self.data_file} does not exists!\n")
        
        # Number of subset data points
        self.Nidx = len(self.subset_idx)
        
        return
    
    def __len__(
        self,
    ) -> int:
        
        return self.Nidx
    
    
    def __getitem__(
        self,
        idx: int,
    ) -> Dict[str, torch.tensor]:
        
        return self._get_properties(self.subset_idx[idx])
    
    
    def get_properties(
        self,
        idx: int,
    ) -> Dict[str, torch.tensor]:
        
        # Load property data
        properties = self._get_properties(self.subset_idx[idx])
        
        return properties
