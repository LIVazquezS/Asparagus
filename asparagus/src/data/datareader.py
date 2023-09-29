import os
import logging
from typing import Optional, List, Dict, Tuple, Union, Any

import numpy as np 

import ase.db as ase_db

import torch

from .. import data
from .. import utils
from .. import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

__all__ = ['DataReader']

class DataReader():
    """
    DataReader class that contain functions to read and adapt data.
    """
    
    def __init__(
        self,
        data_file: Optional[str] = None,
        load_properties: Optional[List[str]] = None,
        alt_property_labels: Optional[Dict[str, List[str]]] = None,
    ):
        """
        Parameters
        ----------
        data_file: str, optional, default None
            if None, the data read by this class are returned as list.
            Else, the data are stored in the reference database file and just
            the data file path is returned.
        load_properties: List(str)
            Subset of properties to load to dataset file
        alt_property_labels: dictionary, optional, default None
            Dictionary of alternative property labeling to replace
            non-valid property labels with the valid one if possible.
        """
    
        # Assign reference data set path
        self.data_file = data_file
        
        # Assign property list
        self.load_properties = load_properties
        
        # Check alternative property labels
        if alt_property_labels is None:
            self.alt_property_labels = settings._alt_property_labels
        else:
            self.alt_property_labels = alt_property_labels

        # Default property labels
        self.default_property_labels = [
            'atoms_number', 'atomic_numbers', 'cell', 'pbc', 
            'idx_i', 'idx_j', 'pbc_offset']
        
        # Assign dataset format with respective load function
        self.data_format_load = {
            'db':       self.load_db,
            'asedb':    self.load_ase,
            'npz':      self.load_npz,
            'traj':     self.load_traj,
            }
        
        
        return

    def load(
        self,
        data_source: str,
        data_format: str,
        load_properties: Optional[List[str]] = None,
        unit_properties: Optional[Dict[str, str]] = None,
        alt_property_labels: Optional[Dict[str, List[str]]] = None,
        **kwargs,
    ):
        """
        Load data from respective dataset format.
        
        Parameters
        ----------
        data_source: str
            Path to data source file of asparagus type
        data_format: str
            Dataset format of 'data_source'
        load_properties: List(str)
            Subset of properties to load
        unit_properties: dict, optional, default None
            Dictionary from properties (keys) to corresponding unit as a 
            string (item), e.g.:
                {property: unit}: { 'positions': 'Ang',
                                    'energy': 'eV',
                                    'force': 'eV/Ang', ...}
        alt_property_labels: dictionary, optional, default None
            Dictionary of alternative property labeling to replace
            non-valid property labels with the valid one if possible.
        """
        
        # Check property list
        if load_properties is None:
            load_properties = self.load_properties
        
        # Check alternative property labels
        if alt_property_labels is None:
            alt_property_labels = self.alt_property_labels
        else:
            update_alt_property_labels = self.alt_property_labels
            update_alt_property_labels.update(alt_property_labels)
            alt_property_labels = update_alt_property_labels
        
        # Load data from source of respective format
        if self.data_format_load.get(data_format) is None:
            raise SyntaxError(
                f"Data format '{data_format:s}' of data source file " 
                + f"'{data_source:s}' is unknown!\n"
                + "Supported data formats are: "
                + f"{self.data_format_load.keys()}"
                )
        else:
            _ = self.data_format_load[data_format](
                data_source,
                load_properties,
                unit_properties=unit_properties,
                alt_property_labels=alt_property_labels,
                **kwargs)

        return
        

    def load_db(
        self, 
        data_source: str,
        load_properties: List[str],
        unit_properties: Optional[Dict[str, str]] = None,
        alt_property_labels: Optional[Dict[str, str]] = None,
        **kwargs,
    ):
        """
        Load data from asparagus dataset format.
        
        Parameters
        ----------
        data_source: str
            Path to data source file of asparagus type
        load_properties: List(str)
            Subset of properties to load
        unit_properties: dict, optional, default None
            Dictionary from properties (keys) to corresponding unit as a 
            string (item), e.g.:
                {property: unit}: { 'energy', 'eV',
                                    'force', 'eV/Ang', ...}
        alt_property_labels: dictionary, optional, default None
            Dictionary of alternative property labeling to replace
            non-valid property labels with the valid one if possible.
        """
        
        # Check if data source is empty
        with data.connect(data_source) as db:
            Ndata = db.count()
        if Ndata == 0:
            logger.warning(
                f"WARNING:\nData source '{data_source:s}' is empty!")
            return

        # Get data sample to compare property labels
        with data.connect(data_source) as db:
            data_sample = db.get(1)[0]

        # Check alternative property labels
        if alt_property_labels is None:
            alt_property_labels = self.alt_property_labels

        # Assign data source property labels to valid labels.
        assigned_properties = {}
        for custom_label in data_sample.keys():
            
            # Skip default system properties
            if custom_label in self.default_property_labels:
                continue
            
            match, modified, valid_label = utils.check_property_label(
                custom_label, settings._valid_properties, alt_property_labels)
            if match:
                assigned_properties[valid_label] = custom_label
            elif modified:
                logger.warning(
                    f"WARNING:\nProperty key '{custom_label:s}' in "
                    + f"database '{data_source:s}' is not a valid label!\n"
                    + f"Property key '{custom_label:s}' is assigned as "
                    + f"'{valid_label:s}'.\n")
            else:
                logger.warning(
                    f"WARNING:\nUnknown property '{custom_label:s}' in "
                    + f"database '{data_source:s}'!\nProperty ignored.\n")

        # Check if all properties in 'load_properties' are found
        found_properties = [
            prop in assigned_properties
            for prop in load_properties]
        for ip, prop in enumerate(load_properties):
            if not found_properties[ip]:
                logger.error(
                    f"ERROR:\nRequested property '{prop}' in " + 
                    f"'load_properties' is not found in " + 
                    f"database '{data_source}'!\n")
        if not all(found_properties):
            raise ValueError(
                f"Not all properties in 'load_properties' are found " +
                f"in database '{data_source}'!\n")

        # Get data source metadata
        with data.connect(data_source) as db:
            source_metadata = db.get_metadata()
        
        # Property match summary and unit conversion
        if unit_properties is None:
            
            # Set default conversion factor dictionary
            unit_conversion = {}
            for prop in assigned_properties.keys():
                unit_conversion[prop] = 1.0
            
            message = (
                f"INFO:\nProperties from database "
                + f"'{data_source}'!\n"
                + f" Property Label | Data Unit      | Source Label   |\n"
                + f"-"*17*3
                + f"\n")
            for prop, item in assigned_properties.items():
                if source_metadata['unit_properties'].get(prop) is None:
                    source_metadata['unit_properties'][prop] = None
                message += (
                    f"{prop:<16s} "
                    + f"{source_metadata['unit_properties'][prop]:<16s} "
                    + f"{item:<16s} ")

        else:
            
            # Check units of positions and properties
            unit_conversion = {}
            unit_mismatch = {}
            
            for prop in assigned_properties.keys():
                if source_metadata['unit_properties'].get(prop) is None:
                    source_metadata['unit_properties'][prop] = 'None'
                    unit_conversion[prop] = 1.0
                    unit_mismatch[prop] = False
                else:
                    unit_conversion[prop], unit_mismatch[prop] = (
                        utils.check_units(
                            unit_properties[prop], 
                            source_metadata['unit_properties'].get(prop))
                        )
        
            message = (
                f"INFO:\nProperty assignment from database "
                + f"'{data_source}'!\n"
                + f" Property Label | Data Unit      |"
                + f" Source Label   | Source Unit    | Conversion Fac.\n"
                + f"-"*17*5
                + f"\n")
            for prop, item in assigned_properties.items():
                message += (
                    f" {prop:<16s} {unit_properties[prop]:<16s} "
                    + f"{item:<16s} "
                    + f"{source_metadata['unit_properties'][prop]:<16s} "
                    + f"{unit_conversion[prop]:11.9e}\n")
        
        # Print Source information
        logger.info(message)
        
        # Open source dataset
        db_source = data.connect(data_source)
        
        # If not dataset file is given, load source data to memory
        if self.data_file is None:
            
            # Add atoms systems to list
            all_atoms_properties = []
            logger.info(
                f"INFO:\nLoad {Ndata} data point from '{data_source}'!\n")
            
            for idx in range(Ndata):
                
                # Atoms system data
                atoms_properties = {}
                
                # Get atoms object and property data
                source = db_source.get(idx + 1)[0]
                
                # Fundamental properties
                atoms_properties['atoms_number'] = source['atoms_number']
                atoms_properties['atomic_numbers'] = source['atomic_numbers']
                atoms_properties['positions'] = (
                    unit_conversion['positions']*source['positions'])
                atoms_properties['cell'] = (
                    unit_conversion['positions']*source['cell'])
                atoms_properties['pbc'] = source['pbc']
                if 'charge' not in source.keys():
                    atoms_properties['charge'] = 0.0
                else:
                    atoms_properties['charge'] = source['charge']
                
                # Collect properties
                for prop, item in assigned_properties.items():
                    if prop in load_properties:
                        atoms_properties[prop] = (
                            unit_conversion[prop]*source[item])

                # Add atoms system data
                all_atoms_properties.append(atoms_properties)
        
        # If dataset file is given, write to dataset
        else:
            
            # Add atom systems to database
            all_atoms_properties = self.data_file
            atoms_properties = {}
            with data.connect(self.data_file) as db:
            
                logger.info(
                    f"INFO:\nWriting '{data_source}' to database " + 
                    f"'{self.data_file}'!\n" +
                    f"{Ndata} data point will be added.\n")
                
                for idx in range(Ndata):
                    
                    # Get atoms object and property data
                    source = db_source.get(idx + 1)[0]
                    
                    # Fundamental properties
                    atoms_properties['atoms_number'] = source['atoms_number']
                    atoms_properties['atomic_numbers'] = (
                        source['atomic_numbers'])
                    atoms_properties['positions'] = (
                        unit_conversion['positions']*source['positions'])
                    atoms_properties['cell'] = (
                        unit_conversion['positions']*source['cell'])
                    atoms_properties['pbc'] = source['pbc']
                    if 'charge' not in source.keys():
                        atoms_properties['charge'] = 0.0
                    else:
                        atoms_properties['charge'] = source['charge']
                    
                    # Collect properties
                    for prop, item in assigned_properties.items():
                        if prop in load_properties:
                            atoms_properties[prop] = (
                                unit_conversion[prop]*source[item])

                    ## Check for atom pair indices
                    #if not 'idx_i' in atoms_properties.keys():
                        
                        ## Create atom object
                        #atoms = Atoms(
                            #numbers=source['atomic_numbers'],
                            #positions=source['positions'],
                            #cell=source['cell'][0],
                            #pbc=source['pbc'])
                        
                        ## Create atom pair indices
                        #idx_i, idx_j, pbc_offset = neighbor_list(
                            #'ijS',
                            #atoms,
                            #100.0,
                            #self_interaction=False)
                        
                        #atoms_properties['idx_i'] = idx_i
                        #atoms_properties['idx_j'] = idx_j
                        #atoms_properties['pbc_offset'] = pbc_offset
                        
                    # Write to ASE database file
                    db.write(properties=atoms_properties)

        # Print completion message
        logger.info(
            f"INFO:\nLoading from Asparagus dataset '{data_source}' " 
            + "complete!\n")

        return all_atoms_properties
        
    def load_ase(
        self, 
        data_source: str,
        load_properties: List[str],
        unit_properties: Optional[Dict[str, str]] = None,
        alt_property_labels: Optional[Dict[str, str]] = None,
        **kwargs,
    ):
        """
        Load data from asparagus dataset format.
        
        Parameters
        ----------
        data_source: str
            Path to data source file of asparagus type
        load_properties: List(str)
            Subset of properties to load
        unit_properties: dict, optional, default None
            Dictionary from properties (keys) to corresponding unit as a 
            string (item), e.g.:
                {property: unit}: { 'energy', 'eV',
                                    'force', 'eV/Ang', ...}
        alt_property_labels: dictionary, optional, default None
            Dictionary of alternative property labeling to replace
            non-valid property labels with the valid one if possible.
        """
        
        # Check if data source is empty
        with ase_db.connect(data_source) as db:
            Ndata = db.count()
        if Ndata == 0:
            logger.warning(
                f"WARNING:\nData source '{data_source:s}' is empty!")
            return

        # Get data sample to compare property labels
        with ase_db.connect(data_source) as db:
            data_sample = db.get(1)
            
        # Check alternative property labels
        if alt_property_labels is None:
            alt_property_labels = self.alt_property_labels

        # Assign data source property labels to valid labels.
        assigned_properties = {}
        for custom_label in data_sample:
            
            # Skip default system properties
            if custom_label in self.default_property_labels:
                continue
            
            match, modified, valid_label = utils.check_property_label(
                custom_label, settings._valid_properties, alt_property_labels)
            if match:
                assigned_properties[valid_label] = custom_label
            elif modified:
                logger.warning(
                    f"WARNING:\nProperty key '{custom_label:s}' in "
                    + f"database '{data_source:s}' is not a valid label!\n"
                    + f"Property key '{custom_label:s}' is assigned as "
                    + f"'{valid_label:s}'.\n")
            else:
                logger.warning(
                    f"WARNING:\nUnknown property '{custom_label:s}' in "
                    + f"database '{data_source:s}'!\nProperty ignored.\n")

        # Check if all properties in 'load_properties' are found
        found_properties = [
            prop in assigned_properties
            for prop in load_properties]
        for ip, prop in enumerate(load_properties):
            if not found_properties[ip]:
                logger.error(
                    f"ERROR:\nRequested property '{prop}' in " + 
                    f"'load_properties' is not found in " + 
                    f"database '{data_source}'!\n")
        if not all(found_properties):
            raise ValueError(
                f"Not all properties in 'load_properties' are found " +
                f"in database '{data_source}'!\n")
        
        if unit_properties is None:
            
            # Set default conversion factor dictionary
            unit_conversion = {}
            for prop in assigned_properties.keys():
                unit_conversion[prop] = 1.0
            
            message = (
                f"INFO:\nProperties from database "
                + f"'{data_source}'!\n"
                + f" Property Label | Data Unit      | Source Label   |\n"
                + f"-"*17*3
                + f"\n")
            for prop, item in assigned_properties.items():
                message += (
                    f"{prop:<16s} "
                    + f"{'ASE unit':<16s} "
                    + f"{item:<16s} ")

        else:
            
            # Check units of positions and properties
            unit_conversion = {}
            unit_mismatch = {}
            
            for prop in assigned_properties.keys():
                unit_conversion[prop], unit_mismatch[prop] = (
                    utils.check_units(
                        unit_properties[prop], 
                        None)
                    )

            message = (
                f"INFO:\nProperty assignment from database "
                + f"'{data_source}'!\n"
                + f" Property Label | Data Unit      |"
                + f" Source Label   | Source Unit    | Conversion Fac.\n"
                + f"-"*17*5
                + f"\n")
            for prop, item in assigned_properties.items():
                message += (
                    f" {prop:<16s} {unit_properties[prop]:<16s} "
                    + f"{item:<16s} "
                    + f"{'ASE unit':<16s} "
                    + f"{unit_conversion[prop]:11.9e}\n")

        # Print Source information
        logger.info(message)

        # Open source dataset
        db_source = ase_db.connect(data_source)

        # If not dataset file is given, load source data to memory
        if self.data_file is None:

            # Add atoms systems to list
            all_atoms_properties = []
            logger.info(
                f"INFO:\nLoad {Ndata} data point from '{data_source}'!\n")
            
            for idx in range(Ndata):
                
                # Atoms system data
                atoms_properties = {}
                
                # Get atoms object and property data
                atoms = db_source.get_atoms(idx + 1)
                source = db_source.get(idx + 1)
                
                # Fundamental properties
                atoms_properties['atoms_number'] = (
                    atoms.get_global_number_of_atoms())
                atoms_properties['atomic_numbers'] = (
                    atoms.get_atomic_numbers())
                atoms_properties['positions'] = (
                    unit_conversion['positions']*atoms.get_positions())
                atoms_properties['cell'] = (
                    unit_conversion['positions']
                    *np.array(list(atoms.get_cell())))[0]
                atoms_properties['pbc'] = atoms.get_pbc()
                if 'charge' in source:
                    atoms_properties['charge'] = source['charge']
                elif 'charges' in source:
                    atoms_properties['charge'] = sum(source['charges'])
                elif 'initial_charges' in source:
                    atoms_properties['charge'] = sum(source['initial_charges'])
                else:
                    atoms_properties['charge'] = 0.0
                
                # Collect properties
                for prop, item in assigned_properties.items():
                    if prop in load_properties:
                        atoms_properties[prop] = (
                            unit_conversion[prop]*source[item])

                ## Check for atom pair indices
                #if not 'idx_i' in atoms_properties.keys():
                    
                    ## Create atom pair indices
                    #idx_i, idx_j, pbc_offset = neighbor_list(
                        #'ijS',
                        #atoms,
                        #100.0,
                        #self_interaction=False)
                    
                    #atoms_properties['idx_i'] = idx_i
                    #atoms_properties['idx_j'] = idx_j
                    #atoms_properties['pbc_offset'] = pbc_offset

                # Add atoms system data
                all_atoms_properties.append(atoms_properties)

        # If dataset file is given, write to dataset
        else:
            
            # Add atom systems to database
            all_atoms_properties = self.data_file
            atoms_properties = {}
            with data.connect(self.data_file) as db:
            
                logger.info(
                    f"INFO:\nWriting '{data_source}' to database " + 
                    f"'{self.data_file}'!\n" +
                    f"{Ndata} data point will be added.\n")
                
                for idx in range(Ndata):
                    
                    # Get atoms object and property data
                    atoms = db_source.get_atoms(idx + 1)
                    source = db_source.get(idx + 1)
                    
                    # Fundamental properties
                    atoms_properties['atoms_number'] = (
                        atoms.get_global_number_of_atoms())
                    atoms_properties['atomic_numbers'] = (
                        atoms.get_atomic_numbers())
                    atoms_properties['positions'] = (
                        unit_conversion['positions']*atoms.get_positions())
                    print(atoms.get_positions())
                    print(atoms_properties['positions'])
                    print()
                    atoms_properties['cell'] = (
                        unit_conversion['positions']
                        *np.array(atoms.get_cell()))[0]
                    atoms_properties['pbc'] = atoms.get_pbc()
                    if 'charge' in source:
                        atoms_properties['charge'] = source['charge']
                    elif 'charges' in source:
                        atoms_properties['charge'] = sum(source['charges'])
                    elif 'initial_charges' in source:
                        atoms_properties['charge'] = sum(
                            source['initial_charges'])
                    else:
                        atoms_properties['charge'] = 0.0
                    
                    # Collect properties
                    for prop, item in assigned_properties.items():
                        if prop in load_properties:
                            atoms_properties[prop] = (
                                unit_conversion[prop]*source[item])

                    ## Check for atom pair indices
                    #if not 'idx_i' in atoms_properties.keys():
                        
                        ## Create atom pair indices
                        #idx_i, idx_j, pbc_offset = neighbor_list(
                            #'ijS',
                            #atoms,
                            #100.0,
                            #self_interaction=False)
                        
                        #atoms_properties['idx_i'] = idx_i
                        #atoms_properties['idx_j'] = idx_j
                        #atoms_properties['pbc_offset'] = pbc_offset
                        
                    # Write to ASE database file
                    db.write(properties=atoms_properties)

        # Print completion message
        logger.info(
            f"INFO:\nLoading from ASE database '{data_source}' complete!\n")

        return all_atoms_properties

    def load_npz(
        self, 
        data_source: str,
        load_properties: List[str],
        unit_properties: Optional[Dict[str, str]] = None,
        source_unit_properties: Optional[Dict[str, str]] = None,
        alt_property_labels: Optional[Dict[str, str]] = None,
        **kwargs,
    ):
        """
        Load data from npz dataset format.
        
        Parameters
        ----------
        data_source: str
            Path to data source file of asparagus type
        load_properties: List(str)
            Subset of properties to load
        unit_properties: dict, optional, default None
            Dictionary from properties (keys) to corresponding unit as a 
            string (item), e.g.:
                {property: unit}: { 'energy', 'eV',
                                    'force', 'eV/Ang', ...}
        source_unit_properties: dict, optional, default None
            As 'unit_properties' but four source npz data.
        alt_property_labels: dictionary, optional, default None
            Dictionary of alternative property labeling to replace
            non-valid property labels with the valid one if possible.
        """
        
        # Open npz file
        source = np.load(data_source)
        
        # Check if data source is empty
        Nprop = len(source.keys())
        if Nprop == 0:
            logger.warning(
                f"WARNING:\nData source '{data_source:s}' is empty!")
            return

        # Check alternative property labels
        if alt_property_labels is None:
            alt_property_labels = self.alt_property_labels

        # Assign data source property labels to valid labels.
        assigned_properties = {}
        for custom_label in source.keys():
            
            match, modified, valid_label = utils.check_property_label(
                custom_label, settings._valid_properties, alt_property_labels)
            if match:
                assigned_properties[valid_label] = custom_label
            elif modified:
                logger.warning(
                    f"WARNING:\nProperty key '{custom_label:s}' in "
                    + f"database '{data_source:s}' is not a valid label!\n"
                    + f"Property key '{custom_label:s}' is assigned as "
                    + f"'{valid_label:s}'.\n")
            else:
                logger.warning(
                    f"WARNING:\nUnknown property '{custom_label:s}' in "
                    + f"database '{data_source:s}'!\nProperty ignored.\n")
        
        # Atom numbers
        if 'atoms_number' in assigned_properties.keys():
            atoms_number = source[assigned_properties['atoms_number']]
            Ndata = len(atoms_number)
        else:
            raise ValueError(
                f"Property 'atoms_number' not found in npz dataset "
                + f"'{self.data_file}'!\n")
        
        # Atomic number
        if 'atomic_numbers' in assigned_properties.keys():
            atomic_numbers = source[assigned_properties['atomic_numbers']]
        else:
            raise ValueError(
                f"Property 'atomic_numbers' not found in npz dataset "
                + f"'{self.data_file}'!\n")
        
        # Atom positions
        if 'positions' in assigned_properties.keys():
            positions = source[assigned_properties['positions']]
        else:
            raise ValueError(
                f"Property 'positions' not found in npz dataset "
                + f"'{self.data_file}'!\n")
        
        # Total atoms charge
        if 'charge' in assigned_properties.keys():
            charge = source[assigned_properties['charge']]
        else:
            charge = np.zeros(Ndata, dtype=float)
            logger.warning(
                f"WARNING:\nProperty 'charge' not found in npz dataset "
                + f"'{self.data_file}'!\nCharges are assumed to be zero.")
        
        # Cell information
        if 'cell' in assigned_properties.keys():
            cell = source[assigned_properties['cell']]
        else:
            cell = np.zeros((Ndata, 3), dtype=float)
            logger.info(
                f"INFO:\nNo cell information in npz dataset "
                + f"'{self.data_file}'!\n")
        
        # PBC information
        if 'pbc' in assigned_properties.keys():
            pbc = source[assigned_properties['pbc']]
        else:
            pbc = np.zeros((Ndata, 3), dtype=bool)
            logger.info(
                f"INFO:\nNo pbc information in npz dataset "
                + f"'{self.data_file}'!\n")

        # Check if all properties in 'load_properties' are found
        found_properties = [
            prop in assigned_properties.keys()
            for prop in load_properties]
        for ip, prop in enumerate(load_properties):
            if not found_properties[ip]:
                logger.error(
                    f"ERROR:\nRequested property '{prop}' in "
                    + f"'load_properties' is not found in Numpy "
                    + f"dataset '{data_source}'!\n")
        if not all(found_properties):
            raise ValueError(
                f"Not all properties in 'load_properties' are found "
                + f"in Numpy dataset '{data_source}'!\n")
        
        # Property match summary and unit conversion
        if unit_properties is None and source_unit_properties is None:
            
            # Set default conversion factor dictionary
            unit_conversion = {}
            for prop in assigned_properties.keys():
                unit_conversion[prop] = 1.0
            
            message = (
                f"INFO:\nProperties from database "
                + f"'{data_source}'!\n"
                + f" Property Label | Data Unit      | Source Label   |\n"
                + f"-"*17*3
                + f"\n")
            for prop, item in assigned_properties.items():
                message += (
                    f"{prop:<16s} "
                    + f"{'None':<16s} "
                    + f"{item:<16s}\n")
            
        elif unit_properties is None: 
            
            # Set default conversion factor dictionary
            unit_conversion = {}
            for prop in assigned_properties.keys():
                unit_conversion[prop] = 1.0
            
            message = (
                f"INFO:\nProperties from database "
                + f"'{data_source}'!\n"
                + f" Property Label | Data Unit      | Source Label   |\n"
                + f"-"*17*3
                + f"\n")
            for prop, item in assigned_properties.items():
                if source_unit_properties.get(item) is None:
                    source_unit_properties[prop] = 'None'
                elif source_unit_properties.get(prop) is None:
                    source_unit_properties[prop] = 'None'
                message += (
                    f"{prop:<16s} "
                    + f"{source_unit_properties[prop]:<16s} "
                    + f"{item:<16s}\n")

        elif source_unit_properties is None: 
            
            # Set default conversion factor dictionary
            unit_conversion = {}
            for prop in assigned_properties.keys():
                unit_conversion[prop] = 1.0
            
            message = (
                f"INFO:\nProperties from database "
                + f"'{data_source}'!\n"
                + f" Property Label | Data Unit      | Source Label   |\n"
                + f"-"*17*3
                + f"\n")
            for prop, item in assigned_properties.items():
                if unit_properties.get(prop) is None:
                    unit_properties[prop] = 'None'
                message += (
                    f"{prop:<16s} "
                    + f"{unit_properties[prop]:<16s} "
                    + f"{item:<16s}\n")

        else:
            
            # Check units of positions and properties
            unit_conversion = {}
            unit_mismatch = {}
            
            for prop in assigned_properties.keys():
                if source_unit_properties.get(item) is None:
                    source_unit_properties[prop] = 'None'
                    unit_conversion[prop] = 1.0
                    unit_mismatch[prop] = False
                elif source_unit_properties.get(prop) is None:
                    source_unit_properties[prop] = 'None'
                    unit_conversion[prop] = 1.0
                    unit_mismatch[prop] = False
                    
                else:
                    unit_conversion[prop], unit_mismatch[prop] = (
                        utils.check_units(
                            unit_properties[prop], 
                            source_unit_properties.get(prop))
                        )
        
            message = (
                f"INFO:\nProperty assignment from database "
                + f"'{data_source}'!\n"
                + f" Property Label | Data Unit      |"
                + f" Source Label   | Source Unit    | Conversion Fac.\n"
                + f"-"*17*5
                + f"\n")
            for prop, item in assigned_properties.items():
                message += (
                    f" {prop:<16s} {unit_properties[prop]:<16s} "
                    + f"{item:<16s} "
                    + f"{source_unit_properties[prop]:<16s} "
                    + f"{unit_conversion[prop]:11.9e}\n")
            
        # Print Source information
        logger.info(message)
        
        # If not dataset file is given, load source data to memory
        if self.data_file is None:
            
            # Add atoms systems to list
            all_atoms_properties = []
            logger.info(
                f"INFO:\nLoad {Ndata} data point from '{data_source}'!\n")
            
            for idx in range(Ndata):
                
                # Atoms system data
                atoms_properties = {}
                
                # Fundamental properties
                atoms_properties['atoms_number'] = atoms_number[idx]
                atoms_properties['atomic_numbers'] = atomic_numbers[idx]
                atoms_properties['positions'] = (
                    unit_conversion['positions']*positions[idx])
                atoms_properties['cell'] = (
                    unit_conversion['positions']*cell[idx])
                atoms_properties['pbc'] = pbc[idx]
                atoms_properties['charge'] = charge[idx]
                
                # Collect properties
                for prop, item in assigned_properties.items():
                    if prop in self.default_property_labels:
                        continue
                    if prop in load_properties:
                        try:
                            atoms_properties[prop] = (
                                unit_conversion[prop]*source[item][idx])
                        except ValueError as err:
                            raise ValueError(
                                f"Property '{prop:s}' from the npz entry "
                                + f"'{item:s}' could not be loaded!")

                # Add atoms system data
                all_atoms_properties.append(atoms_properties)
        
        # If dataset file is given, write to dataset
        else:
            
            # Add atom systems to database
            all_atoms_properties = self.data_file
            atoms_properties = {}
            with data.connect(self.data_file) as db:
            
                logger.info(
                    f"INFO:\nWriting '{data_source}' to database " + 
                    f"'{self.data_file}'!\n" +
                    f"{Ndata} data point will be added.\n")
                
                for idx in range(Ndata):
                    
                    # Fundamental properties
                    atoms_properties['atoms_number'] = atoms_number[idx]
                    atoms_properties['atomic_numbers'] = atomic_numbers[idx]
                    atoms_properties['positions'] = (
                        unit_conversion['positions']*positions[idx])
                    atoms_properties['cell'] = (
                        unit_conversion['positions']*cell[idx])
                    atoms_properties['pbc'] = pbc[idx]
                    atoms_properties['charge'] = charge[idx]
                    
                    # Collect properties
                    for prop, item in assigned_properties.items():
                        if prop in self.default_property_labels:
                            continue
                        if prop in load_properties:
                            try:
                                atoms_properties[prop] = (
                                    unit_conversion[prop]*source[item][idx])
                            except ValueError as err:
                                raise ValueError(
                                    f"Property '{prop:s}' from the npz entry "
                                    + f"'{item:s}' could not be loaded!")

                    ## Check for atom pair indices
                    #if not 'idx_i' in atoms_properties.keys():
                        
                        ## Create atom object
                        #atoms = Atoms(
                            #numbers=source['atomic_numbers'],
                            #positions=source['positions'],
                            #cell=source['cell'][0],
                            #pbc=source['pbc'])
                        
                        ## Create atom pair indices
                        #idx_i, idx_j, pbc_offset = neighbor_list(
                            #'ijS',
                            #atoms,
                            #100.0,
                            #self_interaction=False)
                        
                        #atoms_properties['idx_i'] = idx_i
                        #atoms_properties['idx_j'] = idx_j
                        #atoms_properties['pbc_offset'] = pbc_offset
                        
                    # Write to ASE database file
                    db.write(properties=atoms_properties)
        
        # Print completion message
        logger.info(
            f"INFO:\nLoading from npz database '{data_source}' complete!\n")

        return all_atoms_properties

    def load_traj(
        self,
    ):
        pass

    def load_atoms(
        self,
        atoms: object, 
        properties: Dict[str, Any],
        load_properties: Optional[List[str]] = None,
        unit_properties: Optional[Dict[str, str]] = None,
        alt_property_labels: Optional[Dict[str, str]] = None,
    ):
        """
        Load atoms object with properties to dataset format.
        
        Parameters
        ----------
        atoms: ASE Atoms object
            ASE Atoms object with conformation belonging to the properties.
        properties: dict
            Atoms object properties
        load_properties: List(str)
            Subset of properties to load to dataset file
        unit_properties: dict, optional, default None
            Dictionary from properties (keys) to corresponding unit as a 
            string (item), e.g.:
                {property: unit}: { 'energy', 'eV',
                                    'force', 'eV/Ang', ...}
        alt_property_labels: dictionary, optional, default None
            Dictionary of alternative property labeling to replace
            non-valid property labels with the valid one if possible.
        """

        # Check property list
        if load_properties is None:
            load_properties = self.load_properties
            
        # Check property unit conversion
        if unit_properties is None:
            
            # Set default conversion factor dictionary
            unit_conversion = {}
            for prop in assigned_properties.keys():
                unit_conversion[prop] = 1.0
        
        else:
            
            # Check units of positions and properties
            unit_conversion = {}
            unit_mismatch = {}
            for prop in unit_properties.keys():
                
                unit_conversion[prop], unit_mismatch[prop] = (
                    utils.check_units(
                        unit_properties[prop], 
                        None)
                    )
        
        # If not dataset file is given, load data to memory
        if self.data_file is None:
                
            # Atoms system data
            atoms_properties = {}
            
            # Fundamental properties
            atoms_properties['atoms_number'] = (
                atoms.get_global_number_of_atoms())
            atoms_properties['atomic_numbers'] = (
                atoms.get_atomic_numbers())
            atoms_properties['positions'] = (
                unit_conversion['positions']*atoms.get_positions())
            atoms_properties['cell'] = (
                unit_conversion['positions']
                *np.array(list(atoms.get_cell())))[0]
            atoms_properties['pbc'] = atoms.get_pbc()
            if properties.get('charge') is None:
                atoms_properties['charge'] = 0.0
            else:
                atoms_properties['charge'] = properties['charge']
            
            # Collect properties
            for prop, item in properties.items():
                atoms_properties[prop] = (
                    unit_conversion[prop]*item)

        # If dataset file is given, write to dataset
        else:
            
            # Atoms system data
            atoms_properties = {}
            
            with data.connect(self.data_file) as db:
                
                # Fundamental properties
                atoms_properties['atoms_number'] = (
                    atoms.get_global_number_of_atoms())
                atoms_properties['atomic_numbers'] = (
                    atoms.get_atomic_numbers())
                atoms_properties['positions'] = (
                    unit_conversion['positions']*atoms.get_positions())
                atoms_properties['cell'] = (
                    unit_conversion['positions']
                    *np.array(list(atoms.get_cell())))[0]
                atoms_properties['pbc'] = atoms.get_pbc()
                if properties.get('charge') is None:
                    atoms_properties['charge'] = 0.0
                else:
                    atoms_properties['charge'] = properties['charge']
                
                # Collect properties
                for prop, item in properties.items():
                    if prop in load_properties:
                        atoms_properties[prop] = (
                            unit_conversion[prop]*item)

                ## Check for atom pair indices
                #if not 'idx_i' in atoms_properties.keys():
                    
                    ## Create atom pair indices
                    #idx_i, idx_j, pbc_offset = neighbor_list(
                        #'ijS',
                        #atoms,
                        #100.0,
                        #self_interaction=False)
                    
                    #atoms_properties['idx_i'] = idx_i
                    #atoms_properties['idx_j'] = idx_j
                    #atoms_properties['pbc_offset'] = pbc_offset
                        
                # Write to ASE database file
                db.write(properties=atoms_properties)

        return atoms_properties



