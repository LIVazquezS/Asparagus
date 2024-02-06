import os
import numpy as np
import logging
from typing import Optional, List, Dict, Tuple, Union, Any

import ase

from .. import data
from .. import settings
from .. import utils
from .. import interface
from .. import sample

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

__all__ = ['ReCalculator']

# ======================================
# Sample Data ReCalculator Class
# ======================================

class ReCalculator:
    """
    General Calculator class to recalculate sample structures.
    """
    
    def __init__(
        self,
        config: Optional[Union[str, dict, object]] = None,
        recalc_interface: Optional[str] = None,
        recalc_calculator: Optional[Union[str, object]] = None,
        recalc_calculator_args: Optional[Dict[str, Any]] = None,
        recalc_properties: Optional[List[str]] = None,
        recalc_source_data_file: Optional[Union[str, List[str]]] = None,
        recalc_target_data_file: Optional[Union[str, List[str]]] = None,
        recalc_directory: Optional[str] = None,
        **kwargs,
    ):
        """
        Initialize ReCalculator class.

        Parameters
        ----------

        config: (str, dict, object)
            Either the path to json file (str), dictionary (dict) or
            settings.config class object of model parameters
        recalc_interface: str, optional, default 'ase'
            Calculator interface for the recalculation. Available options are
            'ase', 'cc' or 'template'.
        recalc_calculator: (str, callable object), optional, default 'XTB'
            Definition of the calculator type. The input can be either directly
            a calculator class object of the particular interface definition or
            a string with available calculator classes.
        recalc_calculator_args: dict, optional, default {}
            Calculator option dictionary.
        recalc_properties: List[str], optional, default None
            List of system properties which are computed by the calculator 
            class. By default all available properties will be stored.
        recalc_source_data_file: (str, list), optional, default None
            Source database file name or list of file names which contains the reference samples and shall be recalculate.
        recalc_target_data_file: (str, list), optional, default None
            Target database file name to store the new sample property data.
            If the file name is not defined, source file name is used with
            a 'recalc_' prefix.
        recalc_directory: str, optional, default ''
            Working directory for working files.
        
        Returns
        -------
        callable object
            ReCalculator class object
        """

        ##########################################
        # # # Check ReCalculator Class Input # # #
        ##########################################
        
        # Get configuration object
        config = settings.get_config(config)

        # Check input parameter, set default values if necessary and
        # update the configuration dictionary
        config_update = {}
        for arg, item in locals().items():

            # Skip 'config' argument and possibly more
            if arg in [
                    'self', 'config', 'config_update', 'kwargs', '__class__']:
                continue

            # Take argument from global configuration dictionary if not defined
            # directly
            if item is None:
                item = config.get(arg)

            # Set default value if the argument is not defined (None)
            if arg in settings._default_args.keys() and item is None:
                item = settings._default_args[arg]

            # Check datatype of defined arguments
            if arg in settings._dtypes_args.keys():
                match = utils.check_input_dtype(
                    arg, item, settings._dtypes_args, raise_error=True)

            # Append to update dictionary
            config_update[arg] = item

            # Assign as class parameter
            setattr(self, arg, item)

        # Update global configuration dictionary
        config.update(config_update)
        
        # Check source and target file name
        self.recalc_source_data_file, self.recalc_target_data_file = (
            self.check_data_file_names())
        
        # Prepare calculator
        self.available_interface = ['ase', 'cc', 'template']
        self.get_calculator()
        
    
    def check_data_file_names(
        self, 
        recalc_source_data_file: Optional[Union[str, List[str]]] = None,
        recalc_target_data_file: Optional[Union[str, List[str]]] = None,
    ):
        """
        Check and prepare source and target data file names
        """
        
        # Prepare input
        if recalc_source_data_file is None:
            recalc_source_data_file = self.recalc_source_data_file
        if recalc_target_data_file is None:
            recalc_target_data_file = self.recalc_target_data_file
        
        if utils.is_string(recalc_source_data_file):
            recalc_source_data_file = [recalc_source_data_file]
        if recalc_target_data_file is None:
            recalc_target_data_file = []
            for data_file in recalc_source_data_file:
                head, tail = os.path.split(data_file)
                recalc_target_data_file.append(
                    os.path.join(head, "recalc_" + tail))
        elif utils.is_string(recalc_target_data_file):
            recalc_target_data_file = [recalc_target_data_file]
        
        # Check input
        if len(recalc_source_data_file) != len(recalc_target_data_file):
            raise ValueError(
                f"Number of source data files 'recalc_source_data_file' and " +
                f"target data files 'recalc_target_data_file' are " +
                f"different ({len(recalc_source_data_file):d} vs. " +
                f"{len(recalc_target_data_file):d})!")
        
        return recalc_source_data_file, recalc_target_data_file


    def get_calculator(
        self,
        recalc_interface: Optional[str] = None,
        recalc_calculator: Optional[Union[str, object]] = None,
        recalc_calculator_args: Optional[Dict[str, Any]] = None,
    ):

        # Prepare input
        if recalc_interface is None:
            recalc_interface = self.recalc_interface
        if recalc_calculator is None:
            recalc_calculator = self.recalc_calculator
        if recalc_calculator_args is None:
            recalc_calculator_args = self.recalc_calculator_args
            
        # Check calculator interface
        if recalc_interface not in self.available_interface:
            raise ValueError(
                f"Calculator interface 'recalc_interface' " +
                f"({recalc_interface:s}) is not available!\n" +
                f"The calculator interface must be from: " +
                f"{interface.available_interface}")
        
        # Check calculator
        if recalc_interface == 'ase':
            
            # Get ASE calculator
            recalc_calculator, recalc_calculator_tag = ( 
                interface.get_ase_calculator(
                    recalc_calculator, recalc_calculator_args)
                )
            self.recalc_calculator = recalc_calculator
            self.recalc_calculator_tag = recalc_calculator_tag
            
            # Check requested system properties
            for prop in self.recalc_properties:
                # TODO Special calculator properties list for special 
                # properties not supported by ASE such as, e.g., charge, 
                # hessian, etc.
                if prop not in recalc_calculator.implemented_properties:
                    raise ValueError(
                        f"Requested property '{prop:s}' is not implemented " +
                        f"in the ASE calculator '{recalc_calculator_tag}'! " +
                        f"Available ASE calculator properties are:\n" +
                        f"{recalc_calculator.implemented_properties}")
            
            # Define positions and property units
            self.recalc_unit_positions = 'Ang'
            self.recalc_unit_properties = {
                prop: interface.ase_calculator_units.get(prop)
                for prop in self.recalc_properties}
            
        elif recalc_interface == 'cc':
            
            raise NotImplementedError
        
        elif recalc_interface == 'template':
            
            raise NotImplementedError
        
        return


    def run(
        self,
        recalc_data_file_idx: Optional[Union[int, List[int]]] = None,
    ):
        """
        Run the recalculation of sample structures.
        
        Parameters
        ----------

        recalc_data_file_idx: (int, list(int)), optional, default None
            Index or list of indices for the source data file list to cover 
            in the recalculation run.
        """
        
        # Check source data file selection
        recalc_data_file_selection = self.check_data_file_idx(
            recalc_data_file_idx)
        
        # Iterate over source data files
        for ifile, data_file in enumerate(self.recalc_source_data_file):
            
            # Skip unselected source data files
            if not recalc_data_file_selection[ifile]:
                continue
            
            # Open source data file
            with data.connect(data_file) as db:
                
                print(db.get_metadata())
                print(db[1])
            
    def check_data_file_idx(self, data_file_idx):
        """
        Check data file selection input
        """
    
        if data_file_idx is None:
            
            # Select all
            data_file_selection = np.ones(
                len(self.recalc_source_data_file), dtype=bool)
            
        elif utils.is_integer(data_file_idx):
            
            # Check selection index and data file list length
            if (
                    data_file_idx >= len(self.recalc_source_data_file)
                    or data_file_idx < -len(self.recalc_source_data_file)
                ):
                raise ValueError(
                    f"Data file selection 'data_file_idx' " +
                    f"({data_file_idx}) " +
                    f"is out of range of defined source data file list " +
                    f"({len(self.recalc_source_data_file)})!")
            
            # Select data file of index
            data_file_selection = np.zeros(
                len(self.recalc_source_data_file), dtype=bool)
            data_file_selection[data_file_idx] = True
            
        elif utils.is_integer_array(data_file_idx):
            
            # Select data file of respective indices
            data_file_selection = np.zeros(
                len(self.recalc_source_data_file), dtype=bool)
            for idx in data_file_idx:
                
                # Check selection index and sample system length
                if (
                        idx >= len(self.recalc_source_data_file)
                        or idx < -len(self.recalc_source_data_file)
                    ):
                    raise ValueError(
                        f"Data file selection in 'data_file_idx' " +
                        f"({idx}) " +
                        f"is out of range of defined source data file list " +
                        f"({len(self.recalc_source_data_file)})!")
                
                # Select system
                data_file_selection[idx] = True
                
        else:
            
            raise ValueError(
                f"Data file selection in 'data_file_idx' " +
                f"has a wrong type '{type(data_file_idx)}'!\n"
                )
        
        return data_file_selection
