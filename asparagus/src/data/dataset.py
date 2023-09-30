import os
import logging
from typing import Optional, List, Dict, Tuple, Union, Any, Callable

#from ase import Atoms
#import ase.db as ase_db
#from ase.neighborlist import neighbor_list

import numpy as np

import torch

from .. import data
from .. import utils
from .. import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

__all__ = ['DataSet', 'DataSubSet', 'get_metadata']


def get_metadata(
    data_file: str,
) -> Dict[str, Any]:
    """
    Extract metadata from a data file.

    Parameters
    ----------
    data_file: str
        Reference Asparagus dataset file

    Returns
    -------
        dict,
            Asparagus dataset metadata
    """

    if os.path.exists(data_file):
        return DataSet(data_file).get_metadata()
    else:
        return {}


class DataSet():
    """
    DataSet class containing and loading reference data from files
    """

    def __init__(
        self,
        data_file: str,
        load_properties: Optional[List[str]] = None,
        unit_properties: Optional[Dict[str, str]] = None,
        data_overwrite: Optional[bool] = False,
        **kwargs
    ):
        """
        Parameters
        ----------
        data_file: str
            Reference Asparagus database file
        load_properties: List(str)optional, default None
            Subset of properties to load
        unit_properties: dict, optional, default None
            Dictionary from properties (keys) to corresponding unit as a
            string (item), e.g.:
                {property: unit}: { 'energy', 'eV',
                                    'force', 'eV/Ang', ...}
        data_overwrite: bool, optional, default False
            True: Overwrite 'data_file' (if exist) with 'data_source' data.
            False: Add data to 'data_file' (if exist) from 'data_source'.
        """

        # Assign arguments
        self.data_file = data_file

        # Remove data_file if overwrite is requested
        if os.path.exists(self.data_file) and data_overwrite:
            os.remove(self.data_file)

        # Check and assign metadata
        self.metadata = self.check_metadata(
            load_properties=load_properties,
            unit_properties=unit_properties,
            )

        # Initialize database file of the dataset
        self.initialize_database()

        self.load_properties = self.metadata['load_properties']
        self.unit_properties = self.metadata['unit_properties']

        # Initialize DataReader class parameter
        self.datareader = None

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
    ) -> Dict[str, torch.tensor]:

        with data.connect(self.data_file) as db:
            row = db.get(idx + 1)[0]

        return row

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
        load_properties: Optional[List[str]] = None,
        unit_properties: Optional[Dict[str, str]] = None,
        update_metadata: Optional[bool] = True,
    ) -> Dict[str, Any]:
        """
        Check (and update) metadata from the database file with current one

        Parameters
        ----------
        update_metadata: bool, optional, default True
            Update metadata in the database with a merged version of
            if the stored metadata and new one if no significant conflict
            arise between them.

        Returns
        -------
            dict
                Metadata of the database
        """

        # Check for custom data file path
        if data_file is None:
            data_file = self.data_file

        # Get metadata from dataset
        if os.path.exists(data_file):
            metadata = self.get_metadata(data_file)
        else:
            metadata = {}

        # Check metadata loaded properties
        if load_properties is None and metadata.get('load_properties') is None:
            raise SyntaxError(
                "Loading Properties is neither defined by the input nor "
                + "the data set.")
        elif load_properties is None:
            load_properties = metadata['load_properties']
        elif metadata.get('load_properties') is None:
            pass
        else:
            comp_properties = np.logical_not(
                np.array(
                    [
                        prop in metadata['load_properties']
                        for prop in load_properties
                    ], dtype=bool))
            if any(comp_properties):
                raise ValueError(
                    f"Existing database file '{data_file}' "
                    + "does not include properties "
                    + f"{np.array(load_properties)[comp_properties]}, "
                    + "which is/are requested by 'load_properties'!")

        # Check metadata loaded properties units
        if unit_properties is None and metadata.get('unit_properties') is None:
            raise SyntaxError(
                "Property units are neither defined by the input nor "
                + "the data set.")
        elif unit_properties is None:
            unit_properties = metadata['unit_properties']
        elif metadata.get('unit_properties') is not None:
            comp_units = []
            for key, item in unit_properties.items():
                if metadata['unit_properties'][key] != item:
                    comp_units.append(True)
                    logger.warning(
                        f"WARNING:\nDeviation in property unit for '{key}'!\n"
                        + " database file: "
                        + f"'{metadata['unit_properties'][key]}'\n"
                        + f" Current input: '{item}'")
                else:
                    comp_units.append(False)
            if any(comp_units):
                raise ValueError(
                    f"Property units in existing database file '{data_file}' "
                    + "deviates from current input of 'unit_properties'!")

        # Check positions unit
        if 'positions' not in unit_properties:
            if metadata.get('unit_properties') is None:
                unit_properties['positions'] = (
                    settings._default_args['data_unit_positions'])
            else:
                unit_properties['positions'] = metadata['unit_properties']

        # Update metadata with database metadata
        if update_metadata:
            metadata.update({
                'load_properties': load_properties,
                'unit_properties': unit_properties,
                })

        return metadata

    def initialize_database(
        self,
        data_file: Optional[str] = None,
        metadata: Optional[Dict] = None,
    ):
        """
        Initialize database file of the dataset
        """

        # Get database file path
        if data_file is None:
            data_file = self.data_file

        # Get metadata
        if metadata is None:
            metadata = self.metadata

        # Initialize database
        self.set_metadata(data_file, metadata)
        with data.connect(data_file) as db:
            db.init_systems()

    def load(
        self,
        data_source: str,
        data_format: Optional[str] = None,
        alt_property_labels: Optional[Dict[str, str]] = None,
        **kwargs
    ):
        """
        Load data from reference data file
        """

        # Detect data file extension if not given
        if data_format is None:
            data_format = data_source.split('.')[-1]

        # Check if data_source already loaded
        metadata = self.get_metadata()
        if metadata.get('data_source') is None:
            metadata['data_source'] = []
            metadata['data_format'] = []
        elif data_source in metadata['data_source']:
            logger.warning(
                f"WARNING:\nData source '{data_source}' already "
                + f"written to dataset '{self.data_file}'! "
                + "Loading data source is skipped.\n")
            return
        metadata['data_source'].append(data_source)
        metadata['data_format'].append(data_format)
        metadata['data_uptodate_property_scaling'] = False
        self.set_metadata(metadata=metadata)

        # Check and, in case, initialize DataReader
        if self.datareader is None:
            self.datareader = data.DataReader(
                data_file=self.data_file,
                alt_property_labels=alt_property_labels)

        # Load data file
        self.datareader.load(
            data_source,
            data_format,
            metadata['load_properties'],
            unit_properties=metadata['unit_properties'],
            alt_property_labels=alt_property_labels)

        return

    def add_atoms(
        self,
        atoms: object,
        properties: Dict[str, Any],
        alt_property_labels: Optional[Dict[str, str]] = None,
    ):
        """
        Add ASE Atoms system and properties
        """

        # Check and, in case, initialize DataReader
        if self.datareader is None:
            self.datareader = data.DataReader(
                data_file=self.data_file,
                alt_property_labels=alt_property_labels)

        # Get dataset metadata
        metadata = self.get_metadata()

        # Load from ASE atoms object and property list
        self.datareader.load_atoms(
            atoms,
            properties,
            load_properties=metadata['load_properties'],
            unit_properties=metadata['unit_properties'],
            alt_property_labels=alt_property_labels)

        return


class DataSubSet(DataSet):
    """
    DataSubSet class iterating and returning over a subset of DataSet.
    """

    def __init__(
        self,
        data_file: str,
        subset_idx: List[int],
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
