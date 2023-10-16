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
            Subset of properties to load.
        unit_properties: dict, optional, default None
            Dictionary from properties (keys) to corresponding unit as a
            string (item), e.g.:
                {property: unit}: { 'positions': 'Ang',
                                    'energy': 'eV',
                                    'force': 'eV/Ang', ...}
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
            **kwargs
            )

        # Initialize database file of the dataset
        self.initialize_database()

        # Update loaded properties and units
        self.load_properties = self.metadata['load_properties']
        self.unit_properties = self.metadata['unit_properties']

        # Initialize DataReader class parameter
        self.datareader = None

        return

    def __len__(
        self,
    ) -> int:

        if os.path.isfile(self.data_file):
            with data.connect(self.data_file) as db:
                return db.count()
        else:
            return 0

    def __getitem__(
        self,
        idx: int,
    ) -> Dict[str, torch.tensor]:

        return self._get_properties(idx)

    def __setitem__(
        self,
        idx: int,
        properties: Dict[str, torch.tensor],
    ):

        return self._set_properties([idx], [properties])
    
    def get(
        self,
        idx: int,
    ) -> Dict[str, torch.tensor]:

        return self._get_properties(idx)

    def get_properties(
        self,
        idx: int,
    ) -> Dict[str, torch.tensor]:

        return self._get_properties(idx)

    def set_properties(
        self,
        idx: Union[int, List[int]],
        properties: Union[Dict[str, torch.tensor], List[Dict]],
    ):

        if utils.is_integer(idx):
            idx = [idx]
        if utils.is_dictionary(properties):
            properties = [properties]

        return self._set_properties(idx, properties)

    def _get_properties(
        self,
        idx: int,
    ) -> Dict[str, torch.tensor]:

        with data.connect(self.data_file, mode='r') as db:
            row = db.get(idx + 1)[0]

        return row

    def _set_properties(
        self,
        idcs: List[int],
        properties: List[Dict[str, torch.tensor]],
    ):

        with data.connect(self.data_file, mode='a') as db:
            for idx, props in zip(idcs, properties):
                row_id = db.write(props, row_id=idx + 1)

        return

    def update_properties(
        self,
        idx: Union[int, List[int]],
        properties: Union[Dict[str, torch.tensor], List[Dict]],
    ):

        if utils.is_integer(idx):
            idx = [idx]
        if utils.is_dictionary(properties):
            properties = [properties]

        return self._update_properties(idx, properties)

    def _update_properties(
        self,
        idcs: List[int],
        properties: List[Dict[str, torch.tensor]],
    ):

        with data.connect(self.data_file, mode='a') as db:
            for idx, props in zip(idcs, properties):
                row_id = db.update(row_id=idx + 1, properties=props)

        return

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
        with data.connect(data_file, mode='r') as db:
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
        with data.connect(data_file, mode='a') as db:
            db.set_metadata(metadata)

    def check_metadata(
        self,
        data_file: Optional[str] = None,
        load_properties: Optional[List[str]] = None,
        unit_properties: Optional[Dict[str, str]] = None,
        update_metadata: Optional[bool] = True,
        overwrite_unit_properties: Optional[bool] = False,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Check (and update) metadata from the database file with current one

        Parameters
        ----------
        update_metadata: bool, optional, default True
            Update metadata in the database with a merged version of
            if the stored metadata and new one if no significant conflict
            arise between them.
        overwrite_unit_properties: bool, optional, default False
            In case of conflicting property units in metadata and input
            'unit_properties', overwrite metadata units with units in
            'unit_properties'.

        Returns
        -------
            dict
                Metadata of the database
        """

        # Check for custom data file path
        if data_file is None:
            data_file = self.data_file

        # Get metadata from dataset
        if os.path.isfile(data_file):
            metadata = self.get_metadata(data_file)
        else:
            metadata = {}

        # Check metadata loaded properties
        metadata = self.check_metadata_load_properties(
            metadata, load_properties)

        # Check metadata loaded properties units
        metadata = self.merge_metadata_unit_properties(
            metadata, unit_properties, 
            overwrite_unit_properties=overwrite_unit_properties)

        # Update metadata with database metadata
        if update_metadata:
            self.set_metadata(metadata=metadata)

        return metadata

    def check_metadata_load_properties(
        self,
        metadata: Dict[str, Any],
        load_properties: Dict[str, str],
    ) -> Dict[str, Any]:
        """
        Check and merge loaded property definition of metadata and input
        """

        # Compatibility check from previous version to remove eventually
        # 'positions' or 'charge' from loaded properties as they are stored in
        # the database by default. Also remove from 'load_properties' list for
        # the same reason.
        if metadata.get('load_properties') is not None:
            metadata['load_properties'] = list(metadata['load_properties'])
            if 'positions' in metadata['load_properties']:
                metadata['load_properties'].remove('positions')
            if 'charge' in metadata['load_properties']:
                metadata['load_properties'].remove('charge')
        if load_properties is not None:
            load_properties = list(load_properties)
            if 'positions' in load_properties:
                load_properties.remove('positions')
            if 'charge' in load_properties:
                load_properties.remove('charge')

        # Get loaded properties from metadata
        meta_load_properties = metadata.get('load_properties')

        # If nothing is defined - error
        if load_properties is None and meta_load_properties is None:
            raise SyntaxError(
                "Properties to load are neither defined by the input nor "
                + "by the dataset.")
        # If metadata is not defined put input - take input
        elif load_properties is not None and meta_load_properties is None:
            metadata['load_properties'] = list(load_properties)
        # If metadata is defined, but input is not - take metadata (useless)
        elif load_properties is None and meta_load_properties is not None:
            load_properties = metadata['load_properties']
        # If metadata and input are defined - check for conflicts
        else:
            check_properties = np.array([
                prop not in metadata['load_properties']
                for prop in load_properties], dtype=bool)
            if any(check_properties):
                raise ValueError(
                    f"Existing dataset '{self.data_file}' "
                    + "does not include properties "
                    + f"{np.array(load_properties)[check_properties]}, "
                    + "which is/are requested by 'load_properties'!")

        return metadata

    def merge_metadata_unit_properties(
        self,
        metadata: Dict[str, Any],
        unit_properties: Dict[str, str],
        overwrite_unit_properties: Optional[bool] = False,
    ) -> Dict[str, Any]:
        """
        Check and merge unit property definition of metadata and input
        """

        # Get unit properties from metadata
        meta_unit_properties = metadata.get('unit_properties')

        # If nothing is defined - error
        if unit_properties is None and meta_unit_properties is None:
            raise SyntaxError(
                "Property units are neither defined by the input nor "
                + "by the data set.")
        # If metadata is not defined put input - take input
        elif unit_properties is not None and meta_unit_properties is None:
            metadata['unit_properties'] = unit_properties
        # If metadata is defined, but input is not - take metadata (useless)
        elif unit_properties is None and meta_unit_properties is not None:
            unit_properties = metadata['unit_properties']
        # If metadata and input are defined - merge except for conflicts
        else:
            check_units = []
            for prop, item in unit_properties.items():
                if meta_unit_properties.get(prop) is None:
                    metadata['unit_properties'][prop] = item
                    check_units.append(False)
                else:
                    conversion, match = utils.check_units(
                        meta_unit_properties.get(prop), item)
                    if match:
                        check_units.append(False)
                    elif conversion==1.0:
                        message = (
                            "INFO:\nDeviation in property unit labels for "
                            + f"'{prop}' between input 'unit_properties' and "
                            + f"metadata in dataset '{self.data_file}', but "
                            + "with conversion factor of 1.0!\n"
                            + " Dataset metadata: "
                            + f"'{metadata['unit_properties'][prop]}'\n"
                            + f" Current input: '{item}'\n")
                        logger.info(message)
                        check_units.append(False)
                    else:
                        message = (
                            f"Deviation in property unit for '{prop}' "
                            + "between input 'unit_properties' and metadata"
                            + f"in dataset '{self.data_file}'!\n"
                            + " Dataset metadata: "
                            + f"'{metadata['unit_properties'][prop]}'\n"
                            + f" Current input: '{item}'\n")
                        if overwrite_unit_properties:
                            check_units.append(False)
                            message = "WARNING:\n" + message
                            message += "Dataset unit is overwritten!\n"
                            logger.warning(message)
                        else:
                            check_units.append(True)
                            message = "ERROR:\n" + message
                            logger.error(message)
            if any(check_units):
                raise ValueError(
                    "Property units in existing dataset file "
                    + f"'{self.data_file:s}' "
                    + "deviates from current input of 'unit_properties'!")

        # Check for charge unit or add default as charge unit was not
        # defined in 'unit_properties'
        if 'charge' not in metadata['unit_properties']:
            metadata['unit_properties']['charge'] = (
                settings._default_units['charge'])

        # Check for positions unit or add default as positions unit was not
        # defined in 'unit_properties'
        if 'positions' not in metadata['unit_properties']:
            metadata['unit_properties']['positions'] = (
                settings._default_args['data_unit_positions'])

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
        self.set_metadata(data_file=data_file, metadata=metadata)
        with data.connect(data_file, mode='a') as db:
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
            alt_property_labels=alt_property_labels,
            **kwargs)

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

    def __setitem__(
        self,
        idx: int,
        properties: Dict[str, torch.tensor],
    ):

        return self._set_properties([self.subset_idx[idx]], [properties])

    def get(
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

    def set_properties(
        self,
        idx: Union[int, List[int]],
        properties: Union[Dict[str, torch.tensor], List[Dict]],
    ):

        if utils.is_integer(idx):
            idx = [idx]
        if utils.is_dictionary(properties):
            properties = [properties]

        return self._set_properties(
            [self.subset_idx[idxi] for idxi in idx], properties)

    def update_properties(
        self,
        idx: Union[int, List[int]],
        properties: Union[Dict[str, torch.tensor], List[Dict]],
    ):

        if utils.is_integer(idx):
            idx = [idx]
        if utils.is_dictionary(properties):
            properties = [properties]

        return self._update_properties(
            [self.subset_idx[idxi] for idxi in idx], properties)
