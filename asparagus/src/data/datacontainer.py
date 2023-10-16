import os
import logging
from typing import Optional, List, Dict, Tuple, Union, Any, Callable

import numpy as np

from .. import data
from .. import utils
from .. import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

__all__ = ['DataContainer']


class DataContainer():
    """
    A PhysNet-specific data storage build on
    Pytorch Lightening LightningDataModule
    """

    def __init__(
        self,
        config: Optional[Union[str, dict, object]] = None,
        data_file: Optional[str] = None,
        data_source: Optional[Union[str, List[str]]] = None,
        data_format: Optional[Union[str, List[str]]] = None,
        data_alt_property_labels: Optional[dict] = None,
        data_unit_positions: Optional[str] = None,
        data_load_properties: Optional[List[str]] = None,
        data_unit_properties: Optional[Dict[str, str]] = None,
        data_num_train: Optional[Union[int, float]] = None,
        data_num_valid: Optional[Union[int, float]] = None,
        data_num_test: Optional[Union[int, float]] = None,
        data_train_batch_size: Optional[int] = None,
        data_valid_batch_size: Optional[int] = None,
        data_test_batch_size: Optional[int] = None,
        data_num_workers: Optional[int] = None,
        data_workdir: Optional[str] = None,
        data_overwrite: Optional[bool] = None,
        data_seed: Optional[int] = None,
        **kwargs,
    ):
        """
        DataContainer object that manage the distribution of the reference
        data from one or multiple databases into a DataSet object and provide
        DataSubSets for training, validation and test sets.

        Parameters
        ----------
        config: (str, dict, object)
            Either the path to json file (str), dictionary (dict) or
            settings.config class object of model parameters
        data_file: str, optional, default 'data.db'
            Reference database file
        data_source: (str, List(str)), optional, default None
            Path to reference data set(s)
        data_format: (str, List(str)), optional, default file extension
            Dataset format of 'data_source'
        data_alt_property_labels: dictionary, optional, default
                'settings._alt_property_labels'
            Dictionary of alternative property labeling to replace
            non-valid property labels with the valid one if possible.
        data_unit_positions: str, optional, default 'Ang'
            Unit of the atom positions ('Ang' or 'Bohr') and other unit
            cell information.
        data_load_properties: List(str), optional,
                default ['energy', 'forces', 'dipole']
            Set of properties to store in the DataSet
        data_unit_properties: dictionary, optional, default {'energy':'eV'}
            Dictionary from properties (keys) to corresponding unit as a
            string (item), e.g.:
                {property: unit}: { 'energy', 'eV',
                                    'forces', 'eV/Ang', ...}
        data_num_train: (int, float), optional, default 0.8 (80% of data)
            Number of training data points [absolute (>1) or relative
            (<= 1.0)].
        data_num_valid: (int, float), optional, default 0.1 (10% of data)
            Number of validation data points [absolute (>1) or relative
            (<= 1.0)].
        data_num_test: (int, float), optional, default 0.1 (10% of data)
            Number of test data points [absolute (>1) or relative (< 1.0)].
        data_train_batch_size: int, optional, default 64
            Training batch size

        data_valid_batch_size: int, optional, default 64
            Validation batch size
        data_test_batch_size:  int, optional, default 64
            Test batch size
        data_num_workers: int, optional, default 1
            Number of data loader workers
        data_workdir: str, optional, default '.'
            Copy data here as part of setup, e.g. to a local file system
            for faster performance.
        data_overwrite: bool, optional, default False
            Overwrite database files with reference data from
            'data_source' if available.
        data_seed: (int, float), optional, default: np.random.randint(1E6)
            Define seed for random data splitting.

        Returns
        -------
        object
            DataContainer for data management

        """

        super().__init__()

        #####################################
        # # # Check DataContainer Input # # #
        #####################################

        # Get configuration object
        config = settings.get_config(config)

        # Additionally, check input from existing DataSet reference file if
        # not flagged for overwrite
        if data_file is None:
            data_file = config.get('data_file')
        if data_overwrite:
            metadata = {}
        else:
            if config.get('data_overwrite'):
                metadata = {}
            else:
                metadata = data.get_metadata(data_file)

        # Check loaded properties, if not defined in config or input,
        # use item in metadata, otherwise it would take default that can
        # clash with dataset metadata
        if (
                config.get('data_load_properties') is None 
                and data_load_properties is None
        ):
            data_load_properties = metadata.get('load_properties')

        # Check property units, if not defined in config or input,
        # use item in metadata, otherwise it would take default that can
        # clash with dataset metadata
        if (
                config.get('data_unit_properties') is None 
                and data_unit_properties is None
        ):
            data_unit_properties = metadata.get('unit_properties')
        # Else, merge eventually config into input but keep input in case
        # of conflict
        elif (
            (config.get('data_unit_properties') is not None)
            and (data_unit_properties is not None)
        ):
            data_unit_properties = {
                **config.get('data_unit_properties'), **data_unit_properties}

        # Check input parameter, set default values if necessary and
        # update the configuration dictionary
        config_update = {}
        for arg, item in locals().items():

            # Skip 'config' argument and possibly more
            if arg in [
                    'self', 'config', 'config_update', 'metadata', 'kwargs',
                    '__class__']:
                continue

            # Take argument from global configuration dictionary if not defined
            # directly
            if item is None:
                item = config.get(arg)

            # Get argument from data set file if defined
            if item is None and arg in metadata:
                item = metadata.get(arg)

            # Set default value if the argument is not defined (None)
            if arg in settings._default_args.keys():
                if item is None:
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

        #####################################
        # # # Check Data Property Input # # #
        #####################################

        # Initialize data flag (False until setup is finished)
        self.data_avaiable = False

        # Make defined data_source iterable if necessary
        if not utils.is_array_like(self.data_source):
            self.data_source = [self.data_source]

        # Check data_source file extension
        if not len(self.data_format):
            self.data_format = []
            for path in self.data_source:
                self.data_format.append(path.split('.')[-1])
        elif utils.is_string(self.data_format):
            self.data_format = [self.data_format]*len(self.data_source)
        elif len(self.data_format) != len(self.data_source):
            raise ValueError(
                "Number of arguments in 'data_format' does not match " +
                "number of arguments in 'data_source'!")

        # Combine alternative property label input 'data_alt_property_labels'
        # with default setting, check for repetitions.
        self.data_alt_property_labels = utils.combine_dictionaries(
            self.data_alt_property_labels, settings._alt_property_labels,
            logger=logger, logger_info=(
                "Alternative property labels 'data_alt_property_labels':\n"))

        # Check for unknown property labels in data_load_properties
        # and replace if possible with internally used property label in
        # data_alt_property_labels
        for ip, prop in enumerate(self.data_load_properties):
            match, modified, new_prop = utils.check_property_label(
                prop, settings._valid_properties,
                self.data_alt_property_labels)
            if match and modified:
                logger.warning(
                    f"WARNING:\nProperty key '{prop}' in " +
                    "'data_load_properties' is not a valid label!\n" +
                    f"Property key '{prop}' is replaced by '{new_prop}'.\n")
                self.data_load_properties[ip] = new_prop
            elif not match:
                raise ValueError(
                    f"Unknown property ('{prop}') in 'data_load_properties'!")

        # Check for unknown property labels in data_unit_properties
        # and replace if possible with internally used property label in
        # data_alt_property_labels
        props = list(self.data_unit_properties.keys())
        for prop in props:
            match, modified, new_prop = utils.check_property_label(
                prop, settings._valid_properties,
                self.data_alt_property_labels)
            if match and modified:
                logger.warning(
                    f"WARNING:\nProperty key '{prop}' in " +
                    "'data_unit_properties' is not a valid label!\n" +
                    f"Property key '{prop}' is replaced by '{new_prop}'.\n")
                self.data_unit_properties[new_prop] = (
                    self.data_unit_properties.pop(prop))
            elif not match:
                raise ValueError(
                    f"Unknown property ('{prop}') in 'data_unit_properties'!")

        # Check if all units from data_load_properties are defined in
        # data_unit_properties, if not assign default units
        for prop in self.data_load_properties:
            if prop not in self.data_unit_properties.keys():
                logger.warning(
                    f"WARNING:\nNo unit defined for property '{prop}'!\n" +
                    f"Default unit of '{settings._default_units[prop]}' " +
                    "will be used.\n")
                self.data_unit_properties[prop] = settings._default_units[prop]

        # Check if positions unit is defined in data_unit_properties
        if 'positions' not in self.data_unit_properties:
            self.data_unit_properties['positions'] = self.data_unit_positions
        else:
            self.data_unit_positions = self.data_unit_properties['positions']
            config['data_unit_positions'] = self.data_unit_positions

        #########################
        # # # Check DataSet # # #
        #########################

        # Check data set parameters if file does not get overwritten
        if not self.data_overwrite and os.path.isfile(self.data_file):
            with data.connect(self.data_file, mode='r') as db:
                Ndata = db.count()
            if not Ndata:
                logger.warning(
                    f"WARNING:\nData file '{self.data_file}' is empty! " +
                    "File will be overwritten.\n")
                self.data_overwrite = True

        if self.data_overwrite and not len(self.data_source):
            logger.warning(
                "WARNING:\nNo source files are defined in 'data_source'! " +
                f"Data file '{self.data_file}' will not be overwritten.\n")
            self.data_overwrite = False

        # Set up reference data set
        self.dataset_setup(
            data_overwrite=self.data_overwrite,
            **kwargs)

        # Reset data overwrite
        self.data_overwrite = False
        config['data_overwrite'] = False

        return

    def dataset_setup(
        self,
        data_overwrite: Optional[bool] = None,
        **kwargs,
    ):
        """
        Setup the reference data set
        """

        #########################
        # # # DataSet Setup # # #
        #########################

        # Check dataset overwrite parameter
        if data_overwrite is None:
            data_overwrite = self.data_overwrite

        # Initialize dataset
        self.dataset = data.DataSet(
            self.data_file,
            self.data_load_properties,
            self.data_unit_properties,
            data_overwrite=data_overwrite,
            **kwargs)

        # Load reference data set(s) from defined source data path(s)
        for ip, (data_path, data_format) in enumerate(
            zip(self.data_source, self.data_format)
        ):
            if os.path.exists(data_path):
                self.dataset.load(
                    data_path,
                    data_format,
                    self.data_alt_property_labels,
                    **kwargs)
            else:
                logger.warning(
                    f"WARNING:\nFile {data_path} from 'data_source' " +
                    "does not exist! Skipped.\n")

        # Update datacontainer parameters
        self.metadata = self.dataset.get_metadata()
        self.data_source = self.metadata.get('data_source')
        self.data_format = self.metadata.get('data_format')
        self.data_load_properties = self.metadata.get('load_properties')
        self.data_unit_properties = self.metadata.get('unit_properties')

        # Prepare data split into training, validation and test set
        Ndata = len(self.dataset)

        # Stop further setup if no data are available
        if not Ndata:
            raise ValueError(
                f"No data are available in {self.data_file}!\n")

        ###########################
        # # # Data Separation # # #
        ###########################

        # Training set size
        if self.data_num_train < 0.0:
            raise ValueError(
                "Number of training set samples 'data_num_train' " +
                f"({self.data_num_train}) is lower then zero and invalid!\n")
        elif self.data_num_train <= 1.0:
            self.rel_train = float(self.data_num_train)
            self.data_num_train = int(Ndata*self.rel_train)
            self.rel_train = float(self.data_num_train)/float(Ndata)
        elif self.data_num_train <= Ndata:
            self.rel_train = float(self.data_num_train)/float(Ndata)
        else:
            raise ValueError(
                "Number of training set samples 'data_num_train' " +
                f"({self.data_num_train}) is larger than the total number " +
                f"of data samples ({Ndata})!\n")

        # Validation set size
        if self.data_num_valid < 0.0:
            raise ValueError(
                "Number of validation set samples 'data_num_valid' " +
                f"({self.data_num_valid}) is lower then zero and invalid!\n")
        elif self.data_num_valid < 1.0:
            self.rel_valid = float(self.data_num_valid)
            if (self.rel_train + self.rel_valid) > 1.0:
                rel_valid = 1.0 - float(self.data_num_train)/float(Ndata)
                logger.warning(
                    f"WARNING:\nRatio of training set ({self.rel_train})" +
                    f"and validation set samples ({self.rel_valid}) " +
                    "are larger 1.0!\n" +
                    "Ratio of validation set samples is set to " +
                    f"{rel_valid}.\n")
                self.rel_valid = rel_valid
            self.data_num_valid = int(round(Ndata*self.rel_valid))
            self.rel_valid = float(self.data_num_valid)/float(Ndata)
        elif self.data_num_valid <= (Ndata - self.data_num_train):
            self.rel_valid = float(self.data_num_valid)/float(Ndata)
        else:
            data_num_valid = int(Ndata - self.data_num_train)
            logger.warning(
                f"WARNING:\nNumber of training set ({self.data_num_train})" +
                "and validation set samples " +
                f"({self.data_num_valid}) are larger then number of " +
                f"data samples ({Ndata})!\n" +
                "Number of validation set samples is set to " +
                f"{data_num_valid}.\n")
            self.data_num_valid = data_num_valid
            self.rel_valid = float(self.data_num_valid)/float(Ndata)

        # Test set size
        if self.data_num_test is None:
            self.data_num_test = (
                Ndata - self.data_num_train - self.data_num_valid)
            self.rel_test = float(self.data_num_test)/float(Ndata)
        elif self.data_num_test < 0.0:
            raise ValueError(
                "Number of test set samples 'data_num_test' " +
                f"({self.data_num_test}) is lower then zero and invalid!\n")
        elif self.data_num_test < 1.0:
            self.rel_test = float(self.data_num_test)
            if (self.rel_test + self.rel_train + self.rel_valid) > 1.0:
                rel_test = (
                    1.0 - float(self.data_num_train + self.data_num_valid)
                    / float(Ndata))
                logger.warning(
                    f"WARNING:\nRatio of test set ({self.rel_test})" +
                    "with training and validation set samples " +
                    f"({self.rel_train}, {self.rel_valid}) " +
                    "are larger 1.0!\n" +
                    "Ratio of test set samples is set to " +
                    f"{rel_test}.\n")
                self.rel_test = rel_test
            self.data_num_test = int(round(Ndata*self.rel_test))
        elif self.data_num_test <= (
                Ndata - self.data_num_train - self.data_num_valid):
            self.rel_test = float(self.data_num_test)/float(Ndata)
        else:
            data_num_test = int(
                Ndata - self.data_num_train - self.data_num_valid)
            logger.warning(
                f"WARNING:\nNumber of training ({self.data_num_train}), " +
                f"validation set ({self.data_num_valid}) and " +
                f"test set samples ({self.data_num_test}) are larger " +
                f"then number of data samples ({Ndata})!\n" +
                "Number of test set samples is set to " +
                f"{data_num_test}.\n")
            self.data_num_test = data_num_test
            self.rel_test = float(self.data_num_test)/float(Ndata)

        ############################
        # # # DataSubSet Setup # # #
        ############################

        # Select training, validation and test data indices randomly
        np.random.seed(self.data_seed)
        idx_data = np.random.permutation(np.arange(Ndata))
        self.idx_train = idx_data[:self.data_num_train]
        self.idx_valid = idx_data[
            self.data_num_train:(self.data_num_train + self.data_num_valid)]
        self.idx_test = idx_data[
            (self.data_num_train + self.data_num_valid):
            (self.data_num_train + self.data_num_valid + self.data_num_test)]

        # Prepare training, validation and test subset
        self.train_set = data.DataSubSet(
            self.data_file,
            self.idx_train,
            self.data_load_properties,
            self.data_unit_properties,
            False)
        self.valid_set = data.DataSubSet(
            self.data_file,
            self.idx_valid,
            self.data_load_properties,
            self.data_unit_properties,
            False)
        self.test_set = data.DataSubSet(
            self.data_file,
            self.idx_test,
            self.data_load_properties,
            self.data_unit_properties,
            False)
        logger.info(
            f"INFO:\n{Ndata:d} reference data points are distributed on " +
            "data subsets.\n" +
            f"Training data:   {len(self.train_set): 6d} " +
            f"({len(self.train_set)/Ndata*100: 3.1f}%)\n" +
            f"Validation data: {len(self.valid_set): 6d} " +
            f"({len(self.valid_set)/Ndata*100: 3.1f}%)\n" +
            f"Test data:       {len(self.test_set): 6d} " +
            f"({len(self.test_set)/Ndata*100: 3.1f}%)\n")

        ############################
        # # # DataLoader Setup # # #
        ############################

        # Prepare training, validation and test data loader
        self.train_loader = data.DataLoader(
            self.train_set,
            self.data_train_batch_size,
            True,
            self.data_num_workers)
        self.valid_loader = data.DataLoader(
            self.valid_set,
            self.data_valid_batch_size,
            False,
            self.data_num_workers)
        self.test_loader = data.DataLoader(
            self.test_set,
            self.data_test_batch_size,
            False,
            self.data_num_workers)

        # Prepare dictionaries as pointers between dataset label and the
        # respective DataSubSet and DataLoader objects
        self.all_data_sets = {
            'train': self.train_set,
            'valid': self.valid_set,
            'test': self.test_set}
        self.all_data_loder = {
            'train': self.train_loader,
            'valid': self.valid_loader,
            'test': self.test_loader}

        # Set data flag
        self.data_avaiable = True

    def get_property_scaling(
        self,
        overwrite: Optional[bool] = False,
        property_atom_scaled: Optional[List[str]] = ['energy'],
    ) -> Dict[str, List[float]]:
        """
        Get property scaling factors and shift terms equivalent to
        the property mean value and standard deviation.
        """

        # Get current metadata dictionary
        metadata = self.dataset.get_metadata()

        # Initialize scaling and shift parameter dictionary
        property_scaling = {}
        for prop in metadata.get('load_properties'):
            # List of property mean value and standard deviation
            property_scaling[prop] = [0.0, 0.0]

        # Check property scaling status
        if (
            metadata.get('data_uptodate_property_scaling') is not None 
            and not overwrite
        ):
            return metadata.get('data_property_scaling')

        # Iterate over training data properties and compute property mean
        logger.info(
            "INFO:\nStart computing means for training data property. " +
            "This might take a moment.")
        Nsamples = 0.0
        for sample in self.train_set:

            # Iterate over sample properties
            for prop in metadata.get('load_properties'):

                vals = sample.get(prop).numpy().reshape(-1)

                # Scale by atom number if requested
                if prop in property_atom_scaled:
                    Natoms = sample.get('atoms_number').numpy().reshape(-1)
                    vals /= Natoms.astype(float)

                scalar = np.mean(vals)
                property_scaling[prop][1] = (
                    property_scaling[prop][1]
                    + (scalar - property_scaling[prop][1])/(Nsamples + 1.0)
                    ).item()

            # Increment sample counter
            Nsamples += 1.0

        # Iterate over training data properties and compute standard deviation
        logger.info(
            "INFO:\nStart computing standard deviation for training data " +
            "property. This might take a moment.")
        for sample in self.train_set:

            # Iterate over sample properties
            for prop in metadata.get('load_properties'):

                vals = sample.get(prop).numpy().reshape(-1)
                Nvals = len(vals)

                # Scale by atom number if requested
                if prop in property_atom_scaled:
                    Natoms = sample.get('atoms_number').numpy().reshape(-1)
                    vals /= Natoms.astype(float)

                for scalar in vals:

                    property_scaling[prop][0] = (
                        property_scaling[prop][0]
                        + (scalar - property_scaling[prop][1])**2/Nvals)

        logger.info("INFO:\nDone.\n")

        # Iterate over sample properties to complete standard deviation
        for prop in metadata.get('load_properties'):

            property_scaling[prop][0] = np.sqrt(
                property_scaling[prop][0]/Nsamples).item()

        # Update property scaling
        if metadata.get('data_property_scaling') is None:
            metadata['data_property_scaling'] = property_scaling
        else:
            metadata['data_property_scaling'].update(property_scaling)
        metadata['data_uptodate_property_scaling'] = True
        self.dataset.set_metadata(metadata=metadata)

        return property_scaling

    def get_datalabels(self) -> List[str]:
        """
        Return the list of all available data set labels which return
        DataSubSet or DataLoader objects with the respective function
        (get_dataset and get_dataloader).
        """
        return self.all_data_sets.keys()

    def get_dataset(self, label) -> Callable:
        """
        Return as specific DataSubSet object ('train', 'valid' or 'test')
        """
        return self.all_data_sets.get(label)

    def get_dataloader(self, label) -> Callable:
        """
        Return as specific DataLoader object ('train', 'valid' or 'test')
        """
        return self.all_data_loder.get(label)

    def get_metadata(self) -> Dict[str, Any]:
        """
        Return metadata
        """
        return self.dataset.get_metadata()

    def get_info(self) -> Dict[str, Any]:
        """
        Return data information
        """

        return {
            'data_load_properties': self.data_load_properties,
            'data_unit_properties': self.data_unit_properties,
            'data_num_train': self.data_num_train,
            'data_num_valid': self.data_num_valid,
            'data_num_test': self.data_num_test,
            'data_overwrite': self.data_overwrite,
            }
