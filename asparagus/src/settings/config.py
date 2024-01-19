
import os
import json
import logging
from typing import Optional, List, Dict, Tuple, Union

from .. import utils
from .. import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ======================================
# Configuration Functions
# ======================================


def get_config(
    config: Optional[Union[str, dict, object]] = None,
    config_file: Optional[str] = None,
    config_global: Optional[bool] = True,
    **kwargs,
):
    """
    Initialize Configuration class object. If 'config' input is already
    a class object, return itself.

    Parameters
    ----------

    config: (str, dict, object), optional, default None
        Either the path to json file (str), dictionary (dict) or
        configuration object of the same class (object) containing
        global model parameters
    config_file: str, optional, default see settings.default['config_file']
        Store global parameter configuration in json file of given path.
    config_global: bool, optional, default True
        If True, 'config_file' as json file path is set as default
        location for global model parameters.
    kwargs: dict, optional, default {}
        Keyword arguments for configuration parameter which are added to
        'config' or overwrite 'config' content.

    Returns
    -------
        object
            Configuration parameter object
    """

    # If 'config' being a config class object
    if utils.is_callable(config):

        # Reset configuration file path
        config.set_config_file(config_file, config_global)

        # Update configuration with keyword arguments
        if len(kwargs):
            config.update(kwargs)

        return config

    # Otherwise initialize Configuration class object
    else:

        return settings.Configuration(
            config=config,
            config_file=config_file,
            config_global=config_global,
            **kwargs)


# ======================================
# Configuration Class
# ======================================

class Configuration():
    """
    Global configuration object that contains all parameter about the
    model and training procedure.


    Parameters
    ----------

    config: (str, dict), optional, default None
        Either the path to json file (str) or dictionary (dict) containing
        global model parameters
    config_file: str, optional, default see settings.default['config_file']
        Store global parameter configuration in json file of given path.
    config_global: bool, optional, default True
        If True, 'config_file' as json file path is set as default
        location for global model parameters.
    kwargs: dict, optional, default {}
        Keyword arguments for configuration parameter which are added to
        'config' or overwrite 'config' content.

    Returns
    -------
    object
        Configuration parameter object


    """

    def __init__(
        self,
        config: Optional[Union[str, dict]] = None,
        config_file: Optional[str] = None,
        config_global: Optional[bool] = True,
        **kwargs,
    ):
        """
        Initialize config object.

        """

        # Initialize class parameter
        self.config_dict = {}
        self.config_indent = 4

        # Check and set configuration file path.
        if config_file is None:
            self.config_file = settings._global_config_file
        elif utils.is_string(config_file):
            self.config_file = config_file
        else:
            raise ValueError(
                "Input 'config_file' is not of valid data type!\n" +
                "Data type 'str' is expected for the config file path " +
                f"but '{type(config_file)}' is given.")

        # If no 'config' input given, load from global configuration file path
        if config is None:
            self.config_dict = self.read(self.config_file)
        # For 'config' being file path (str), load from json file and set
        # file path as default
        elif utils.is_string(config):
            self.config_file = config
            self.config_dict = self.read(self.config_file)
        # For 'config' being a dictionary, take it and look for config file
        # path in dictionary
        elif utils.is_dictionary(config):
            self.config_dict = config
            if config.get('config_file') is not None:
                self.config_file = config.get('config_file')
        else:
            raise ValueError(
                "Input 'config' is not of valid data type!\n" +
                "Data type 'dict', 'str' or a config class object " +
                f"is expected but '{type(config)}' is given.")

        # Set config_file as
        self.set_config_file(self.config_file, config_global)

        # Update configuration dictionary with keyword arguments
        if len(kwargs):
            self.update(kwargs)

        # Save current configuration dictionary to file
        self.dump()

    def __getitem__(self, args):
        return self.config_dict.get(args)

    def __setitem__(self, arg, item):
        self.config_dict[arg] = item
        self.dump()

    def __contains__(self, arg):
        return arg in self.config_dict.keys()

    def __call__(self, args):
        return self.config_dict.get(args)

    def items(self):
        for key, item in self.config_dict.items():
            yield key, item

    def get(self, args):
        if utils.is_array_like(args):
            return [self.config_dict.get(arg) for arg in args]
        else:
            return self.config_dict.get(args)

    def keys(self):
        return self.config_dict.keys()

    def read(
        self, 
        config_file: str,
    ) -> Dict:

        if os.path.exists(config_file):
            with open(config_file, 'r') as f:
                config_dict = json.load(f)
        elif os.path.exists(os.path.join(config_file, self.config_file)):
            with open(os.path.join(config_file, self.config_file), 'r') as f:
                config_dict = json.load(f)
        else:
            config_dict = {}

        return config_dict

    def set_config_file(self, config_file, config_global):
        """
        Check file path of the configuration json file and, if requested,
        set as global configuration file path
        """

        # If 'config_file' is None: (1) take 'config_file' input from
        # 'config' input or else (2) take global configuration file path
        self.config_file = self.check_config_file(config_file)

        # If requested, set 'config_file' as global configuration file path
        if config_global:
            settings.set_global_config_file(self.config_file)

    def check_config_file(self, config_file):
        """
        Check file path of the configuration json file and return
        the current file path.
        """

        if config_file is None:

            if self.config_dict.get('config_file') is None:
                config_file = settings._global_config_file
            else:
                config_file = self.config_dict.get('config_file')

        # If 'config_file' is defined, store in configuration dictionary
        elif utils.is_string(config_file):

            self.config_dict['config_file'] = config_file

        else:

            raise ValueError(
                "Input 'config_file' is not of valid data type!\n" +
                f"Data type 'str' is expected but '{type(config_file)}' " +
                "is given.")

        return config_file

    def update(
        self,
        config_new: Optional[Union[str, dict, object]],
        overwrite: Optional[bool] = True,
        verbose: Optional[bool] = True,
    ):
        """
        Update configuration dictionary.

        Parameters
        ----------

        config_new: (str, dict, object)
            Either the path to json file (str), dictionary (dict) or
            configuration object of the same class (object) containing
            new model parameters.
        overwrite: bool, optional, default True
            If True, 'config_new' input will be added and eventually overwrite
            existing entries in the configuration dictionary.
            If False, each input in 'config_new' will be only added if it is
            not already defined in the configuration dictionary.
        verbose: bool, optional, default True
            For conflicting entries between 'config_new' and current
            configuration dictionary, print further information.
        """

        # Check config_new input
        if utils.is_string(config_new):
            config_new = self.read(config_new)
        elif utils.is_dictionary(config_new):
            pass
        elif utils.is_callable(config_new):
            config_new = config_new.get_dictionary()
        else:
            raise ValueError(
                "Input 'config_new' is not of valid data type!\n" +
                "Data type 'dict', 'str' or a config class object " +
                f"is expected but '{type(config_new)}' is given.")

        # Return if update dictionary is empty
        if not len(config_new):
            logger.info("INFO:\nEmpty update configuration dictionary!\n")
            return

        # Show update information
        msg = (
            "Global parameter configuration update of " +
            f"'{self.config_file}'.\n")
        if overwrite:
            msg += (
                "Current configuration entries will be overwritten " +
                "in case of conflicting entries.\n")
        else:
            msg += (
                "Current configuration entries will be only extended  " +
                "by new configuration entries " +
                "but not in case of conflicting entries.\n")
        logger.info("INFO:\n" + msg)

        # Prepare additional information output
        msg = ""

        # Iterate over new configuration dictionary
        for key, item in config_new.items():

            # Check for conflicting keyword
            conflict = key in self.config_dict.keys()

            # Add or update parameter
            if conflict and overwrite:

                self.config_dict[key] = config_new.get(key)
                if verbose:
                    msg += f"Conflict! Overwrite parameter '{key}'.\n"

            elif not conflict:

                self.config_dict[key] = config_new.get(key)
                if verbose:
                    msg += f"Adding parameter '{key}'.\n"

            else:

                if verbose:
                    msg += f"Conflict! Ignore parameter '{key}'.\n"

        # Show additional information output
        if verbose:
            logger.info("INFO:\n" + msg)

        # Store changes in file
        self.dump()

        return

    def dump(
        self,
        config_file: Optional[str] = None
    ):
        """
        Save configuration dictionary to json file

        Parameters
        ----------

        config_file: str, optional, default None
            Dump current config dictionary in this file path.
        """

        # Initialize dictionary with JSON compatible parameter types
        config_dump = {}

        # Iterate over configuration parameters
        for key, item in self.config_dict.items():
            
            # Skip callable class objects
            if utils.is_callable(item):
                continue
            
            # Convert numeric values to integer or float
            if utils.is_integer(item):
                config_dump[key] = int(item)
            elif utils.is_numeric(item):
                config_dump[key] = float(item)
            # Also store dictionaries,
            elif utils.is_dictionary(item):
                config_dump[key] = item
            # strings or bools
            elif utils.is_string(item) or utils.is_bool(item):
                config_dump[key] = item
            # and converted arrays as python lists,
            # but nothing else which might be to fancy
            elif utils.is_array_like(item):
                config_dump[key] = list(item)
            else:
                continue

        if config_file is None:
            with open(self.config_file, 'w') as f:
                json.dump(
                    config_dump, f, 
                    indent=self.config_indent,
                    default=str)
        else:
            with open(config_file, 'w') as f:
                json.dump(
                    config_dump, f, 
                    indent=self.config_indent,
                    default=str)

    def check(
        self,
        check_dtype: Optional[bool] = True,
        check_default: Optional[bool] = True,
    ):
        """
        Check model parameter configuration by data type and set default.

        Parameters
        ----------

        check_dtype: bool, optional, default True
            Check data type of model parameter and raise Error for mismatch
        check_default: bool, optional, default True
            If model parameter is None replace with parameter from default list
            if available.
        """

        for key, item in self.config_dict.items():

            # Check if input parameter is None, if so take default value
            if item is None and check_default:
                if key in settings._default_args.keys():
                    self.config_dict[key] = settings._default_args[key]

            # Check datatype of defined arguments
            if key in settings._dtypes_args.keys() and check_dtype:
                _ = utils.check_input_dtype(
                    key, self.config_dict[key], settings._dtypes_args,
                    raise_error=True)

    def get_file_path(self):
        return self.config_file

    def get_dictionary(self):
        return self.config_dict
