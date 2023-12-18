import os
import json
import logging
from typing import Optional, List, Dict, Tuple, Union, Any

from .. import utils
from .. import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def check_config(
    config: Optional[Union[str, dict, object]] = None,
    config_file: Optional[str] = None,
    **kwargs,
) -> object:
    """
    Check configuration settings input in form of configuration
    class/dictionary object, configuration json-file path or direct keyword
    arguments (kwargs).

    In case of conflicting input, following priority is given from top to
    bottom (higher one overwrite input of lower priority).
    1. Keyword argument input
    2. Configuration class/dictionary object
    3. Configuration json-file

    Returns combined configuration as configuration class object.

    Parameters
    ----------
    config: Optional[Union[str, dict, object]]
        Configuration class/dictionary object, configuration json-file path or
        direct keyword arguments (kwargs).
    config_file: Optional[str]
        Configuration json-file path.
    **kwargs
        Keyword arguments for configuration settings.



    """

    # Check 'config_file' input type if given
    if config_file is not None and not utils.is_string(config_file):
        raise ValueError(
            "Configuration file path 'config_file' is ill defined " +
            f"(string expected but got {type(config_file)})!\n")

    # Check 'config' input and handle conflicts
    if config is None and config_file is None:

        # Initialize config dictionary
        config = {}

        # Get global configuration file path
        if settings._global_config_file is not None:
            config_file = settings._global_config_file
        else:
            config_file = settings._default_args['config_file']

        # Load configuration dictionary if available
        if os.path.exists(config_file):
            with open(config_file, 'r') as f:
                config_load = json.load(f)
        else:
            config_load = {}

        logger.info(
            "INFO:\nNo configuration or file path given!" +
            f"Default file path is used '{config_file}'.\n")

    elif utils.is_string(config):

        # Reassign config argument if config_file not given
        if config_file is None:
            logger.info(
                f"INFO:\nInput 'config' is a string ({config}) and assumed " +
                " to be the configuration file path 'config_file' instead!\n")
            config_file = config[:]
        elif utils.is_string(config_file) and config == config_file:
            logger.info(
                f"INFO:\nInput 'config' is a string ({config}) and " +
                "equivalent to configuration file path argument " +
                "'config_file'. Input in 'config' is ignored!\n")
        else:
            logger.WARNING(
                f"WARNING:\nInput 'config' is a string ({config}) and " +
                "conflicts with configuration file path 'config_file' " +
                f"({config_file})! Input in 'config' is ignored!\n")

        # Initialize config dictionary
        config = {}

    elif utils.is_dictionary(config) or utils.is_callable(config):

        # If config class object get respective dictionary
        if utils.is_callable(config):
            config = config.get_dictionary()

        # Check for conflicting configuration file path arguments
        if config_file is not None and config.get('config_file') is not None:
            if config_file != config.get('config_file'):
                logger.WARNING(
                    f"WARNING:\nInput argument 'config_file' ({config_file}) " +
                    f"differs from configuration dictionary input in " +
                    f"'config' ({config.get('config_file')})! " +
                    f"Input argument 'config_file' is kept and the " + 
                    f"respective configuration dictionary input is ignored!\n")
        elif config_file is None and config.get('config_file') is not None:
            config_file = config.get('config_file')
        else:
            if settings._global_config_file is not None:
                config_file = settings._global_config_file
            else:
                config_file = settings._default_args['config_file']

    else:

        raise ValueError(
            f"Neither inputs 'config' and 'config_file' are valid!\n" +
            f"'config' of type '{type(config)}' is expected to be a " +
            f"dictionary or config class object." +
            f"'config_file' of type '{type(config_file)}' is expected to be " +
            f"string of a json-file path.")
    
    # Set global configuration file path
    settings.set_global_config_file(config_file)
    
    # Update configuration according to priority list:
    # Configuration dictionary from json-file as configuration class object
    config_obj = settings.config()
    # gets updated by config dictionary input
    config_obj.update(config, overwrite=True)
    # gets updated by keyword argument input
    config_obj.update(kwargs, overwrite=True)
    
    
    # Check configuration entries data type and set default
    for key, item in config_obj.items():
        
        # Check if input parameter is None, if so take default value
        if key in settings._default_args.keys():
            if item is None:
                config_obj[key] = settings._default_args[key]
        
        # Check datatype of defined arguments
        if key in settings._dtypes_args.keys():
            _ = utils.check_input_dtype(
                key, config_obj[key], settings._dtypes_args, 
                raise_error=True)
    
    return config_obj
    
