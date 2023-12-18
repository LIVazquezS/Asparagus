import os
import numpy as np

import torch

from .. import utils

# --------------- ** Checking input parameter ** ---------------

def check_input_dtype(
    arg, item, dtypes_args, raise_error=True, return_info=False):
    """
    Check it the item (not None) of arg(ument) matchs the expectation 
    in dtypes_args.

    Parameters
    ----------
    arg : str
        Argument name.
    item : object
        Item to be checked.
    dtypes_args : dict
        Dictionary with argument name as key and a list of functions
        as value. The functions should return a bool and the input
        and expected dtype.
    raise_error : bool, optional
        Raise error if item does not match expectation. The default is True.
    return_info : bool, optional
        Return information about the check. The default is False.

    """
    
    if arg in dtypes_args.keys() and len(dtypes_args[arg]) and item is not None:
        matches, dtype_expectation = [], []
        for is_dtype in dtypes_args[arg]:
            match, dtype_input, dtype_expected = is_dtype(item, verbose=True)
            matches.append(match)
            dtype_expectation.append(dtype_expected)
        if not any(matches) and raise_error:
            raise ValueError(
                f"Argument '{arg:s}' has wrong dtype!\n" +
                f"Input: {dtype_input}\n" +
                f"Expected: {dtype_expectation}")
        
        if return_info:
            return any(matches), dtype_input, dtype_expectation
        else:
            return any(matches)
            
    
    else:
        
        if return_info:
            return False, None, None
        else:
            return False


# --------------- ** Checking property labels ** ---------------

def check_property_label(
    property_label, valid_properties, alt_properties, return_modified=True):
    """
    Validate the property label by comparing with valid property labels in 
    'valid_properties' or compare with alternative labels in 'alt_properties'.
    If valid or found in 'alt_properties', the valid lower case form is
    returned with bool for match and if modified.

    Parameters
    ----------

    property_label : str
        Property label to be checked.
    valid_properties : list
        List of valid property labels.
    alt_properties : dict
        Dictionary with alternative property labels as keys and valid property
        labels as values.
    return_modified : bool, optional
        Return if property label was modified. The default is True.


    """
    
    # Check if property label is valid
    if property_label.lower() in valid_properties:
        # If already lower case or not
        if property_label.lower() == property_label:
            if return_modified:
                return True, False, property_label.lower()
            else:
                return True
        else:
            if return_modified:
                return True, True, property_label.lower()
            else:
                return True
            
    # Check if a valid alternative can be found for property label
    for key, items in alt_properties.items():
        if utils.is_string(items):
            items_lower = [items.lower()]
        else:
            items_lower = [item.lower() for item in items]
        if property_label.lower() in items_lower:
            if return_modified:
                return True, True, key.lower()
            else:
                return True
        
    # Property label is not valid nor is an alternative found.
    if return_modified:
        return False, False, property_label
    else:
        return False
    

# --------------- ** Combine dictionaries ** ---------------

def combine_dictionaries(
    dictA, dictB, logger=None, logger_info='', raise_error=False):
    """
    Combine two dictionaries and check for repetitions.
    In case of same item (element in item) at different keys, 
    the assignment of 'dictA' is kept.

    Parameters
    ----------
    dictA : dict
        Dictionary A.
    dictB : dict
        Dictionary B.
    logger : logging.Logger, optional
        Logger for warnings. The default is None.
    logger_info : str, optional
        Information for logger. The default is ''.
    raise_error : bool, optional
        Raise error if item is already assigned. The default is False.
    """
    
    # Combined dictionary
    dictC = {}
    
    # Observed items in dictA and dictB
    observed_items = []
    
    # Iterate over dictA
    for keyA, itemsA in dictA.items():
        dictC[keyA] = []
        for itemA in itemsA:
            # If itemA already appeared, raise error or drop itemA
            if itemA in observed_items:
                if itemA in dictC[keyA]:
                    if logger is not None:
                        logger.warning(
                            f"WARNING:\n{logger_info}" +
                            f"Item '{itemA}' already assigned in " + 
                            f"key '{keyA}'!\n")
                elif raise_error:
                    raise ValueError(
                        f"Item '{itemA}' in key '{keyA}' already assigned!")
                else:
                    if logger is not None:
                        logger.warning(
                            f"WARNING:\n{logger_info}" +
                            f"Item '{itemA}' in key '{keyA}' " +
                            f"already assigned elsewhere!\n")
            # Else add itemA to keyA list
            else:
                dictC[keyA].append(itemA)
                observed_items.append(itemA)

    # Iterate over dictB
    for keyB, itemsB in dictB.items():
        if keyB not in dictC.keys():
            dictC[keyB] = []
        for itemB in itemsB:
            # If itemB already appeared, raise error or drop itemB
            if itemB in observed_items:
                if itemB in dictC[keyB]:
                    if logger is not None:
                        logger.warning(
                            f"WARNING:\n{logger_info}" +
                            f"Item '{itemB}' already " +
                            f"assigned in key '{keyB}'!\n")
                elif raise_error:
                    raise ValueError(
                        f"Item '{itemB}' in key '{keyB}' already assigned!")
                else:
                    if logger is not None:
                        logger.warning(
                            f"WARNING:\n{logger_info}" +
                            f"Item '{itemB}' in key '{keyB}' " +
                            f"already assigned elsewhere!\n")
            # Else add itemB to keyB list
            else:
                dictC[keyB].append(itemB)
                observed_items.append(itemB)

    return dictC

