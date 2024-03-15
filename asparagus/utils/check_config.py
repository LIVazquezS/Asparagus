import os
import json
import logging
from typing import Optional, List, Dict, Tuple, Union, Any

from .. import utils
from .. import settings

# --------------- ** Checking input parameter ** ---------------

def check_input_dtype(
    arg, 
    item, 
    dtypes_args, 
    raise_error=True, 
    return_info=False):
    """
    Check it the item (not None) of arg(ument) matchs the expectation 
    in dtypes_args.
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
    property_label, 
    valid_property_labels: Optional[List[str]] = None,
    alt_property_labels: Optional[Dict[str, List[str]]] = None, 
    return_modified: Optional[bool] = True,
) -> Dict[str, List[str]]:
    """
    Validate the property label by comparing with valid property labels in 
    'valid_property_label' or compare with alternative labels in
    'alt_property_labels'. If valid or found in 'alt_properties', the valid
    lower case form is returned with bool for match and if modified.

    Parameters
    ----------
    property_label : str
        Property labels to be checked.
    valid_property_labels : list(str), optional, default None
        List of valid property labels. If not defined, valid property labels
        are taken from settings._valid_properties.
    alt_property_labels: dict, optional, default None
        Dictionary with alternative property labels as keys and valid property
        labels as values. If not defined, no check for alternatively spelled
        properties is done.
    return_modified : bool, optional, default True
        Return if property label was modified.

    """
    
    # Check if property label is valid
    if valid_property_labels is None:
        valid_property_labels = settings._valid_properties
    if property_label.lower() in valid_property_labels:
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
    if alt_property_labels is None:
        alt_property_labels = settings._alt_property_labels
    for key, items in alt_property_labels.items():
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

def merge_dictionaries(
    dict_old: Dict[str, Any],
    dict_new: Dict[str, Any],
    keep: Optional[bool] = False,
):
    """
    Merge keys and items of both dictionaries. If 'keep' is False, update
    key in dict_old with item of dict_new.
    """

    # Check dictionaries
    if dict_old is None and dict_new is None:
        return {}
    if dict_old is None:
        return dict_new
    if dict_new is None:
        return dict_old

    # Iterate over keys
    for key, item in dict_old.items():
        if key in dict_new:
            if keep:
                dict_new[key] = item
        else:
            dict_new[key] = item

    return dict_new

def merge_dictionary_lists(
    dictA: Dict[str, List[str]], 
    dictB: Dict[str, List[str]],
):
    """
    Combine two dictionaries lists and check for conflicts in item lists.
    If an item in the lists reüeats in dictA,  the assignment of 'dictA' is 
    kept.
    """
    
    # Combined dictionary
    dictC = {}
    
    # Observed items in dictA and dictB
    observed_items = []
    
    # Iterate over dictA
    for keyA, itemsA in dictA.items():
        dictC[keyA] = itemsA
        for itemA in itemsA:
            if itemA not in observed_items:
                dictC[keyA].append(itemA)
                observed_items.append(itemA)

    # Iterate over dictB
    for keyB, itemsB in dictB.items():
        if keyB not in dictC.keys():
            dictC[keyB] = []
        for itemB in itemsB:
            # If itemB already appeared, raise error or drop itemB
            if itemB not in observed_items:
                dictC[keyB].append(itemB)
                observed_items.append(itemB)

    return dictC

# --------------- ** Combine Default Dictionaries ** ---------------

def get_default_args(
    self_class, 
    self_module,
):
    """
    Combine available default argument dictionaries. In case of conflicts, the
    priority is from top to bottom: self_class, self_module, settings
    """
    
    # Get default argument dictionary
    default_args = settings._default_args
    
    # Add and overwrite with module default arguments
    if hasattr(self_module, '_default_args'):
        default_args.update(self_module._default_args)
    
    # Add and overwrite with class default arguments
    if hasattr(self_class, '_default_args'):
        default_args.update(self_class._default_args)

    return default_args


def get_dtype_args(
    self_class, 
    self_module,
):
    """
    Combine available argument data type dictionaries. In case of conflicts,
    the priority is from top to bottom: self_class, self_module, settings
    """
    
    # Get default argument dictionary
    dtype_args = settings._dtypes_args
    
    # Add and overwrite with module arguments data types
    if self_module is not None and hasattr(self_module, '_dtypes_args'):
        dtype_args.update(self_module._dtypes_args)
    
    # Add and overwrite with class arguments data types
    if self_class is not None and hasattr(self_class, '_dtypes_args'):
        dtype_args.update(self_class._dtypes_args)
    
    return dtype_args
