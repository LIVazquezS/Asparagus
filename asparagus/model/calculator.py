
import torch

#======================================
# Calculator Model Provision
#======================================

def get_Model_PhysNet():
    from .physnet import Model_PhysNet
    return Model_PhysNet

def get_Model_PaiNN():
    raise NotImplementedError()
    from .painn import Model_PaiNN
    return Model_PaiNN


#======================================
# Calculator Model Assignment
#======================================

model_available = {
    'PhysNet'.lower(): get_Model_PhysNet(),
    'PaiNN'.lower(): get_Model_PaiNN,
    }

def get_model_calculator(
    model_type: str,
    **kwargs,
) -> torch.nn.Module:
    """
    Model calculator selection

    Parameters
    ----------
    model_type: str
        Model calculator type, e.g. 'PhysNet'
    **kwargs: dict, optional
        Keyword arguments for model calculator

    Returns
    -------
    torch.nn.Module
        Calculator model object for property prediction
    """

    # Check input parameter
    if model_type is None:
        raise SyntaxError("No model type is defined by 'model_type'!")
    
    # Return requested calculator model
    if model_type.lower() in model_available:
        return model_available[model_type.lower()](
            model_type=model_type,
            **kwargs)
    else:
        raise ValueError(
            f"Calculator model type input '{model_type:s}' is not known!\n" +
            "Choose from:\n" + str(model_available.keys()))
