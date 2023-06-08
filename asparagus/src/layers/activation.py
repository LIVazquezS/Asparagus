# TODO: First attempt to implement loss functions, maybe this can be converted to a class
import torch

from .. import utils

__all__ = [
    'get_activation_fn', 'swish', 'softplus', 'shifted_softplus', 
    'scaled_shifted_softplus', 'self_normalizing_shifted_softplus', 
    'smooth_ELU', 'self_normalizing_smooth_ELU', 'self_normalizing_asinh', 
    'self_normalizing_tanh', 'linear']

#======================================
# Activation functions
#======================================

# Google's swish function
@torch.jit.script
def swish(x: torch.Tensor):
    return x * torch.nn.functional.sigmoid(x)

# First time softplus was used as activation function: "Incorporating
# Second-Order Functional Knowledge for Better Option Pricing"
# (https://papers.nips.cc/paper/1920-incorporating-second-order-functional-knowledge-for-better-option-pricing.pdf)
@torch.jit.script
def _softplus(x: torch.Tensor):
    return torch.log1p(torch.exp(x))


@torch.jit.script
def softplus(x):
    # This definition is for numerical stability for x larger than 15 (single
    # precision) or x larger than 34 (double precision), there is no numerical
    # difference anymore between the softplus and a linear function
    return torch.where(x < 15.0, _softplus(torch.where(x < 15.0, x, torch.zeros_like(x))), x)


@torch.jit.script
def shifted_softplus(x: torch.Tensor):
    # return softplus(x) - np.log(2.0)
    return torch.nn.functional.softplus(x) - torch.log(torch.tensor(2.0))


# This ensures that the function is close to linear near the origin!
@torch.jit.script
def scaled_shifted_softplus(x: torch.Tensor):
    return 2 * shifted_softplus(x)


# Is not really self-normalizing sadly...
@torch.jit.script
def self_normalizing_shifted_softplus(x: torch.Tensor):
    return 1.875596256135042 * shifted_softplus(x)


# General: ln((exp(alpha)-1)*exp(x)+1)-alpha
@torch.jit.script
def smooth_ELU(x: torch.Tensor):
    # (e-1) = 1.718281828459045
    return torch.log1p(1.718281828459045 * torch.exp(x)) - 1.0  


@torch.jit.script
def self_normalizing_smooth_ELU(x: torch.Tensor):
    return 1.574030675714671 * smooth_ELU(x)


@torch.jit.script
def self_normalizing_asinh(x: torch.Tensor):
    return 1.256734802399369 * torch.asinh(x)


@torch.jit.script
def self_normalizing_tanh(x: torch.Tensor):
    return 1.592537419722831 * torch.tanh(x)


@torch.jit.script
def linear(x: torch.Tensor):
    return x


#======================================
# Function Assignment  
#======================================

functions_avaiable = {
    'swish'.lower(): swish, 
    'softplus'.lower(): softplus, 
    'shifted_softplus'.lower(): shifted_softplus,
    'scaled_shifted_softplus'.lower(): scaled_shifted_softplus,
    'self_normalizing_shifted_softplus.lower()':
        self_normalizing_shifted_softplus,
    'smooth_ELU'.lower(): smooth_ELU, 
    'self_normalizing_smooth_ELU'.lower(): self_normalizing_smooth_ELU,
    'self_normalizing_asinh'.lower(): self_normalizing_asinh,
    'self_normalizing_tanh'.lower(): self_normalizing_tanh, 
    'linear'.lower(): linear,
    }

def get_activation_fn(name):
    """
    Get activation function by defined name.
    
    Parameters
    ----------
        
        name: (str, object)
            If name is a str than it checks for the corresponding activation
            function and return the function object.
            The input will be given if it is already a function object.
            
    Returns
    -------
        object
            Activation function
    """
    
    if name is None:
        
        return functions_avaiable['linear']
    
    elif utils.is_object(name):
        
        return name
    
    elif utils.is_string(name):
        
        if name.lower() in [key.lower() for key in functions_avaiable.keys()]:
            return functions_avaiable[name.lower()]
        else:
            raise ValueError(
                f"Activation function input '{name}' is not valid!" +
                f"Choose from:\n" +
                str(functions_avaiable.keys()))
    else:
        
        raise ValueError(
            f"Activation function input of type '{type(name)}' " +
            f"is not valid! Input 'name' has to be an object or 'str' from;\n" +
            str(functions_avaiable.keys()))
