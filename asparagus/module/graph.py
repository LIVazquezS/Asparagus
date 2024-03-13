
import torch

#======================================
#  Graph Model Provision
#======================================

def get_Graph_PhysNet():
    from .mdu_physnet import Graph_PhysNet
    return Graph_PhysNet

def get_Graph_PaiNN():
    from .mdu_painn import Graph_PaiNN
    return Graph_PaiNN


#======================================
#  Graph Model Assignment
#======================================

graph_module_available = {
    'PhysNet'.lower(): get_Graph_PhysNet(),
    'PaiNN'.lower(): Graph_PaiNN,
    }

def get_graph_module(
    graph_type: str,
    **kwargs,
) -> torch.nn.Module:
    """
    Graph module selection

    Parameters
    ----------
    graph_type: str
        Graph model representation of the atomistic structural information
        e.g. 'PhysNet'.
    **kwargs: dict, optional
        Keyword arguments for graph module initialization

    Returns
    -------
    torch.nn.Module
        Graph model object to encode atomistic structural information
    """

    # Check input parameter
    if graph_type is None:
        raise SyntaxError("No graph module type is defined by 'graph_type'!")
    
    # Return requested graph module
    if graph_type.lower() in nput_model_available:
        return graph_model_available[graph_type.lower()](
            graph_type=graph_type,
            **kwargs)
    else:
        raise ValueError(
            f"Graph model type input '{graph_type:s}' is not known!\n" +
            "Choose from:\n" + str(graph_model_available.keys()))
