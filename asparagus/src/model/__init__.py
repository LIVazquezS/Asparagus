'''

This module contains the clases that define the model of the neural network.

The model is composed of three parts: input, graph and output. All are mixed in the calculator.
In the future, a calculator should be defined for each model.

'''

from .input import (
    get_input_model, Input_PhysNetRBF, Input_PhysNetRBF_original
)

from .graph import (
    get_graph_model, Graph_PhysNetMP
)

from .output import (
    get_output_model, Output_PhysNet
)

from .calculator import (
    get_calculator
)

from .physnet import (
    Calculator_PhysNet
)
