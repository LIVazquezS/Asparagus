import time
import socket
import threading
import numpy as np

from typing import Optional, List, Dict, Tuple, Union, Any

import torch

from .. import utils

def header(
    config_file: str
) -> None:
    """
    Provide the Asparagus header. 

    Parameters
    ----------
    config_file: str
        Current configuration file path

    Returns
    -------
    str
        Asparagus header
    """

    current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    host = socket.gethostname()

    print(f"""
       '   _______                                                  ______                  _ _        
       '  (_______)                                                (____  \                | | |       
       '   _______  ___ ____  _____  ____ _____  ____ _   _  ___    ____)  )_   _ ____   __| | | _____ 
       '  |  ___  |/___)  _ \(____ |/ ___|____ |/ _  | | | |/___)  |  __  (| | | |  _ \ / _  | || ___ |
       '  | |   | |___ | |_| / ___ | |   / ___ ( (_| | |_| |___ |  | |__)  ) |_| | | | ( (_| | || ____|
       '  |_|   |_(___/|  __/\_____|_|   \_____|\___ |____/(___/   |______/|____/|_| |_|\____|\_)_____)
       '               |_|                     (_____|                            
       '
       '                        Authors: K. Toepfer and L.I. Vazquez-Salazar
       '                        Date: {current_time:s}
       '                        Running on: {host:s}
       '                        Details of this run are stored in: {config_file:s} 
       ' ---------------------------------------------------------------------------------------------     
       """)

def detach_tensor(x):
    """
    Detach a torch tensor from the computational graph

    Parameters
    ----------
    x: Any
        Input variable to detach

    Returns
    -------
    Any
        Detached input variable
    """
    if utils.in_cuda(x):
        x.cpu()
        x.detach().numpy()
    else:
        x.detach().numpy()
    return x

def flatten_array_like(x):
    for xi in x:
        if utils.is_string(xi):
            yield x
        else:
            try:
                yield from flatten_array_like(xi)
            except TypeError:
                yield xi


def segment_sum(
    data: torch.Tensor,
    segment_ids: torch.Tensor,
    device: Optional[str] = 'cpu',
    debug: Optional[bool] = False,
) -> torch.Tensor:
    """
    Adapted from : 
        https://gist.github.com/bbrighttaer/207dc03b178bbd0fef8d1c0c1390d4be
    Analogous to tf.segment_sum :
        https://www.tensorflow.org/api_docs/python/tf/math/segment_sum

    Parameters
    ----------
    data: torch.Tensor
        A pytorch tensor of the data for segmented summation.
    segment_ids: torch.Tensor, shape(N)
        A 1-D tensor containing the indices for the segmentation.

    Returns
    -------
    torch.Tensor
        A tensor of the same type as data containing the results of the
        segmented summation.
    """

    if debug:

        if not all(
            segment_i <= segment_j for segment_i, segment_j
            in zip(segment_ids[:-1], segment_ids[1:])
        ):

            raise AssertionError("Elements of 'segment_ids' must be sorted")

        if len(segment_ids.shape) != 1:
            raise AssertionError("'segment_ids' have to be a 1-D tensor")

        if data.shape[0] != segment_ids.shape[0]:
            raise AssertionError(
                "'data' and 'segment_ids'should be the same size at "
                + f"dimension 0 but are ({data.shape[0]:d}) and "
                + f"({segment_ids.shape[0]}).")

    num_segments = len(torch.unique(segment_ids))
    return unsorted_segment_sum(
        data, segment_ids, num_segments, device=data.device)

def unsorted_segment_sum(
    data: torch.Tensor,
    segment_ids: torch.Tensor,
    num_segments: int,
    device: Optional[str] = 'cpu',
    debug: Optional[bool] = False,
) -> torch.Tensor:
    """
    Computes the sum along segments of a tensor. Analogous to
    tf.unsorted_segment_sum.

    Parameters
    ----------
    data: torch.Tensor
        A tensor whose segments are to be summed.
    segment_ids: torch.Tensor, shape(N)
        The segment indices tensor.
    num_segments: int
        The number of segments.

    Returns
    -------
    torch.Tensor
        A tensor of same data type as the data argument.
    """

    if debug:

        assert all([i in data.shape for i in segment_ids.shape]), "'segment_ids.shape' should be a prefix of 'data.shape'!"

        if len(segment_ids.shape) == 1:
            s = torch.prod(torch.tensor(data.shape[1:])).long().to(device)
            segment_ids = segment_ids.repeat_interleave(s).view(
                segment_ids.shape[0], *data.shape[1:]).to(device)

        assert data.shape == segment_ids.shape, "'data.shape' and 'segment_ids.shape' should be equal!"

    else:

        s = torch.prod(torch.tensor(data.shape[1:])).long().to(device)
        segment_ids = segment_ids.repeat_interleave(s).view(
            segment_ids.shape[0], *data.shape[1:]).to(device)

    shape = [num_segments] + list(data.shape[1:])
    tensor = torch.zeros(
        *shape, dtype=torch.float64, device=device).scatter_add(
            0, segment_ids, data)

    return tensor


def softplus_inverse(x):
    """
    Numerically stable inverse of softplus transform
    .. math:: f(x) = x + \log(1 - \exp(x))

    Parameters
    ----------
    x: torch.Tensor
        A tensor of any shape.


    """
    return x + np.log(-np.expm1(-x))


def gather_nd(
    params,
    indices,
):
    """
    The input indices must be a 2d tensor in the form of [[a,b,..,c],...],
    which represents the location of the elements.
    This function comes from: https://discuss.pytorch.org/t/implement-tf-gather-nd-in-pytorch/37502/6

    Parameters
    ----------
    params: torch.Tensor
        A tensor of any shape.
    indices: torch.Tensor
        A 2d tensor in the form of [[a,b,..,c],...]
    """

    # Generate indices
    indices = indices.t().long()
    ndim = indices.size(0)
    idx = torch.zeros_like(indices[0]).long()
    m = 1

    for i in range(ndim)[::-1]:
        idx = idx + indices[i] * m
        m *= params.size(i)

    params = params.reshape((-1, *tuple(torch.tensor(params.size()[ndim:]))))
    return params[idx]


def printProgressBar(
    iteration,
    total,
    prefix='',
    suffix='',
    decimals=1,
    length=100,
    fill='#',
    printEnd="\r",
):
    """

    Call in a loop to create terminal progress bar

    Parameters
    ----------

    iteration (int) Required:
        current iteration (Int)
    total (int) Required:
        total iterations
    prefix (str) Optional:
        prefix string
    suffix (str) Optional:
        suffix string
    decimals Optional:
        positive number of decimals in percent complete
    length (Int) Optional:
        Character length of bar (Int)
    fill (str) Optional:
        bar fill character
    printEnd (str) Optional:
        end character (e.g. "/r", "/r/n") (Str)
    """

    percent = (
        ("{0:." + str(decimals) + "f}").format(
            100 * (iteration / float(total)))
        )
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)

    print("\r{0} |{1}| {2}% {3}".format(
        prefix, bar, percent, suffix), end=printEnd)

    # Print New Line on Complete
    if iteration == total:
        print()
