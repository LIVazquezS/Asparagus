
import os
import logging
from typing import Optional, List, Dict, Tuple, Union, Any

import numpy as np

import torch

from ... import utils

"""
Torch implementation of Grimme's D3 method (only Becke-Johnson damping is 
implemented) 

Grimme, Stefan, et al. "A consistent and accurate ab initio parametrization of 
density functional dispersion correction (DFT-D) for the 94 elements H-Pu." 
The Journal of Chemical Physics 132, 15 (2010): 154104.
"""

#======================================
# Dispersion Correction Coefficients
#======================================

# Relative filepath to package folder
package_directory = os.path.dirname(os.path.abspath(__file__))

# Tables with reference values
d3_c6ab = torch.from_numpy(
    np.load(os.path.join(package_directory, "tables_d3", "c6ab.npy")))
d3_r0ab = torch.from_numpy(
    np.load(os.path.join(package_directory, "tables_d3", "r0ab.npy")))
d3_rcov = torch.from_numpy(
    np.load(os.path.join(package_directory, "tables_d3", "rcov.npy")))
d3_r2r4 = torch.from_numpy(
    np.load(os.path.join(package_directory, "tables_d3", "r2r4.npy")))
d3_maxc = 5  # maximum number of coordination complexes

# Global dispersion correction parameters (default values for HF)
d3_s6 = torch.FloatTensor([1.0000])
d3_s8 = torch.FloatTensor([0.9171])
d3_a1 = torch.FloatTensor([0.3385])
d3_a2 = torch.FloatTensor([2.8830])
d3_k1 = torch.FloatTensor([16.000])
d3_k2 = torch.FloatTensor([4./3.])
d3_k3 = torch.FloatTensor([-4.000])


#======================================
# Dispersion Correction Functions
#======================================


def _smootherstep(
    r: torch.Tensor,
    cutoff: torch.Tensor
) -> torch.Tensor:
    '''
    Computes a smooth step from 1 to 0 in the width of 1 Bohr
    before the cutoff
    '''
    
    cuton = cutoff - 1
    x = (cutoff - r) / (cutoff - cuton)
    
    return torch.where(
        x < cuton,
        torch.ones_like(x),
        1.0 + ((-6.0*x + 15.0)*x - 10.0)*x**3)


def _ncoord(
    Zi: torch.Tensor, 
    Zj: torch.Tensor, 
    r: torch.Tensor, 
    idx_i: torch.Tensor, 
    cutoff: torch.Tensor = None, 
    k1: torch.Tensor = d3_k1, 
    rcov: torch.Tensor = d3_rcov, 
    device='cpu'
) -> torch.Tensor:
    '''
    Compute coordination numbers by adding an inverse damping function
    '''
    
    rcov = rcov.to(device)
    rco = torch.gather(rcov, 0, Zi.type(torch.int64)) + torch.gather(rcov, 0, Zj.type(torch.int64))
    rr = rco.type(r.dtype) / r
    damp = 1.0 / (1.0 + torch.exp(-k1 * (rr - 1.0)))
    if cutoff is not None:
        damp *= _smootherstep(r, cutoff)
    x = utils.segment_sum(damp, idx_i, device=device)
    return x


def _getc6(ZiZj, nci, ncj, c6ab=d3_c6ab, k3=d3_k3, device='cpu'):
    '''
    Interpolate c6
    '''
    # Gather the relevant entries from the table
    c6ab = torch.from_numpy(c6ab).to(device)
    c6ab_ = utils.gather_nd(c6ab, ZiZj).type(nci.dtype)  # check(?)
    # Calculate c6 coefficients
    c6mem = -1.0e99 * torch.ones_like(nci, device=device)
    r_save = 1.0e99 * torch.ones_like(nci, device=device)
    rsum = torch.zeros_like(nci, device=device)
    csum = torch.zeros_like(nci, device=device)
    for i in range(d3_maxc):
        for j in range(d3_maxc):
            cn0 = c6ab_[:, i, j, 0]
            cn1 = c6ab_[:, i, j, 1]
            cn2 = c6ab_[:, i, j, 2]
            r = (cn1 - nci) ** 2 + (cn2 - ncj) ** 2
            r_save = torch.where(r < r_save, r, r_save)
            c6mem = torch.where(r < r_save, cn0, c6mem)
            tmp1 = torch.exp(k3 * r)
            rsum += torch.where(cn0 > 0.0, tmp1, torch.zeros_like(tmp1, device=device))
            csum += torch.where(cn0 > 0.0, tmp1 * cn0, torch.zeros_like(tmp1, device=device))
    c6 = torch.where(rsum > 0.0, csum / rsum, c6mem)
    return c6


def edisp(Z, r, idx_i, idx_j, cutoff=None, r2=None, r6=None, r8=None, s6=d3_s6,
          s8=d3_s8, a1=d3_a1, a2=d3_a2, k1=d3_k1, k2=d3_k2, k3=d3_k3,
          c6ab=d3_c6ab, r0ab=d3_r0ab, rcov=d3_rcov, r2r4=d3_r2r4, device='cpu'):
    '''
    Compute d3 dispersion energy in Hartree
    r: distance in bohr!
    '''
    r2r4 = torch.from_numpy(r2r4).to(device)
    # Compute all necessary quantities
    Zi = torch.gather(Z, 0, idx_i)
    Zj = torch.gather(Z, 0, idx_j)
    ZiZj = torch.stack([Zi, Zj], axis=1)  # necessary for gathering
    # Coordination numbers
    nc = _ncoord(Zi, Zj, r, idx_i, cutoff=cutoff, rcov=rcov, device=device)
    nci = torch.gather(nc, 0, idx_i)
    ncj = torch.gather(nc, 0, idx_j)
    c6 = _getc6(ZiZj, nci, ncj, c6ab=c6ab, k3=k3, device=device)  # c6 coefficients
    c8 = 3 * c6 * torch.gather(r2r4, 0, Zi).type(c6.dtype) * torch.gather(r2r4, 0, Zj).type(c6.dtype)  # c8 coefficient

    # Compute all necessary powers of the distance
    if r2 is None:
        r2 = r**2
    if r6 is None:
        r6 = r2**3
    if r8 is None:
        r8 = r6*r2

    # Becke-Johnson damping, zero-damping introduces spurious repulsion
    # and is therefore not supported/implemented
    tmp = a1 * torch.sqrt(c8 / c6) + a2
    tmp2 = tmp ** 2
    tmp6 = tmp2 ** 3
    tmp8 = tmp6 * tmp2
    if cutoff is None:
        e6 = 1 / (r6 + tmp6)
        e8 = 1 / (r8 + tmp8)
    else:  # apply cutoff
        cut2 = cutoff ** 2
        cut6 = cut2 ** 3
        cut8 = cut6 * cut2
        cut6tmp6 = cut6 + tmp6
        cut8tmp8 = cut8 + tmp8
        e6 = 1 / (r6 + tmp6) - 1 / cut6tmp6 + 6 * cut6 / cut6tmp6 ** 2 * (r / cutoff - 1)
        e8 = 1 / (r8 + tmp8) - 1 / cut8tmp8 + 8 * cut8 / cut8tmp8 ** 2 * (r / cutoff - 1)
        e6 = torch.where(r < cutoff, e6, torch.zeros_like(e6))
        e8 = torch.where(r < cutoff, e8, torch.zeros_like(e8))
    e6 = -0.5 * s6 * c6 * e6
    e8 = -0.5 * s8 * c8 * e8
    z = utils.segment_sum(e6 + e8, idx_i, device=device)
    # idu = len(torch.unique(idx_i))
    # z = (e6+e8).new_zeros(idu).index_add(0,idx_i.type(torch.int64),e6+e8)
    return z


