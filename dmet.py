import os, sys
from functools import reduce

import numpy, scipy
from scipy import linalg

import pyscf
from pyscf import gto
from pyscf import scf
from pyscf import fci

from .utils import print_matrix
from .utils import RestrictedElectronicStructureProblem
from .utils import RestrictedElectronicStructureResult

def solve_fci(prob_obj, conv_tol = 1e-10):
    """
    Solve the Hamiltonian using FCI.

    Args:
        ham_obj: a Hamiltonian object
        conv_tol: convergence tolerance

    Returns:
        A tuple of the FCI energy and the FCI object
    """

    # Dump the information
    nsite  = hub_obj.nsite
    nelecs = hub_obj.nelecs
    nelec_alpha, nelec_beta = nelecs
    assert nelec_alpha == nelec_beta

    h0 = hub_obj.get_h0()
    h1 = hub_obj.get_h1()
    h2 = hub_obj.get_h2()

    # Build the FCI object
    fci_obj = pyscf.fci.direct_spin1.FCI()
    fci_obj.max_cycle = 1000
    fci_obj.conv_tol = conv_tol
    fci_obj.kernel(h1, h2, nsite, (nelec_alpha, nelec_beta))
    return fci_obj.e_tot + h0, fci_obj

if __name__ == "__main__":
    nsite = 4
    nelecs = (2, 2)
    hub_u = 1.0

    hub_obj = build_hub_model(nsite, nelecs, hub_u)
    solve_hub_fci(hub_obj, conv_tol = 1e-10)
