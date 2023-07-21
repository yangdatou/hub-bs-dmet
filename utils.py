import os, sys
from functools import reduce

import numpy, scipy
from scipy import linalg

import pyscf
from pyscf import gto
from pyscf import scf
from pyscf import fci

from pyscf.tools.dump_mat import dump_rec
print_matrix = lambda c, t=None, stdout=sys.stdout: ((print("\n" + t, file=stdout) if t is not None else 0), dump_rec(stdout, c))

class ElectronicStructureProblem(object):
    """
    The electronic structure problem in an orthogonal
    basis is defined by the following quantities:
        - h0: the constant term in the Hamiltonian, e.g. the nuclear repulsion
        - h1: the one-electron integrals
        - h2: the two-electron integrals, in the physicist's notation
        - norb: the number of orbitals
        - nelecs: the number of electrons
        - spin: the spin of the system, to label the shape of the integrals
    """

    h0 = None
    h1 = None
    h2 = None

    spin   = None
    norb   = None
    nelecs = None

    verbose   = 0
    stdout    = None
    conv_tol  = 1e-6
    max_cycle = 200

    def __init__(self):
        raise NotImplementedError

class RestrictedElectronicStructureProblem(ElectronicStructureProblem):
    spin = 1

    def __init__(self, norb: int, nelecs: tuple[int, int], h0: float = 0.0, h1: numpy.ndarray = None, h2: numpy.ndarray = None):
        if h1 is None:
            h1 = numpy.zeros((norb, norb))

        if h2 is None:
            h2 = numpy.zeros((norb, norb, norb, norb))

        # Check the number of electrons
        nelec_alpha, nelec_beta = nelecs
        assert nelec_alpha == nelec_beta

        # Check the shapes
        assert isinstance(h0, float)
        assert h1.shape == (norb, norb)
        assert h2.shape == (norb, norb, norb, norb)

        # Build the object
        self.norb   = norb
        self.nelecs = (nelec_alpha, nelec_beta)
        self.h0     = h0
        self.h1     = h1
        self.h2     = h2

class Solver(object):
    pass

class ElectronicStructureResult(object):
    pass
        
def solve_restricted_hartree_fock(prob_obj : ElectronicStructureProblem, dm0 : numpy.ndarray = None):
    """
    Solves the restricted Hartree-Fock problem for the given electronic structure problem.

    Args:
        prob_obj (ElectronicStructureProblem): The electronic structure problem to solve.
        dm0 (numpy.ndarray, optional): The initial density matrix. Defaults to None.

    Returns:
        RestrictedHartreeFockResult: The result of the calculation.
    """
    # TODO: can accept more than just RestrictedElectronicStructureProblem
    assert isinstance(prob_obj, RestrictedElectronicStructureProblem)

    # Extract necessary information from the ElectronicStructureProblem object
    norb = prob_obj.norb
    nelec_alpha, nelec_beta = prob_obj.nelecs
    spin      = nelec_alpha - nelec_beta
    nelectron = nelec_alpha + nelec_beta

    h0 = prob_obj.h0
    h1 = prob_obj.h1
    h2 = prob_obj.h2

    if dm0 is not None:
        dm0 = numpy.asarray(dm0)

        if dm0.ndim == 2:
            dm0_alph = dm0 * 0.5
            dm0_beta = dm0 * 0.5

        else:
            assert dm0.shape == (2, norb, norb)
            dm0_alph = dm0[0]
            dm0_beta = dm0[1]

        dm0 = dm0_alph + dm0_beta

    # Build the fake Mole object
    mol_obj = gto.Mole()
    mol_obj.verbose   = prob_obj.verbose
    mol_obj.stdout    = prob_obj.stdout
    mol_obj.spin      = spin
    mol_obj.nelectron = nelectron
    mol_obj.nao       = norb
    mol_obj.build()

    # Build the RHF object
    hf_obj = scf.RHF(mol_obj)
    hf_obj.verbose   = prob_obj.verbose
    hf_obj.max_cycle = prob_obj.max_cycle
    hf_obj.conv_tol  = prob_obj.conv_tol
    hf_obj.get_hcore = lambda *args: h1
    hf_obj.get_ovlp  = lambda *args: numpy.eye(norb)
    hf_obj._eri      = h2

    # Solve the RHF problem
    ene_hf = hf_obj.kernel(dm0=dm0) + h0

    # Save the results
    coeff_hf = hf_obj.mo_coeff
    occ_hf   = hf_obj.mo_occ

    class RestrictedHartreeFock(ElectronicStructureResult):
        """
        The result of a restricted Hartree-Fock calculation.
        """
        hf_obj    = None
        _ene_hf   = None
        _coeff_hf = None
        _occ_hf   = None

        def get_ene(self):
            """
            Returns the total energy of the calculation.
            """
            return self._ene_hf

        def get_r_rdm1(self):
            """
            Returns the restricted one-particle reduced density matrix.
            """
            r_rdm1 = self.hf_obj.make_rdm1(self._coeff_hf, self._occ_hf)
            return r_rdm1

        def get_u_rdm1(self):
            """
            Returns the unrestricted one-particle reduced density matrix.
            """
            r_rdm1 = self.hf_obj.make_rdm1(self._coeff_hf, self._occ_hf)
            u_rdm1 = (r_rdm1 * 0.5, r_rdm1 * 0.5)
            u_rdm1 = numpy.asarray(u_rdm1)
            return u_rdm1

    res = RestrictedHartreeFock()
    res.hf_obj    = hf_obj
    res._ene_hf   = ene_hf
    res._coeff_hf = coeff_hf
    res._occ_hf   = occ_hf
    return res

def solve_unrestricted_hartree_fock(prob_obj : ElectronicStructureProblem, dm0 : numpy.ndarray = None):
    """
    Solves the unrestricted Hartree-Fock problem for the given electronic structure problem.

    Args:
        prob_obj (ElectronicStructureProblem): The electronic structure problem to solve.
        dm0 (numpy.ndarray, optional): The initial density matrix. Defaults to None.

    Returns:
        UnrestrictedHartreeFockResult: The result of the calculation.
    """
    # TODO: can accept more than just RestrictedElectronicStructureProblem
    assert isinstance(prob_obj, RestrictedElectronicStructureProblem)

    # Extract necessary information from the ElectronicStructureProblem object
    norb = prob_obj.norb
    nelec_alpha, nelec_beta = prob_obj.nelecs
    spin      = nelec_alpha - nelec_beta
    nelectron = nelec_alpha + nelec_beta

    h0 = prob_obj.h0
    h1 = prob_obj.h1
    h2 = prob_obj.h2

    # Build the initial density matrix
    if dm0 is not None:
        dm0 = numpy.asarray(dm0)

        if dm0.ndim == 2:
            dm0_alph = dm0 * 0.5
            dm0_beta = dm0 * 0.5

        else:
            assert dm0.shape == (2, norb, norb)
            dm0_alph = dm0[0]
            dm0_beta = dm0[1]

        dm0 = (dm0_alph, dm0_beta)

    # Build the fake Mole object
    mol_obj = gto.Mole()
    mol_obj.verbose   = prob_obj.verbose
    mol_obj.stdout    = prob_obj.stdout
    mol_obj.spin      = spin
    mol_obj.nelectron = nelectron
    mol_obj.nao       = norb
    mol_obj.build()

    # Build the UHF object
    hf_obj = scf.UHF(mol_obj)
    hf_obj.verbose   = prob_obj.verbose
    hf_obj.max_cycle = prob_obj.max_cycle
    hf_obj.conv_tol  = prob_obj.conv_tol
    hf_obj.get_hcore = lambda *args: h1
    hf_obj.get_ovlp  = lambda *args: numpy.eye(norb)
    hf_obj._eri      = h2

    # Solve the UHF problem
    ene_hf = hf_obj.kernel(dm0=dm0) + h0

    # Save the results
    coeff_hf = hf_obj.mo_coeff
    occ_hf   = hf_obj.mo_occ

    class UnrestrictedHartreeFock(ElectronicStructureResult):
        """
        The result of an unrestricted Hartree-Fock calculation.
        """
        hf_obj    = None
        _ene_hf   = None
        _coeff_hf = None
        _occ_hf   = None

        def get_ene(self):
            """
            Returns the total energy of the calculation.
            """
            return self._ene_hf

        def get_r_rdm1(self):
            """
            Returns the restricted one-particle reduced density matrix.
            """
            u_rdm1 = self.hf_obj.make_rdm1(self._coeff_hf, self._occ_hf)
            r_rdm1 = u_rdm1[0] + u_rdm1[1]
            return r_rdm1

        def get_u_rdm1(self):
            """
            Returns the unrestricted one-particle reduced density matrix.
            """
            u_rdm1 = self.hf_obj.make_rdm1(self._coeff_hf, self._occ_hf)
            u_rdm1 = numpy.asarray(u_rdm1)
            return u_rdm1

    res = UnrestrictedHartreeFock()
    res.hf_obj    = hf_obj
    res._ene_hf   = ene_hf
    res._coeff_hf = coeff_hf
    res._occ_hf   = occ_hf
    return res

def solve_direct_spin1_full_configuration_interaction(prob_obj : ElectronicStructureProblem):
    """
    Solves the full configuration interaction problem for the given electronic structure problem.

    Args:
        prob_obj (ElectronicStructureProblem): The electronic structure problem to solve.

    Returns:
        FullConfigurationInteractionResult: The result of the calculation.
    """
    # TODO: can accept more than just RestrictedElectronicStructureProblem
    assert isinstance(prob_obj, RestrictedElectronicStructureProblem)

    # Extract necessary information from the ElectronicStructureProblem object
    norb = prob_obj.norb
    nelec_alpha, nelec_beta = prob_obj.nelecs
    spin      = nelec_alpha - nelec_beta
    nelectron = nelec_alpha + nelec_beta

    h0 = prob_obj.h0
    h1 = prob_obj.h1
    h2 = prob_obj.h2

    # Create an instance of the FCI solver and set its parameters
    fci_obj = fci.direct_spin1.FCI()
    fci_obj.verbose   = prob_obj.verbose
    fci_obj.spin      = spin
    fci_obj.max_cycle = prob_obj.max_cycle
    fci_obj.conv_tol  = prob_obj.conv_tol
    ene_fci, vec_fci  = fci_obj.kernel(h1, h2, norb, nelectron, ci0=None, ecore=h0)

    class DirectSpin1FullConfigurationInteraction(ElectronicStructureResult):
        """
        The result of a full configuration interaction calculation.
        """

        fci_obj  = None
        _ene_fci = None
        _vec_fci = None

        def get_ene(self):
            """
            Returns the total energy of the calculation.
            """
            return self.fci_obj.e_tot

        def get_r_rdm1(self):
            """
            Returns the restricted one-particle reduced density matrix.
            """
            rdm1_alph, rdm1_beta = self.fci_obj.make_rdm1s(vec_fci, norb, nelectron)
            r_rdm1 = rdm1_alph + rdm1_beta
            u_rdm1 = numpy.asarray((rdm1_alph, rdm1_beta))
            return r_rdm1

        def get_u_rdm1(self):
            """
            Returns the unrestricted one-particle reduced density matrix.
            """
            rdm1_alph, rdm1_beta = self.fci_obj.make_rdm1s(vec_fci, norb, nelectron)
            r_rdm1 = rdm1_alph + rdm1_beta
            u_rdm1 = numpy.asarray((rdm1_alph, rdm1_beta))
            return u_rdm1

    res = DirectSpin1FullConfigurationInteraction()
    res.fci_obj = fci_obj
    res.ene_fci = ene_fci
    res.vec_fci = vec_fci
    return res