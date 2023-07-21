import os, sys, typing
from typing import List, Tuple, Callable
sys.path.append(".")

from functools import reduce

import numpy, scipy
from scipy import linalg
from scipy.optimize import basinhopping

import jax
import jax.numpy as jnumpy
from jax.scipy.linalg import eigh
from jax.scipy.optimize import minimize

def gen_loss_function_r(h1e: numpy.ndarray, rdm1_tag: numpy.ndarray, fit_inds: numpy.ndarray,
                             nelecs: Tuple[int, int], loss_func_type: int):
    """
    Generate the loss function for the fitting problem.

    The meanfield is assumed to be restricted, meaning that the state is the eigenstate of Sz and S2,
    where the alpha and beta electrons are doubly occupied in the same set of orbitals.

    Args:
        h1e (numpy.ndarray, shape=(nsite, nsite)):
            The one-body Hamiltonian, assumed to be identical for alpha and beta electrons.
        rdm1_tag (numpy.ndarray, shape=(nsite, nsite)):
            The target total density matrix.
        fit_inds (numpy.ndarray, shape=(nfrag, nimp)):
            List of indices for the fitting problem. Each element of the list is a list of indices for a fragment.
            The size of each fragment is assumed to be the same.
        nelecs (Tuple[int, int]):
            The number of alpha and beta electrons.
        loss_func_type (int):
            The type of the loss function.
            - 1: The loss function is the norm of the difference between the impurity block of the target
                 and the fitted density matrix.
            - 2: The loss function is the norm of the difference between the target and the fitted density matrix.

    Returns:
        f (LossFunctionMixin):
            An instance of the LossFunctionMixin class containing:
            - func: The loss function that takes the fitting parameters as input and returns the value of the loss function.
            - grad: The gradient of the loss function.
            - _fill_correlation_potential: A helper function to fill the correlation potential matrix. (for debugging purpose)
            - _get_density_matrix: A helper function to get the density matrix. (for debugging purpose)
    """
    # Convert fit_inds to a JAX array and extract dimensions
    fit_inds = jnumpy.asarray(fit_inds)
    nfrag    = fit_inds.shape[0]
    nimp     = fit_inds.shape[1]
    nsite    = nimp * nfrag

    ind_triu = jnumpy.triu_indices(nimp)
    num_triu = ind_triu[0].shape[0]
    num_parm = num_triu

    # Ensure that dimensions of f1e and rdm1_tag match expected dimensions
    assert h1e.shape      == (nsite, nsite)
    assert rdm1_tag.shape == (nsite, nsite)

    # Build the one-body Hamiltonian for the problem
    f1e = jnumpy.array(h1e)

    def fill_correlation_potential(x):
        # Generate indices for the upper triangle

        # Check that x has expected dimensions
        assert x.shape == (num_triu,)

        # Initialize potential with zeros
        v = jnumpy.zeros((nsite, nsite))

        # Set values in the upper triangle of the correlation potential matrix for each fragment
        for fit_ind in fit_inds:
            ind_fit_0 = fit_ind[ind_triu[0]]
            ind_fit_1 = fit_ind[ind_triu[1]]
            v = v.at[ind_fit_0, ind_fit_1].set(x)

        # Symmetrize the correlation potential matrix
        v_sym = v + jnumpy.transpose(v) - jnumpy.diag(jnumpy.diag(v))
        return v_sym

    def get_density_matrix(f1e):
        # Ensure that the number of alpha and beta electrons are the same
        assert nelecs[0] == nelecs[1]

        # Solve the restricted Hartree-Fock problem to get MO energies and coefficients
        mo_ene_alph, mo_coeff_alph = eigh(f1e)
        mo_ene_beta, mo_coeff_beta = mo_ene_alph, mo_coeff_alph

        # Select indices for occupied orbitals
        occ_idx_alph = jnumpy.argsort(mo_ene_alph)[:nelecs[0]]
        occ_idx_beta = jnumpy.argsort(mo_ene_beta)[:nelecs[1]]

        # Obtain occupied MO coefficients
        coeff_occ_alph = mo_coeff_alph[:, occ_idx_alph]
        coeff_occ_beta = mo_coeff_beta[:, occ_idx_beta]

        # Calculate alpha and beta RDMs
        rdm1_fit_alph  = jnumpy.dot(coeff_occ_alph, coeff_occ_alph.T)
        rdm1_fit_beta  = jnumpy.dot(coeff_occ_beta, coeff_occ_beta.T)

        return rdm1_fit_alph, rdm1_fit_beta

    # Define the loss function based on the specified type
    if loss_func_type == 1:
        def func(x):
            # Fill the correlation potential and calculate f1e
            f1e_fit  = f1e + fill_correlation_potential(x)
            # Obtain the total RDM
            rdm1_fit = (lambda rdm1: rdm1[0] + rdm1[1])(get_density_matrix(f1e_fit))
            # Calculate the difference between the target and fitted RDMs
            rdm1_err = rdm1_tag - rdm1_fit
            # The loss function is the norm of the RDM difference
            err = jnumpy.linalg.norm(rdm1_err)
            return err

    elif loss_func_type == 2:
        def func(x):
            # Fill the correlation potential and calculate f1e
            f1e_fit  = f1e + fill_correlation_potential(x)
            # Obtain the total RDM
            rdm1_fit = (lambda rdm1: rdm1[0] + rdm1[1])(get_density_matrix(f1e_fit))
            # Calculate the difference between the target and fitted RDMs
            rdm1_err = rdm1_tag - rdm1_fit
            # The loss function is the sum of the norms of the RDM differences for each fragment
            err = sum([jnumpy.linalg.norm(rdm1_err[fit_ind][:, fit_ind]) for fit_ind in fit_inds])
            return err

    elif loss_func_type == 3:
        def func(x):
            # Fill the correlation potential and calculate f1e
            f1e_fit  = f1e + fill_correlation_potential(x)
            # Obtain the total RDM
            rdm1_fit = (lambda rdm1: rdm1[0] + rdm1[1])(get_density_matrix(f1e_fit))
            # Calculate the difference between the target and fitted RDMs
            rdm1_err = rdm1_tag - rdm1_fit
            # The loss function is the sum of the norms of the RDM differences for each fragment
            err  = jnumpy.linalg.norm(rdm1_err)
            err += sum([jnumpy.linalg.norm(rdm1_err[fit_ind][:, fit_ind]) for fit_ind in fit_inds])
            return err

    else:
        raise ValueError("Invalid loss function type.")

    # Define a class with the loss function and its gradient, as well as the heler functions
    class _LossFunctionMixin:
        pass

    f = _LossFunctionMixin()

    # Main results
    f.func = (lambda x: numpy.array(func(x)))
    f.grad = (lambda x: numpy.array(jax.grad(func)(x)))
    f.hess = (lambda x: numpy.array(jax.hessian(func)(x)))

    # Helper functions
    f._fill_correlation_potential = fill_correlation_potential
    f._get_density_matrix         = get_density_matrix

    # Other attributes
    f._f1e      = f1e
    f._ind_triu = ind_triu
    f._num_triu = num_triu
    f._num_parm = num_parm

    # Return an instance of the class
    return f

def gen_loss_function_u(h1e: numpy.ndarray, rdm1_tag: numpy.ndarray, fit_inds: numpy.ndarray,
                             nelecs: Tuple[int, int], loss_func_type: int):
    """
    Generate the loss function for the fitting problem.

    The meanfield is assumed to be unrestricted, meaning that the state is the eigenstate of Sz but not S2,
    where the alpha and beta electrons are singly occupied in (possibly) different orbitals.

    Args:
        h1e (numpy.ndarray, shape=(nsite, nsite)):
            The one-body Hamiltonian, assumed to be identical for alpha and beta electrons.
        rdm1_tag (numpy.ndarray, shape=(nsite, nsite)):
            The target total density matrix.
        fit_inds (numpy.ndarray, shape=(nfrag, nimp)):
            List of indices for the fitting problem. Each element of the list is a list of indices for a fragment.
            The size of each fragment is assumed to be the same.
        nelecs (Tuple[int, int]):
            The number of alpha and beta electrons.
        loss_func_type (int):
            The type of the loss function.
            - 1: The loss function is the norm of the difference between the impurity block of the target
                 and the fitted density matrix.
            - 2: The loss function is the norm of the difference between the target and the fitted density matrix.

    Returns:
        f (LossFunctionMixin):
            An instance of the LossFunctionMixin class containing:
            - func: The loss function that takes the fitting parameters as input and returns the value of the loss function.
            - grad: The gradient of the loss function.
            - _fill_correlation_potential: A helper function to fill the correlation potential matrix. (for debugging purpose)
            - _get_density_matrix: A helper function to get the density matrix. (for debugging purpose)
    """
    # Convert fit_inds to a JAX array and extract dimensions
    fit_inds = jnumpy.asarray(fit_inds)
    nfrag    = fit_inds.shape[0]
    nimp     = fit_inds.shape[1]
    nsite    = nimp * nfrag

    ind_triu = jnumpy.triu_indices(nimp)
    num_triu = ind_triu[0].shape[0]
    num_parm = num_triu * 2

    # Ensure that dimensions of f1e and rdm1_tag match expected dimensions
    assert h1e.shape      == (nsite, nsite)
    assert rdm1_tag.shape == (nsite, nsite)

    # Build the one-body Hamiltonian for the problem
    f1e = jnumpy.array((h1e, h1e))

    def fill_correlation_potential(x):
        # Generate indices for the upper triangle

        # Check that x has expected dimensions
        assert x.shape == (num_parm,)

        # Initialize potential with zeros
        va = jnumpy.zeros((nsite, nsite))
        vb = jnumpy.zeros((nsite, nsite))

        # Set values in the upper triangle of the correlation potential matrix for each fragment
        for fit_ind in fit_inds:
            ind_fit_0 = fit_ind[ind_triu[0]]
            ind_fit_1 = fit_ind[ind_triu[1]]

            va = va.at[ind_fit_0, ind_fit_1].set(x[:num_triu])
            vb = vb.at[ind_fit_0, ind_fit_1].set(x[num_triu:])

        # Symmetrize the correlation potential matrix
        va_sym = va + jnumpy.transpose(va) - jnumpy.diag(jnumpy.diag(va))
        vb_sym = vb + jnumpy.transpose(vb) - jnumpy.diag(jnumpy.diag(vb))
        return jnumpy.array((va_sym, vb_sym))

    def get_density_matrix(f1e):
        # Ensure that the number of alpha and beta electrons are the same
        assert nelecs[0] == nelecs[1]

        # Solve the restricted Hartree-Fock problem to get MO energies and coefficients
        mo_ene_alph, mo_coeff_alph = eigh(f1e[0])
        mo_ene_beta, mo_coeff_beta = eigh(f1e[1])

        # Select indices for occupied orbitals
        occ_idx_alph = jnumpy.argsort(mo_ene_alph)[:nelecs[0]]
        occ_idx_beta = jnumpy.argsort(mo_ene_beta)[:nelecs[1]]

        # Obtain occupied MO coefficients
        coeff_occ_alph = mo_coeff_alph[:, occ_idx_alph]
        coeff_occ_beta = mo_coeff_beta[:, occ_idx_beta]

        # Calculate alpha and beta RDMs
        rdm1_fit_alph  = jnumpy.dot(coeff_occ_alph, coeff_occ_alph.T)
        rdm1_fit_beta  = jnumpy.dot(coeff_occ_beta, coeff_occ_beta.T)

        return rdm1_fit_alph, rdm1_fit_beta

    # Define the loss function based on the specified type
    if loss_func_type == 1:
        def func(x):
            # Fill the correlation potential and calculate f1e
            f1e_fit  = f1e + fill_correlation_potential(x)
            # Obtain the total RDM
            rdm1_fit = (lambda rdm1: rdm1[0] + rdm1[1])(get_density_matrix(f1e_fit))
            # Calculate the difference between the target and fitted RDMs
            rdm1_err = rdm1_tag - rdm1_fit
            # The loss function is the norm of the RDM difference
            err = jnumpy.linalg.norm(rdm1_err)
            return err

    elif loss_func_type == 2:
        def func(x):
            # Fill the correlation potential and calculate f1e
            f1e_fit  = f1e + fill_correlation_potential(x)
            # Obtain the total RDM
            rdm1_fit = (lambda rdm1: rdm1[0] + rdm1[1])(get_density_matrix(f1e_fit))
            # Calculate the difference between the target and fitted RDMs
            rdm1_err = rdm1_tag - rdm1_fit
            # The loss function is the sum of the norms of the RDM differences for each fragment
            err = sum([jnumpy.linalg.norm(rdm1_err[fit_ind][:, fit_ind]) for fit_ind in fit_inds])
            return err

    elif loss_func_type == 3:
        def func(x):
            # Fill the correlation potential and calculate f1e
            f1e_fit  = f1e + fill_correlation_potential(x)
            # Obtain the total RDM
            rdm1_fit = (lambda rdm1: rdm1[0] + rdm1[1])(get_density_matrix(f1e_fit))
            # Calculate the difference between the target and fitted RDMs
            rdm1_err = rdm1_tag - rdm1_fit
            # The loss function is the sum of the norms of the RDM differences for each fragment
            err  = jnumpy.linalg.norm(rdm1_err)
            err += sum([jnumpy.linalg.norm(rdm1_err[fit_ind][:, fit_ind]) for fit_ind in fit_inds])
            return err
    else:
        raise ValueError("Invalid loss function type.")

    # Define a class with the loss function and its gradient, as well as the helper functions
    class _LossFunctionMixin:
        pass

    f = _LossFunctionMixin()

    # Main results
    f.func = (lambda x: numpy.array(func(x)))
    f.grad = (lambda x: numpy.array(jax.grad(func)(x)))
    f.hess = (lambda x: numpy.array(jax.hessian(func)(x)))

    # Helper functions
    f._fill_correlation_potential = fill_correlation_potential
    f._get_density_matrix         = get_density_matrix

    # Other attributes
    f._f1e      = f1e
    f._ind_triu = ind_triu
    f._num_triu = num_triu
    f._num_parm = num_parm

    # Return an instance of the class
    return f

def gen_loss_function_g(h1e: numpy.ndarray, rdm1_tag: numpy.ndarray, fit_inds: numpy.ndarray,
                                nelecs: Tuple[int, int], loss_func_type: int):
    """
    Generate the loss function for the fitting problem.

    The meanfield is assumed to be general, meaning that the state is not the eigenstate of Sz or S2,
    where electrons are singly occupied in orbitals without spin labels.

    Args:
        h1e (numpy.ndarray, shape=(nsite, nsite)):
            The one-body Hamiltonian, assumed to be identical for alpha and beta electrons.
        rdm1_tag (numpy.ndarray, shape=(nsite, nsite)):
            The target total density matrix.
        fit_inds (numpy.ndarray, shape=(nfrag, nimp)):
            List of indices for the fitting problem. Each element of the list is a list of indices for a fragment.
            The size of each fragment is assumed to be the same.
        nelecs (Tuple[int, int]):
            The number of alpha and beta electrons. The results may be different from the arguments
            as the generalized Hartree-Fock problem will mix alpha and beta electrons.
        loss_func_type (int):
            The type of the loss function.
            - 1: The loss function is the norm of the difference between the impurity block of the target
                 and the fitted density matrix.
            - 2: The loss function is the norm of the difference between the target and the fitted density matrix.

    Returns:
        f (LossFunctionMixin):
            An instance of the LossFunctionMixin class containing:
            - func: The loss function that takes the fitting parameters as input and returns the value of the loss function.
            - grad: The gradient of the loss function.
            - _fill_correlation_potential: A helper function to fill the correlation potential matrix. (for debugging purpose)
            - _get_density_matrix: A helper function to get the density matrix. (for debugging purpose)
    """

    # Convert fit_inds to a JAX array and extract dimensions
    fit_inds = jnumpy.asarray(fit_inds)
    nfrag    = fit_inds.shape[0]
    nimp     = fit_inds.shape[1]
    nsite    = nimp * nfrag

    ind_triu = jnumpy.triu_indices(nimp)
    num_triu = ind_triu[0].shape[0]
    num_parm = num_triu * 4

    # Ensure that dimensions of f1e and rdm1_tag match expected dimensions
    assert h1e.shape      == (nsite, nsite)
    assert rdm1_tag.shape == (nsite, nsite)

    # Build the one-body Hamiltonian for the problem
    f1e = jnumpy.block([[h1e, jnumpy.zeros((nsite, nsite))], [jnumpy.zeros((nsite, nsite)), h1e]])

    def fill_correlation_potential(x):
        # Check that x has expected dimensions
        assert x.shape == (num_parm,)

        # Initialize potential with zeros
        vaa = jnumpy.zeros((nsite, nsite))
        vba = jnumpy.zeros((nsite, nsite))
        vbb = jnumpy.zeros((nsite, nsite))
        vab = jnumpy.zeros((nsite, nsite))

        # Set values in the upper triangle of the correlation potential matrix for each fragment
        for fit_ind in fit_inds:
            ind_fit_0 = fit_ind[ind_triu[0]]
            ind_fit_1 = fit_ind[ind_triu[1]]

            vaa = vaa.at[ind_fit_0, ind_fit_1].set(x[:num_triu])
            vba = vba.at[ind_fit_0, ind_fit_1].set(x[num_triu:2*num_triu])
            vbb = vbb.at[ind_fit_0, ind_fit_1].set(x[2*num_triu:3*num_triu])
            vab = vab.at[ind_fit_0, ind_fit_1].set(x[3*num_triu:])

        # Symmetrize the correlation potential matrix
        vaa_sym = vaa + jnumpy.transpose(vaa) - jnumpy.diag(jnumpy.diag(vaa))
        vba_sym = vba + jnumpy.transpose(vba) - jnumpy.diag(jnumpy.diag(vba))
        vbb_sym = vbb + jnumpy.transpose(vbb) - jnumpy.diag(jnumpy.diag(vbb))
        vab_sym = vab + jnumpy.transpose(vab) - jnumpy.diag(jnumpy.diag(vab))
        return jnumpy.array(((vaa_sym, vba_sym), (vab_sym, vbb_sym)))

    def get_density_matrix(f1e):
        # Solve the generalized Hartree-Fock problem to get MO energies and coefficients
        mo_ene, mo_coeff = eigh(f1e)

        # Select indices for occupied orbitals
        occ_idx = jnumpy.argsort(mo_ene)[:sum(nelecs)]

        # Obtain occupied MO coefficients
        coeff_occ = mo_coeff[:, occ_idx]

        # Calculate alpha and beta RDMs
        rdm1_fit_g = jnumpy.dot(coeff_occ, coeff_occ.T)
        rdm1_fit_alph = rdm1_fit_g[:nsite, :nsite]
        rdm1_fit_beta = rdm1_fit_g[nsite:, nsite:]

        return rdm1_fit_alph, rdm1_fit_beta

    # Define the loss function based on the specified type
    if loss_func_type == 1:
        def func(x):
            # Fill the correlation potential and calculate f1e
            f1e_fit  = f1e + fill_correlation_potential(x)
            # Obtain the total RDM
            rdm1_fit = (lambda rdm1: rdm1[0] + rdm1[1])(get_density_matrix(f1e_fit))
            # Calculate the difference between the target and fitted RDMs
            rdm1_err = rdm1_tag - rdm1_fit
            # The loss function is the norm of the RDM difference
            err = jnumpy.linalg.norm(rdm1_err)
            return err

    elif loss_func_type == 2:
        def func(x):
            # Fill the correlation potential and calculate f1e
            f1e_fit  = f1e + fill_correlation_potential(x)
            # Obtain the total RDM
            rdm1_fit = (lambda rdm1: rdm1[0] + rdm1[1])(get_density_matrix(f1e_fit))
            # Calculate the difference between the target and fitted RDMs
            rdm1_err = rdm1_tag - rdm1_fit
            # The loss function is the sum of the norms of the RDM differences for each fragment
            err = sum([jnumpy.linalg.norm(rdm1_err[fit_ind][:, fit_ind]) for fit_ind in fit_inds])
            return err

    elif loss_func_type == 3:
        def func(x):
            # Fill the correlation potential and calculate f1e
            f1e_fit  = f1e + fill_correlation_potential(x)
            # Obtain the total RDM
            rdm1_fit = (lambda rdm1: rdm1[0] + rdm1[1])(get_density_matrix(f1e_fit))
            # Calculate the difference between the target and fitted RDMs
            rdm1_err = rdm1_tag - rdm1_fit
            # The loss function is the sum of the norms of the RDM differences for each fragment
            err  = jnumpy.linalg.norm(rdm1_err)
            err += sum([jnumpy.linalg.norm(rdm1_err[fit_ind][:, fit_ind]) for fit_ind in fit_inds])
            return err

    else:
        raise ValueError("Invalid loss function type.")

    # Define a class with the loss function and its gradient, as well as the helper functions
    class _LossFunctionMixin:
        pass

    f = _LossFunctionMixin()

    # Main results
    f.func = (lambda x: numpy.array(func(x)))
    f.grad = (lambda x: numpy.array(jax.grad(func)(x)))
    f.hess = (lambda x: numpy.array(jax.hessian(func)(x)))

    # Helper functions
    f._fill_correlation_potential = fill_correlation_potential
    f._get_density_matrix         = get_density_matrix

    # Other attributes
    f._f1e      = f1e
    f._ind_triu = ind_triu
    f._num_triu = num_triu
    f._num_parm = num_parm

    # Return an instance of the class
    return f

from utils import print_matrix
from utils import RestrictedElectronicStructureProblem

from utils import solve_restricted_hartree_fock
from utils import solve_unrestricted_hartree_fock
from utils import solve_direct_spin1_full_configuration_interaction

from hub import build_hub_model
hub_u = 8.0
nsite  = 6
is_debug = False

for nelecs in [(2, 2), (3, 3), (4, 4)]:
    if is_debug and (not nelecs == (3, 3)):
        continue

    log = open(f"./log/hub-u-{hub_u:6.4f}-nelec-{nelecs[0]+nelecs[1]}" + ".log", "w")

    hub_obj         = build_hub_model(nsite, nelecs, hub_u)
    hub_obj.verbose = 4
    hub_obj.stdout  = log
    dm0_alph, dm0_beta = (lambda xs: (numpy.diag(xs[0]), numpy.diag(xs[1][::-1])))(numpy.asarray([[1, 0] for _ in range(nsite)]).reshape(2, -1))

    res_rhf = solve_restricted_hartree_fock(hub_obj,   dm0=dm0_alph+dm0_beta)
    res_uhf = solve_unrestricted_hartree_fock(hub_obj, dm0=(dm0_alph, dm0_beta))
    res_fci = solve_direct_spin1_full_configuration_interaction(hub_obj)

    ene_rhf = res_rhf.get_ene()
    ene_uhf = res_uhf.get_ene()
    ene_fci = res_fci.get_ene()

    r_rdm1_rhf = res_rhf.get_r_rdm1()
    r_rdm1_uhf = res_uhf.get_r_rdm1()
    r_rdm1_fci = res_fci.get_r_rdm1()
    rdm1_tag   = r_rdm1_fci

    for res in [res_rhf, res_uhf, res_fci]:
        ene    = res.get_ene()
        r_rdm1 = res.get_r_rdm1()
        u_rdm1 = res.get_u_rdm1()

        dm_err = numpy.abs(r_rdm1_fci - r_rdm1) 
        err_max = numpy.max(dm_err)
        err_avg = numpy.linalg.norm(dm_err) / numpy.size(dm_err)

        print(f"\n\n{res.__class__.__name__}", file=log)
        print(f"Energy: {ene:6.4f}, Error Max: {err_max:6.4e}, Avg: {err_avg:6.4e}", file=log)
        print_matrix(r_rdm1,    t="r_rdm1 = ", stdout=log)
        print_matrix(u_rdm1[0], t="u_rdm1_alph = ", stdout=log)
        print_matrix(u_rdm1[1], t="u_rdm1_beta = ", stdout=log)

    for igen_loss, gen_loss in enumerate([gen_loss_function_r, gen_loss_function_u, gen_loss_function_g]):
        for (nimp, loss_func_type) in [(2, 1), (2, 2), (2, 3), (nsite, 1)]:
            if is_debug and (not (nimp == 2 and loss_func_type == 2)):
                continue
            
            f = gen_loss(
                hub_obj.h1, rdm1_tag, nelecs=nelecs, loss_func_type=loss_func_type, 
                fit_inds=numpy.asarray([[i+ifrag*nimp for i in range(nimp)] for ifrag in range(nsite // nimp)])
                )

            print("\n\n" + "#"*20, file=log)
            print(f"{gen_loss.__name__}", file=log)
            print(f"nimp = {nimp}, nsite = {nsite}", file=log)
            print(f"nelecs = {nelecs}", file=log)
            print(f"loss_func_type = {loss_func_type}", file=log)

            global count, ymin
            count = 0
            ymin  = None

            def callback(x, y, accepted):
                f1e_fit  = f._f1e + f._fill_correlation_potential(x)
                rdm1_fit = (lambda rdm1: rdm1[0] + rdm1[1])(f._get_density_matrix(f1e_fit))
                rdm1_err = jnumpy.abs(rdm1_tag - rdm1_fit)
                err_mean = jnumpy.linalg.norm(rdm1_err) / numpy.size(rdm1_err)
                err_max  = jnumpy.max(rdm1_err)

                global count, ymin
                if ymin is None:
                    ymin = y
                else:
                    ymin = min(ymin, y)
                
                if is_debug:
                    print(f"count = {count:4d}, y = {y:6.4e}, ymin = {ymin:6.4e}, " + f"x = [" + " ".join([f"{xi:6.4f}" for xi in x]) + "]")
                #     print_matrix(f1e_fit[0], t="f1e_fit  = ")
                #     print_matrix(f1e_fit[1], t="f1e_fit  = ")
                #     print_matrix(rdm1_fit,   t="rdm1_fit = ")
                #     print_matrix(rdm1_err,   t="rdm1_err = ")
                #     assert count != 10


                print(f"count = {count:4d}, y = {y:6.4e}, ymin = {ymin:6.4e}, " + f"x = [" + " ".join([f"{xi:6.4f}" for xi in x])+"]", file=log)
                count += 1

            kwargs = {
                "method": "bfgs", 
                "jac": f.grad, 
                "tol": 1e-4, 
                "options": {"disp": False, "maxiter": 1000}
                }

            x0 = numpy.zeros(f._num_parm)

            res = basinhopping(
                f.func, x0, niter=1000, T=0.1, stepsize=0.6, disp=False,
                callback=callback, minimizer_kwargs=kwargs, 
                niter_success=100, interval=10, 
                )

            x = res.x
            f1e_fit  = f._f1e + f._fill_correlation_potential(x)
            rdm1_fit = (lambda rdm1: rdm1[0] + rdm1[1])(f._get_density_matrix(f1e_fit))
            rdm1_err = jnumpy.abs(rdm1_tag - rdm1_fit)
            err_mean = jnumpy.linalg.norm(rdm1_err) / numpy.size(rdm1_err)
            err_max  = jnumpy.max(rdm1_err)

            print(f"\nLoss Function = {res.fun:6.4e}, Error Mean: {err_mean:6.4e}, Max: {err_max:6.4e}, Count: {count}", file=log)
            print(f"Success = {res.lowest_optimization_result.success}", file=log)
            print(f"Message = {res.lowest_optimization_result.message}", file=log)
            print(f"x = " + " ".join([f"{xi:6.4f}" for xi in x]), file=log)
            
            print(res, file=log)

            print_matrix(rdm1_fit,   t="rdm1_fit = ", stdout=log)
            print_matrix(rdm1_tag,   t="rdm1_tag = ", stdout=log)
            print("\n\n" + "#"*20, file=log)