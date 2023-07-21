import os, sys
from functools import reduce

import numpy, scipy
from scipy import linalg

import pyscf
from pyscf import gto
from pyscf import scf
from pyscf import fci

from utils import print_matrix
from utils import RestrictedElectronicStructureProblem

from utils import solve_restricted_hartree_fock
from utils import solve_unrestricted_hartree_fock
from utils import solve_direct_spin1_full_configuration_interaction

def build_hub_model(nsite: int, nelecs: tuple[int, int], hub_u: float = 0.0):
    """
    Build the Hubbard model Hamiltonian for a given number of sites, number of
    electrons, and interaction strength.

    Args:
        nsite: number of sites
        nelecs: number of alpha and beta electrons
        u: interaction strength

    Returns:
        A Hubbard model Hamiltonian
    """
    h0 = 0.0

    h1 = numpy.zeros((nsite, nsite))
    for p in range(nsite):
        q = (p + 1) % nsite
        h1[p, q] = h1[q, p] = -1.0
    
    h2 = numpy.zeros((nsite, nsite, nsite, nsite))
    for p in range(nsite):
        h2[p, p, p, p] = hub_u

    nelec_alpha, nelec_beta = nelecs
    assert nelec_alpha == nelec_beta
    hub_obj = RestrictedElectronicStructureProblem(nsite, nelecs, h0=0.0, h1=h1, h2=h2)

    return hub_obj

def fit_density_matrix_r(rdm1_tag, h1e_r, fit_idx_list=None, nelecs=None, err_func=1):
    from scipy.optimize import minimize

    # Assumes that each fragment has the same number of site
    fit_idx_list = numpy.asarray(fit_idx_list)
    nimp  = fit_idx_list.shape[1]
    nsite = nimp * fit_idx_list.shape[0]

    assert h1e_r.shape      == (nsite, nsite)
    assert rdm1_tag.shape == (nsite, nsite)

    # Get the indices of the upper triangle
    triu_idx = numpy.triu_indices(nimp)

    def fill_correlation_potential(x):
        v  = numpy.zeros((nsite, nsite))

        for fit_idx in fit_idx_list:
            fit_idx_0 = fit_idx[triu_idx[0]]
            fit_idx_1 = fit_idx[triu_idx[1]]

            v[fit_idx_0, fit_idx_1] = x
            v[fit_idx_1, fit_idx_0] = x
        
        print("\nx = ", ", ".join(f"{x:6.4f}" for x in x))
        print_matrix(v, "v")

        return v

    def get_density_matrix_u(f1e_r, nelecs):
        assert nelecs[0] == nelecs[1]

        # Solve the restricted Hartree-Fock problem
        mo_ene_alph, mo_coeff_alph = scipy.linalg.eigh(f1e_r)
        mo_ene_beta, mo_coeff_beta = mo_ene_alph, mo_coeff_alph

        occ_idx_alph = numpy.argsort(mo_ene_alph)[:nelecs[0]]
        occ_idx_beta = numpy.argsort(mo_ene_beta)[:nelecs[1]]

        # Build the density matrix
        coeff_occ_alph = mo_coeff_alph[:, occ_idx_alph]
        coeff_occ_beta = mo_coeff_beta[:, occ_idx_beta]
        rdm1_fit_alph  = numpy.dot(coeff_occ_alph, coeff_occ_alph.T)
        rdm1_fit_beta  = numpy.dot(coeff_occ_beta, coeff_occ_beta.T)

        return rdm1_fit_alph, rdm1_fit_beta

    def cost_function(x):
        f1e_r_fit = h1e_r + fill_correlation_potential(x)
        rdm1_fit  = (lambda rho: rho[0] + rho[1])(get_density_matrix_u(f1e_r_fit, nelecs))

        print_matrix(rdm1_tag, "rdm1_tag")
        print_matrix(rdm1_fit, "rdm1_fit")

        if err_func == 1:
            # Error function 1:
            err = numpy.sum([numpy.linalg.norm(rdm1_fit[fit_idx][:, fit_idx] - rdm1_tag[fit_idx][:, fit_idx]) for fit_idx in fit_idx_list]) / nsite / nsite
        elif err_func == 2:
            # Error function 2:
            err = numpy.linalg.norm(rdm1_tag - rdm1_fit) / nsite / nsite
        else:
            raise ValueError(f"Unknown error function: {err_func}")

        print(f"err = {err:12.8f}")
        return err

    x0  = numpy.zeros(nimp * (nimp + 1) // 2)
    res = minimize(cost_function, x0, method="BFGS", tol=1e-8)
    print(res)

def fit_density_matrix_u(rdm1_tag, h1e_u, fit_idx_list=None, nelecs=None, err_func=1):
    from scipy.optimize import minimize

    # Assumes that each fragment has the same number of site
    fit_idx_list = numpy.asarray(fit_idx_list)
    nimp  = fit_idx_list.shape[1]
    nsite = nimp * fit_idx_list.shape[0]

    assert h1e_u.shape     == (2, nsite, nsite)
    assert rdm1_tag.shape == (nsite, nsite)

    # Get the indices of the upper triangle
    triu_idx = numpy.triu_indices(nimp)

    def fill_correlation_potential(x):
        xa = x[:len(x)//2]
        xb = x[len(x)//2:]

        va  = numpy.zeros((nsite, nsite))
        vb  = numpy.zeros((nsite, nsite))

        for fit_idx in fit_idx_list:
            fit_idx_0 = fit_idx[triu_idx[0]]
            fit_idx_1 = fit_idx[triu_idx[1]]

            va[fit_idx_0, fit_idx_1] = xa
            va[fit_idx_1, fit_idx_0] = xa
            vb[fit_idx_0, fit_idx_1] = xb
            vb[fit_idx_1, fit_idx_0] = xb
        
        print("\nx = ", ", ".join(f"{x:6.4f}" for x in x))
        print_matrix(va, "va")
        print_matrix(vb, "vb")

        return numpy.asarray([va, vb])

    def get_density_matrix_u(f1e_u, nelecs):
        assert nelecs[0] == nelecs[1]

        # Solve the restricted Hartree-Fock problem
        mo_ene_alph, mo_coeff_alph = scipy.linalg.eigh(f1e_u[0])
        mo_ene_beta, mo_coeff_beta = scipy.linalg.eigh(f1e_u[1])

        occ_idx_alph = numpy.argsort(mo_ene_alph)[:nelecs[0]]
        occ_idx_beta = numpy.argsort(mo_ene_beta)[:nelecs[1]]

        # Build the density matrix
        coeff_occ_alph = mo_coeff_alph[:, occ_idx_alph]
        coeff_occ_beta = mo_coeff_beta[:, occ_idx_beta]
        rdm1_fit_alph  = numpy.dot(coeff_occ_alph, coeff_occ_alph.T)
        rdm1_fit_beta  = numpy.dot(coeff_occ_beta, coeff_occ_beta.T)

        return rdm1_fit_alph, rdm1_fit_beta

    def cost_function(x):
        f1e_u_fit = h1e_u + fill_correlation_potential(x)
        rdm1_fit  = (lambda rho: rho[0] + rho[1])(get_density_matrix_u(f1e_u_fit, nelecs))

        print_matrix(rdm1_tag, "rdm1_tag")
        print_matrix(rdm1_fit, "rdm1_fit")

        if err_func == 1:
            # Error function 1:
            err = numpy.sum([numpy.linalg.norm(rdm1_fit[fit_idx][:, fit_idx] - rdm1_tag[fit_idx][:, fit_idx]) for fit_idx in fit_idx_list]) / nsite / nsite
        elif err_func == 2:
            # Error function 2:
            err = numpy.linalg.norm(rdm1_tag - rdm1_fit) / nsite / nsite
        else:
            raise ValueError(f"Unknown error function: {err_func}")

        print(f"err = {err:12.8f}")
        return err

    x0  = numpy.zeros(nimp * (nimp + 1))
    res = minimize(cost_function, x0, method="BFGS", tol=1e-8)
    print(res)

    # v = fill_correlation_potential(res.x)
    # rho_fit = get_density_matrix_u(fock_u + v, nelecs)

    # print_matrix(rho_target)
    # print_matrix(rho_fit[0])
    # print_matrix(rho_fit[1])

    # imp_idx = [0, 1]
    # env_idx = [2, 3, 4, 5]

    # rho_imp_env_alph = rho_fit[0][imp_idx][:, env_idx]
    # rho_imp_env_beta = rho_fit[1][imp_idx][:, env_idx]

    # # print_matrix(rho_imp_env_alph)
    # # print_matrix(rho_imp_env_beta)

    # bath_alph = scipy.linalg.svd(rho_imp_env_alph, full_matrices=False)[2].T
    # bath_beta = scipy.linalg.svd(rho_imp_env_beta, full_matrices=False)[2].T
    # print_matrix(bath_alph)
    # print_matrix(bath_beta)

    # bath_tot = numpy.hstack((bath_alph, bath_beta))
    # ovlp_bath = numpy.dot(bath_tot.T, bath_tot)

    # print(bath_tot.shape)
    # print_matrix(bath_tot)
    # print(ovlp_bath.shape)
    # print_matrix(ovlp_bath)

    # u, s, vh = scipy.linalg.svd(ovlp_bath)
    # print(s)

    # from functools import reduce
    # ovlp_bath_ = reduce(numpy.dot, (u, numpy.diag(s), vh))
    # print("err = ", numpy.linalg.norm(ovlp_bath - ovlp_bath_))
    # bath_new = numpy.dot(bath_tot, vh.T)
    # print_matrix(bath_new)
    # assert 1 == 2


    # ovlp_bath_ab = numpy.dot(bath_alph, bath_beta.T)
    # print_matrix(ovlp_bath)

    # print_matrix(bath_alph)
    # print_matrix(bath_beta)
    # assert 1 == 2

    # emb_alph = numpy.zeros((nsite, 4))
    # emb_beta = numpy.zeros((nsite, 4))

    # emb_alph[numpy.ix_(imp_idx, [0, 1])] = numpy.eye(2)
    # emb_beta[numpy.ix_(imp_idx, [0, 1])] = numpy.eye(2)

    # emb_alph[numpy.ix_(env_idx, [2, 3])] = bath_alph.T
    # emb_beta[numpy.ix_(env_idx, [2, 3])] = bath_beta.T

    # # print_matrix(emb_alph)
    # # print_matrix(emb_beta)
    
    # # print(emb_alph.shape)
    # # print(emb_beta.shape)

    # emb_tot = numpy.hstack((emb_alph, emb_beta))
    # ovlp_emb = numpy.dot(emb_tot.T, emb_tot)
    # # print_matrix(ovlp_emb)
    # u, sg, vh = scipy.linalg.svd(ovlp_emb, full_matrices=False)
    # sg_mask = sg > 1e-8

    # u  = u[:, sg_mask]
    # sg = sg[sg_mask]
    # vh = vh[sg_mask, :]

    # from functools import reduce
    # ovlp_emb_ = reduce(numpy.dot, (u, numpy.diag(sg), vh))
    # print(f"err = {numpy.linalg.norm(ovlp_emb - ovlp_emb_)}")
    
    # print(f"u.shape = {u.shape}")
    # print(f"s.shape = {sg.shape}")
    # print(f"vh.shape = {vh.shape}")

def fit_density_matrix_g(rdm1_tag, h1e_g, fit_idx_list=None, nelecs=None, err_func=1):
    from scipy.optimize import minimize

    # Assumes that each fragment has the same number of site
    fit_idx_list = numpy.asarray(fit_idx_list)
    nimp  = fit_idx_list.shape[1]
    nsite = nimp * fit_idx_list.shape[0]

    assert h1e_g.shape    == (2 * nsite, 2 * nsite)
    assert rdm1_tag.shape == (nsite, nsite)

    # Get the indices of the upper triangle
    triu_idx = numpy.triu_indices(2 * nsite)

    def fill_correlation_potential(x):
        v = numpy.zeros((2 * nsite, 2 * nsite))
        v[triu_idx] = x
        return v


    def get_density_matrix_g(h1e_g, nelecs):
        assert nelecs[0] == nelecs[1]
        nelec_tot = nelecs[0] + nelecs[1]

        mo_ene, mo_coeff = scipy.linalg.eigh(h1e_g)
        occ_idx = numpy.argsort(mo_ene)[:nelec_tot]

        coeff_occ = mo_coeff[:, occ_idx]
        rdm1_fit  = numpy.dot(coeff_occ, coeff_occ.T)

        return rdm1_fit

    def cost_function(x):
        f1e_g_fit = h1e_g + fill_correlation_potential(x)
        rdm1_fit  = (lambda rho: rho[:nsite, :nsite] + rho[nsite:, nsite:])(get_density_matrix_g(f1e_g_fit, nelecs))

        print_matrix(rdm1_tag, "rdm1_tag")
        print_matrix(rdm1_fit, "rdm1_fit")

        if err_func == 1:
            # Error function 1:
            err = numpy.sum([numpy.linalg.norm(rdm1_fit[fit_idx][:, fit_idx] - rdm1_tag[fit_idx][:, fit_idx]) for fit_idx in fit_idx_list]) / nsite / nsite
        elif err_func == 2:
            # Error function 2:
            err = numpy.linalg.norm(rdm1_tag - rdm1_fit) / nsite
        else:
            raise ValueError(f"Unknown error function: {err_func}")

        print(f"err = {err:12.8f}")
        return err

    # x0  = numpy.zeros(nsite * (nsite + 1))
    x0 = numpy.random.rand(len(triu_idx[0]))
    res = minimize(cost_function, x0, method="BFGS", tol=1e-4, options={"maxiter": 40000})
    print(res)

if __name__ == "__main__":

    for hub_u in [8.0, 6.0, 4.0, 2.0]:
        nsite  = 10
        nimp   = 2
        nelecs = (4, 4)
        fit_idx_list = numpy.arange(nsite).reshape(-1, nimp)

        hub_obj = build_hub_model(nsite, nelecs, hub_u)
        hub_obj.verbose = 5
        dm0_alph, dm0_beta = (lambda xs: (numpy.diag(xs[0]), numpy.diag(xs[1][::-1])))(numpy.asarray([[1, 0] for _ in range(nsite)]).reshape(2, -1))

        res_rhf = solve_restricted_hartree_fock(hub_obj,   dm0=dm0_alph+dm0_beta)
        res_uhf = solve_unrestricted_hartree_fock(hub_obj, dm0=(dm0_alph, dm0_beta))
        res_fci = solve_direct_spin1_full_configuration_interaction(hub_obj)

        for res in [res_rhf, res_uhf, res_fci]:
            ene = res.get_ene()
            r_rdm1 = res.get_r_rdm1()
            u_rdm1 = res.get_u_rdm1()

            print(f"\n\n{res.__class__.__name__}\n\nenergy = {ene:12.8f}")
            print_matrix(r_rdm1, t="r_rdm1 = ")
            print_matrix(u_rdm1[0], t="u_rdm1_alph = ")
            print_matrix(u_rdm1[1], t="u_rdm1_beta = ")

        ene_rhf = res_rhf.get_ene()
        ene_uhf = res_uhf.get_ene()
        ene_fci = res_fci.get_ene()

        r_rdm1_rhf = res_rhf.get_r_rdm1()
        r_rdm1_uhf = res_uhf.get_r_rdm1()
        r_rdm1_fci = res_fci.get_r_rdm1()

        u_rdm1_rhf = res_rhf.get_u_rdm1()
        u_rdm1_uhf = res_uhf.get_u_rdm1()
        u_rdm1_fci = res_fci.get_u_rdm1()

        fock_rhf = res_rhf.hf_obj.get_hcore()
        fock_uhf = numpy.asarray((fock_rhf, fock_rhf))
        fock_ghf = numpy.block([[fock_rhf, numpy.zeros_like(fock_rhf)], [numpy.zeros_like(fock_rhf), fock_rhf]])

        # print_matrix(fock_rhf, t="fock_rhf = ")
        # print_matrix(fock_uhf[0], t="fock_uhf_alph = ")
        # print_matrix(fock_uhf[1], t="fock_uhf_beta = ")

        # Fit the FCI density matrix block with 
        
        # fit_density_matrix_r(r_rdm1_fci, fock_rhf, fit_idx_list, nelecs, err_func=1)
        # fit_density_matrix_u(r_rdm1_fci, fock_uhf, fit_idx_list, nelecs, err_func=1)
        fit_density_matrix_g(r_rdm1_fci, fock_ghf, fit_idx_list, nelecs, err_func=1)
        fit_density_matrix_g(r_rdm1_fci, fock_ghf, fit_idx_list, nelecs, err_func=2)

        assert 1 == 2

        # fit_density_matrix_r(r_rdm1_fci, fock_rhf, fit_idx_list, nelecs, err_func=2)
        # fit_density_matrix_u(r_rdm1_fci, fock_uhf, fit_idx_list, nelecs, err_func=2)
        # assert 1 == 2