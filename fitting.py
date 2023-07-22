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

from pyscf import lib

class LossFunctionMixin(lib.StreamObject):
    spin   = None

    def __init__(self, h1e: numpy.ndarray, rdm1_tag: numpy.ndarray, fit_inds: numpy.ndarray,
                 nelecs: Tuple[int, int], loss_func_type: int = 1, stdout: typing.TextIO = sys.stdout):
        self.stdout = stdout

        spin = self.spin
        assert spin is not None

        # Convert fit_inds to a JAX array and extract dimensions
        fit_inds = jnumpy.asarray(fit_inds)
        nfrag    = fit_inds.shape[0]
        nimp     = fit_inds.shape[1]
        nsite    = nimp * nfrag

        ind_triu = jnumpy.triu_indices(nimp)
        num_triu = ind_triu[0].shape[0]
        num_parm = num_triu * spin
        
        # Ensure that dimensions of f1e and rdm1_tag match expected dimensions
        assert h1e.shape      == (nsite, nsite)
        assert rdm1_tag.shape == (nsite, nsite)

        self.h1e = h1e
        self.rdm1_tag = rdm1_tag
        self.fit_inds = fit_inds
        self.nelecs = nelecs
        self.loss_func_type = loss_func_type

        self.ind_triu = ind_triu
        self.num_triu = num_triu
        self.num_parm = num_parm

        self.nfrag    = nfrag
        self.nimp     = nimp
        self.nsite    = nsite

        h1es = jnumpy.zeros((spin, nsite, nsite))
        h1es = h1es.at[0].set(h1e)
        h1es = h1es.at[-1].set(h1e)
        self.h1es = h1es

        get_v1es = self._gen_get_v1es()
        get_rdm1 = self._gen_get_rdm1()

        # Helper functions
        self._get_v1es = self._gen_get_v1es()
        self._get_rdm1 = self._gen_get_rdm1()
        
        # If the number of fragments is 1, then all the
        # types of loss functions are equivalent.
        assert loss_func_type == 1 or nfrag != 1

        if loss_func_type == 1:
            def func(xs):
                # Fill the correlation potential and calculate f1e
                v1es_fit = get_v1es(xs)
                f1es_fit = h1es + v1es_fit
                # Obtain the total RDM
                rdm1_fit = get_rdm1(f1es_fit)[0]
                # Calculate the difference between the target and fitted RDMs
                rdm1_err = rdm1_tag - rdm1_fit
                
                # The loss function is the norm of the RDM difference
                err = jnumpy.linalg.norm(rdm1_err)
                return err

        elif loss_func_type == 2:
            def func(xs):
                # Fill the correlation potential and calculate f1e
                v1es_fit = get_v1es(xs)
                f1es_fit = h1es + v1es_fit
                # Obtain the total RDM
                rdm1_fit = get_rdm1(f1es_fit)[0]
                # Calculate the difference between the target and fitted RDMs
                rdm1_err = rdm1_tag - rdm1_fit

                # The loss function is the sum of the norms of the RDM differences for each fragment
                err = sum([jnumpy.linalg.norm(rdm1_err[fit_ind][:, fit_ind]) for fit_ind in fit_inds])
                return err

        elif loss_func_type == 3:
            def func(xs):
                # Fill the correlation potential and calculate f1e
                v1es_fit = get_v1es(xs)
                f1es_fit = h1es + v1es_fit
                # Obtain the total RDM
                rdm1_fit = get_rdm1(f1es_fit)[0]
                # Calculate the difference between the target and fitted RDMs
                rdm1_err = rdm1_tag - rdm1_fit

                # The loss function is the sum of the norms of the RDM differences for each fragment
                err  = jnumpy.linalg.norm(rdm1_err)
                err += sum([jnumpy.linalg.norm(rdm1_err[fit_ind][:, fit_ind]) for fit_ind in fit_inds])
                return err

        else:
            raise ValueError("Invalid loss function type.")

        self.func = (lambda x: numpy.array(func(x)))
        self.grad = (lambda x: numpy.array(jax.grad(func)(x)))
        self.hess = (lambda x: numpy.array(jax.hessian(func)(x)))
        self._dump_info()

    def _dump_info(self):
        info = self.__dict__
        class_name = " " + self.__class__.__name__ + " "
        self.stdout.write("\n\n" + "#" * 20 + class_name + "#" * 20 + "\n")
        self.stdout.write("Loss Function Info:\n")
        
        for k, v in info.items():
            self.stdout.write(f"{k} = {v}\n")

        self.stdout.write("#" * (40 +  len(class_name)) + "\n")

    def _gen_get_v1es(self):
        spin     = self.spin
        nsite    = self.nsite
        nimp     = self.nimp
        nfrag    = self.nfrag

        num_triu = self.num_triu
        ind_triu = self.ind_triu
        fit_inds = self.fit_inds

        def get_v1es(xs):
            assert xs.shape == (spin * num_triu,)
            v1es = jnumpy.zeros((spin, nsite, nsite))

            for s, x in enumerate(jnumpy.split(xs, spin)):
                for fit_ind in fit_inds:
                    ind_0 = fit_ind[ind_triu[0]]
                    ind_1 = fit_ind[ind_triu[1]]
                    v1es  = v1es.at[s, ind_0, ind_1].set(x)

                v1e_sym = v1es[s] + jnumpy.transpose(v1es[s]) - jnumpy.diag(jnumpy.diag(v1es[s]))
                v1es    = v1es.at[s].set(v1e_sym)
            
            return v1es

        return get_v1es

    def _gen_get_rdm1(self, is_debug = False):
        raise NotImplementedError

class RestrictedSpinLossFunction(LossFunctionMixin):
    spin = 1
    def _gen_get_rdm1(self):
        spin     = self.spin
        nsite    = self.nsite
        nimp     = self.nimp
        nfrag    = self.nfrag

        nelec_alph, nelec_beta = self.nelecs

        num_triu = self.num_triu
        ind_triu = self.ind_triu
        fit_inds = self.fit_inds

        def get_rdm1(f1es):
            assert f1es.shape  == (spin, nsite, nsite)
            assert nelec_alpha == nelec_beta

            f1e_aa = f1e_bb = f1es[0]
            mo_ene_alph, mo_coeff_alph = eigh(f1e_aa)
            mo_ene_beta, mo_coeff_beta = mo_ene_alph, mo_coeff_alph

            occ_idx_alph = jnumpy.argsort(mo_ene_alph)[:nelec_alph]
            occ_idx_beta = jnumpy.argsort(mo_ene_beta)[:nelec_beta]

            coeff_occ_alph = mo_coeff_alph[:, occ_idx_alph]
            coeff_occ_beta = mo_coeff_beta[:, occ_idx_beta]

            rdm1_fit_alph  = jnumpy.dot(coeff_occ_alph, coeff_occ_alph.T)
            rdm1_fit_beta  = jnumpy.dot(coeff_occ_beta, coeff_occ_beta.T)

            gdm1 = jnumpy.zeros((2 * nsite, 2 * nsite))
            gdm1 = gdm1.at[:nsite, :nsite].set(rdm1_fit_alph)
            gdm1 = gdm1.at[nsite:, nsite:].set(rdm1_fit_beta)

            return rdm1_fit_alph + rdm1_fit_beta, gdm1

        return get_rdm1

class UnrestrictedSpinLossFunction(LossFunctionMixin):
    spin = 2
    def _gen_get_rdm1(self):
        spin     = self.spin
        nsite    = self.nsite
        nimp     = self.nimp
        nfrag    = self.nfrag

        nelec_alph, nelec_beta = self.nelecs

        num_triu = self.num_triu
        ind_triu = self.ind_triu
        fit_inds = self.fit_inds

        def get_rdm1(f1es):
            assert f1es.shape  == (spin, nsite, nsite)
            assert nelec_alpha == nelec_beta

            f1e_aa, f1e_bb = f1es
            mo_ene_alph, mo_coeff_alph = eigh(f1e_aa)
            mo_ene_beta, mo_coeff_beta = eigh(f1e_bb)

            occ_idx_alph = jnumpy.argsort(mo_ene_alph)[:nelec_alph]
            occ_idx_beta = jnumpy.argsort(mo_ene_beta)[:nelec_beta]

            coeff_occ_alph = mo_coeff_alph[:, occ_idx_alph]
            coeff_occ_beta = mo_coeff_beta[:, occ_idx_beta]

            rdm1_fit_aa  = jnumpy.dot(coeff_occ_alph, coeff_occ_alph.T)
            rdm1_fit_bb  = jnumpy.dot(coeff_occ_beta, coeff_occ_beta.T)

            gdm1_fit = jnumpy.block([[rdm1_fit_aa, jnumpy.zeros((nsite, nsite))], [jnumpy.zeros((nsite, nsite)), rdm1_fit_bb]])

            return rdm1_fit_alph + rdm1_fit_beta, gdm1_fit

        return get_rdm1

class GeneralizedSpinLossFunction(LossFunctionMixin):
    spin = 4
    def _gen_get_rdm1(self):
        spin     = self.spin
        nsite    = self.nsite
        nimp     = self.nimp
        nfrag    = self.nfrag

        nelec_alpha, nelec_beta = self.nelecs

        num_triu = self.num_triu
        ind_triu = self.ind_triu
        fit_inds = self.fit_inds

        def get_rdm1(f1es):
            assert f1es.shape  == (spin, nsite, nsite)
            assert nelec_alpha == nelec_beta

            f1e_aa, f1e_ab, f1e_ba, f1e_bb = f1es
            f1e_g = jnumpy.block([[f1e_aa, f1e_ab], [f1e_ba, f1e_bb]])

            mo_ene_g, mo_coeff_g = eigh(f1e_g)
            occ_idx_g = jnumpy.argsort(mo_ene_g)[:(nelec_alpha + nelec_beta)]

            coeff_occ_g = mo_coeff_g[:, occ_idx_g]
            gdm1_fit = jnumpy.dot(coeff_occ_g, coeff_occ_g.T)

            rdm1_fit_alph = gdm1_fit[:nsite, :nsite]
            rdm1_fit_beta = gdm1_fit[nsite:, nsite:]

            return rdm1_fit_alph + rdm1_fit_beta, gdm1_fit

        return get_rdm1

RLF = RestrictedSpinLossFunction
ULF = UnrestrictedSpinLossFunction
GLF = GeneralizedSpinLossFunction

from utils import print_matrix
from utils import RestrictedElectronicStructureProblem

from utils import solve_restricted_hartree_fock
from utils import solve_unrestricted_hartree_fock
from utils import solve_direct_spin1_full_configuration_interaction

from hub import build_hub_model
hub_u = 8.0
nsite  = 6
is_debug = True

for nelecs in [(2, 2), (3, 3), (4, 4)]:
    if is_debug and (not nelecs == (2, 2)):
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

    for igen_loss, LF in enumerate([RLF, ULF, GLF]):
        for (nimp, loss_func_type) in [(2, 1), (2, 2), (2, 3), (nsite, 1)]:
            if is_debug and (not (nimp == nsite and loss_func_type == 1 and igen_loss == 2)):
                continue
            
            fit_inds = numpy.asarray([[i+ifrag*nimp for i in range(nimp)] for ifrag in range(nsite // nimp)])
            kwargs   = {"stdout": log, "fit_inds": fit_inds, "nelecs": nelecs, "loss_func_type": loss_func_type}

            lf       = LF(hub_obj.h1, rdm1_tag, **kwargs)
            print("\n\n" + "#"*20, file=log)
            print(f"{lf.__class__.__name__}", file=log)
            print(f"nimp = {nimp}, nsite = {nsite}", file=log)
            print(f"nelecs = {nelecs}", file=log)
            print(f"loss_func_type = {loss_func_type}", file=log)

            x0 = numpy.zeros(lf.num_parm)

            global count, ymin
            count = 0
            ymin  = None

            def callback(x, y, accepted):
                # Fill the correlation potential and calculate f1e
                v1es_fit = lf._get_v1es(x)
                f1es_fit = lf.h1es + v1es_fit
                # Obtain the total RDM
                rdm1_fit, gdm1_fit = lf._get_rdm1(f1es_fit)
                # Calculate the difference between the target and fitted RDMs
                rdm1_err = jnumpy.abs(rdm1_tag - rdm1_fit)

                err_mean = jnumpy.linalg.norm(rdm1_err) / numpy.size(rdm1_err)
                err_max  = jnumpy.max(rdm1_err)

                global count, ymin
                ymin = y if ymin is None else min(ymin, y)
                
                if is_debug:
                    print(f"count = {count:4d}, y = {y:6.4e}, ymin = {ymin:6.4e}, " + f"x = [" + " ".join([f"{xi:6.4f}" for xi in x]) + "]")

                print(f"count = {count:4d}, y = {y:6.4e}, ymin = {ymin:6.4e}, " + f"x = [" + " ".join([f"{xi:6.4f}" for xi in x])+"]", file=log)
                count += 1

            kwargs = {
                "method": "bfgs", 
                "jac": lf.grad, "tol": 1e-6, 
                "options": {"disp": False, "maxiter": 1000}
                }

            res = basinhopping(
                lf.func, x0, T=0.1, stepsize=2.0, disp=False,
                callback=callback, minimizer_kwargs=kwargs, 
                niter=4000, niter_success=100, interval=10, 
                )

            x = res.x
            f1es_fit = lf.h1es + lf._get_v1es(x)
            rdm1_fit = lf._get_rdm1(f1es_fit)[0]
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