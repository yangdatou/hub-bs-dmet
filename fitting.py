import os, sys, typing
from typing import List, Tuple, Callable
sys.path.append(".")

from functools import reduce

import numpy, scipy
from scipy import linalg

import jax
from jax import config
config.update("jax_enable_x64", True)
config.update("jax_debug_nans", True)

import jax.numpy as jnumpy
from jax.numpy import diag
from jax.scipy.linalg import eigh
from jax.scipy.linalg import block_diag
from pyscf import lib

# Note: 
# - The code use spin = 1, 2, 4 to represent the spin symmetry
#   of the system. The value of spin is used to determine the
#   shape of all the related matrices and tensors.
# - The variables ending with "_r" are arrays with shape 
#   (nsite, nsite), which means they are identical for alpha
#   and beta spin. "_r" will be omitted if unlikely to cause
#   confusion. For example, h1e is the one-electron Hamiltonian
#   for alpha and beta spin.
# - The density matrices that is a sum of the alpha and beta
#   will be labbeled as "rho_"; the shape shall be (nsite, nsite).
#   Other density matrices will be labelled as "rdm1_" with 
#   the proper suffix. rdm1_r wi
# - The variables ending with "_s" are with shape (spin, nsite, nsite);
#   the variables ending with "_g" are arrays with shape 
#   (2 * nsite, 2 * nsite).

# s_to_g: spin to general
# g_to_s: general to spin
# tot: get the total density from the RDM

def v1e_s_to_g(v1e_s: numpy.ndarray, spin=1) -> numpy.ndarray:
    # s_to_g: spin to general
    assert spin in [1, 2, 4]

    if v1e_s.ndim == 2:
        v1e_s = v1e_s[None]

    nsite = v1e_s.shape[-1]
    assert v1e_s.shape == (spin, nsite, nsite)

    if spin == 1:
        v1e_aa = v1e_bb = v1e_s[0]
        v1e_ab = v1e_ba = jnumpy.zeros((nsite, nsite))

    elif spin == 2:
        v1e_aa = v1e_s[0]
        v1e_ab = v1e_s[1]
        v1e_ab = v1e_ba = jnumpy.zeros((nsite, nsite))

    elif spin == 4:
        v1e_aa = v1e_s[0]
        v1e_ab = v1e_s[1]
        v1e_ba = v1e_s[2]
        v1e_bb = v1e_s[3]
    
    v1e_g = jnumpy.block([[v1e_aa, v1e_ab], [v1e_ba, v1e_bb]])
    return v1e_g

def v1e_g_to_s(v1e_g: numpy.ndarray, spin=1) -> numpy.ndarray:
    assert spin in [1, 2, 4]
    nsite = v1e_g.shape[-1] // 2

    v1e_aa = v1e_g.at[:nsite, :nsite]
    v1e_ab = v1e_g.at[:nsite, nsite:]
    v1e_ba = v1e_g.at[nsite:, :nsite]
    v1e_bb = v1e_g.at[nsite:, nsite:]

    v1e_s = jnumpy.zeros((spin, nsite, nsite))
    if spin == 1:
        assert jnumpy.allclose(v1e_ab, 0)
        assert jnumpy.allclose(v1e_ba, 0)
        assert jnumpy.allclose(v1e_aa, v1e_bb)
        v1e_s = v1e_aa[None]

    if spin == 2:
        assert jnumpy.allclose(v1e_ab, 0.0)
        assert jnumpy.allclose(v1e_ba, 0.0)
        v1e_s = jnumpy.array([v1e_aa, v1e_ab])
    
    if spin == 4:
        v1e_s = jnumpy.array([v1e_aa, v1e_ab, v1e_ba, v1e_bb])

    assert v1e_s.shape == (spin, nsite, nsite)
    return v1e_s

class LossFunctionMixin(lib.StreamObject):
    spin   = None

    def __init__(self, h1e_r: numpy.ndarray, rho_tag: numpy.ndarray, nelecs: Tuple[int, int],
                 nimp: int = 2, loss_func_type: int = 1, stdout: typing.TextIO = sys.stdout):
        self.stdout = stdout

        spin = self.spin
        assert spin in [1, 2, 4]

        # Convert fit_inds to a JAX array and extract dimensions
        # assume all the fragments have the same number of impurity sites.
        # fit_inds = jnumpy.asarray(fit_inds)
        nsite    = h1e_r.shape[0]
        nfrag    = nsite // nimp
        assert nfrag * nimp == nsite 

        num_parm  = spin * nimp * (nimp + 1) // 2
        num_parm -= nimp * (spin == 4)
        
        # Ensure that dimensions of f1e and rho_tag match expected dimensions
        assert h1e_r.shape   == (nsite, nsite)
        assert rho_tag.shape == (nsite, nsite)
        h1e_g = block_diag(h1e_r, h1e_r)
        self.h1e_g = h1e_g

        self.rho_tag  = rho_tag
        self.nelecs   = nelecs
        self.loss_func_type = loss_func_type

        self.num_parm = num_parm

        self.nfrag    = nfrag
        self.nimp     = nimp
        self.nsite    = nsite

        get_v1e_g = self._gen_get_v1e_g()
        get_rho   = self._gen_get_rho()

        # Helper functions
        # self._get_v1es = None # self._gen_get_v1es()
        # self._get_rdm1 = None # self._gen_get_rdm1()
        
        # If the number of fragments is 1, then all the
        # types of loss functions are equivalent.
        assert loss_func_type == 1 or nfrag != 1

        from jax.numpy.linalg import norm
        if loss_func_type == 1:
            def func(x):
                # Fill the correlation potential and calculate f1e
                v1e_g_fit = get_v1e_g(x)
                f1e_g_fit = h1e_g + v1e_g_fit
                # Obtain the total RDM
                rho_fit   = get_rho(f1e_g_fit)
                # Calculate the difference between the target and fitted RDMs
                rho_err   = rho_fit - rho_tag

                # The loss function is the norm of the RDM difference
                return norm(rho_err)

        elif loss_func_type == 2:
            def func(x):
                # Fill the correlation potential and calculate f1e
                v1e_g_fit = get_v1e_g(x)
                f1e_g_fit = h1e_g + v1e_g_fit
                # Obtain the total RDM
                rho_fit   = get_rho(f1e_g_fit)
                # Calculate the difference between the target and fitted RDMs
                rho_err   = rho_fit - rho_tag
                
                # Get the diagonal blocks of the RDM
                inds = jnumpy.arange(nsite).reshape(nimp, nfrag)
                err  = sum([norm(rho_err[jnumpy.ix_(ind, ind)]) for ind in inds])
                return err

        else:
            raise ValueError("Invalid loss function type.")

        self.func = func
        self._dump_info()

    def _dump_info(self):
        info = self.__dict__
        class_name = " " + self.__class__.__name__ + " "
        self.stdout.write("\n\n" + "#" * 20 + class_name + "#" * 20 + "\n")
        self.stdout.write("Loss Function Info:\n")
        
        for k, v in info.items():
            self.stdout.write(f"{k} = {v}\n")

        self.stdout.write("#" * (40 +  len(class_name)) + "\n")

    def _gen_get_v1e_g(self):
        raise NotImplementedError

    def _gen_get_rho(self):
        spin     = self.spin
        nsite    = self.nsite
        nimp     = self.nimp
        nfrag    = self.nfrag

        nelec_tot = self.nelecs[0] + self.nelecs[1]

        def get_rho(f1e_g):
            assert f1e_g.shape  == (2 * nsite, 2 * nsite)
            print("f1eg = \n", f1e_g)
            ene_g, coeff_g = eigh(f1e_g)
            occ_idx_g = jnumpy.argsort(ene_g)[:nelec_tot]
            coeff_occ_g = coeff_g[:, occ_idx_g]
            rho_g = jnumpy.dot(coeff_occ_g, coeff_occ_g.T)
            rho_aa = rho_g[:nsite, :nsite]
            rho_bb = rho_g[nsite:, nsite:]
            return rho_aa + rho_bb

        return get_rho

class RestrictedSpinLossFunction(LossFunctionMixin):
    spin = 1
    def _gen_get_v1e_g(self):
        spin     = self.spin
        nsite    = self.nsite
        nimp     = self.nimp
        nfrag    = self.nfrag
        num_parm = self.num_parm

        def get_v1e_g(x):
            assert x.shape == (num_parm,)

            v1e_imp_aa = jnumpy.zeros((nimp, nimp))
            v1e_imp_aa = v1e_imp_aa.at[jnumpy.triu_indices(nimp)].set(x)
            v1e_imp_aa = (lambda x: x + x.T - diag(diag(x)))(v1e_imp_aa)
            v1e_imp_bb = v1e_imp_aa

            v1e_aa = block_diag(*[v1e_imp_aa for _ in range(nfrag)])
            v1e_bb = v1e_aa

            v1e_g  = block_diag(v1e_aa, v1e_bb)
            return v1e_g

        return get_v1e_g

class UnrestrictedSpinLossFunction(LossFunctionMixin):
    spin = 2
    def _gen_get_v1e_g(self):
        spin     = self.spin
        nsite    = self.nsite
        nimp     = self.nimp
        nfrag    = self.nfrag
        num_parm = self.num_parm

        fit_inds = self.fit_inds

        def get_v1e_g(x):
            assert x.shape == (num_parm,)

            v1e_imp_aa = jnumpy.zeros((nimp, nimp))
            v1e_imp_aa = v1e_imp_aa.at[jnumpy.triu_indices(nimp)].set(x)
            v1e_imp_aa = (lambda x: x + x.T - diag(diag(x)))(v1e_imp_aa)
            v1e_imp_bb = v1e_imp_aa

            v1e_aa = block_diag(*[v1e_imp_aa for _ in range(nfrag)])
            v1e_bb = v1e_aa

            v1e_g  = block_diag(v1e_aa, v1e_bb)
            return v1e_g

        return get_v1e_g

class GeneralizedSpinLossFunction(LossFunctionMixin):
    spin = 4

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
nsite  = 4
is_debug = True

for nelecs in [(2, 2), (3, 3), (4, 4)]:
    if is_debug and (not nelecs == (2, 2)):
        continue

    log = open(f"./log/nsite-{nsite}hub-u-{hub_u:6.4f}-nelec-{nelecs[0]+nelecs[1]}" + ".log", "w")

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
    rho_tag   = r_rdm1_fci

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

    for igen_loss, LF in enumerate([RLF]):
        for (nimp, loss_func_type) in [(2, 2)]:
            kwargs   = {"stdout": log, "nimp": nimp, "nelecs": nelecs, "loss_func_type": loss_func_type}
            lf       = LF(hub_obj.h1, rho_tag, **kwargs)

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
                # # Fill the correlation potential and calculate f1e
                # v1es_fit = lf._get_v1es(x)
                # f1es_fit = lf.h1e_s + v1es_fit
                # # Obtain the total RDM
                # rho_fit, gdm1_fit = lf._get_rdm1(f1es_fit)
                # # Calculate the difference between the target and fitted RDMs
                # rdm1_err = jnumpy.abs(rho_tag - rho_fit)

                # err_mean = jnumpy.linalg.norm(rdm1_err) / numpy.size(rdm1_err)
                # err_max  = jnumpy.max(rdm1_err)

                global count, ymin
                ymin = y if ymin is None else min(ymin, y)
                
                # if is_debug:
                    # print(f"count = {count:4d}, y = {y:6.4e}, ymin = {ymin:6.4e}, " + f"x = [" + " ".join([f"{xi:6.4f}" for xi in x]) + "]")
                #     log.write(f"#{count:4d} {y:6.4e} {accepted}\n")
                #     log.write(f"x = {x}\n")
                #     print_matrix(v1es_fit[0], t="v1es_fit_aa = ", stdout=log)
                #     print_matrix(v1es_fit[3], t="v1es_fit_bb = ", stdout=log)
                #     print_matrix(v1es_fit[1], t="v1es_fit_ab = ", stdout=log)
                #     print_matrix(v1es_fit[2], t="v1es_fit_ba = ", stdout=log)

                print(f"count = {count:4d}, y = {y:6.4e}, ymin = {ymin:6.4e}, " + f"x = [" + " ".join([f"{xi:6.4f}" for xi in x])+"]")
                count += 1

            kwargs = {
                "method": "bfgs", "tol": 1e-6, 
                "options": {"maxiter": 1000}, 
                }

            from minimize import basinhopping
            res = basinhopping(
                lf.func, x0, T=0.1, stepsize=2.0, disp=False,
                callback=callback, minimizer_kwargs=kwargs, 
                niter=4000, niter_success=100, interval=10, 
                )

            x = res.x
            f1es_fit = lf.h1e_s + lf._get_v1es(x)
            rho_fit = lf._get_rdm1(f1es_fit)[0]
            rdm1_err = jnumpy.abs(rho_tag - rho_fit)
            err_mean = jnumpy.linalg.norm(rdm1_err) / numpy.size(rdm1_err)
            err_max  = jnumpy.max(rdm1_err)

            print(f"\nLoss Function = {res.fun:6.4e}, Error Mean: {err_mean:6.4e}, Max: {err_max:6.4e}, Count: {count}", file=log)
            print(f"Success = {res.lowest_optimization_result.success}", file=log)
            print(f"Message = {res.lowest_optimization_result.message}", file=log)
            print(f"x = " + " ".join([f"{xi:6.4f}" for xi in x]), file=log)
            
            print(res, file=log)

            print_matrix(rho_fit,   t="rho_fit = ", stdout=log)
            print_matrix(rho_tag,   t="rho_tag = ", stdout=log)
            print("\n\n" + "#"*20, file=log)