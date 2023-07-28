import numpy as np

import jax
import jax.scipy.optimize

import scipy.optimize
from scipy._lib._util import check_random_state
from scipy.optimize._basinhopping import MinimizerWrapper
from scipy.optimize._basinhopping import RandomDisplacement
from scipy.optimize._basinhopping import AdaptiveStepsize
from scipy.optimize._basinhopping import Metropolis
from scipy.optimize._basinhopping import BasinHoppingRunner

class JaxBasinHoppingRunner(BasinHoppingRunner):
    def one_cycle(self):
        """Do one cycle of the basinhopping algorithm
        """
        self.nstep += 1
        new_global_min = False

        print(self.x)
        accept, minres = self._monte_carlo_step()

        print(minres.fun, minres.x)
        assert 1 == 2

        if accept:
            self.energy = minres.fun
            self.x = np.copy(minres.x)
            self.incumbent_minres = minres  # best minimize result found so far
            new_global_min = self.storage.update(minres)

        # print some information
        if self.disp:
            self.print_report(minres.fun, accept)
            if new_global_min:
                print("found new global minimum on step %d with function"
                      " value %g" % (self.nstep, self.energy))

        # save some variables as BasinHoppingRunner attributes
        self.xtrial = minres.x
        self.energy_trial = minres.fun
        self.accept = accept

        return new_global_min

class JaxMinimizerWrapper(MinimizerWrapper):
    def __call__(self, x0):
        print(self.func(x0))
        
        if self.func is None:
            minres_jax = self.minimizer(x0, **self.kwargs)
        else:
            print(x0)
            minres_jax = self.minimizer(self.func, jax.numpy.array(x0), **self.kwargs)
        
        print(minres_jax.fun, minres_jax.x, minres_jax.success)

        assert 1 == 2
        minres_sci = scipy.optimize.OptimizeResult(
            x=np.array(minres_jax.x),
            success=minres_jax.success,
            status=minres_jax.status,
            fun=minres_jax.fun,
            jac=np.array(minres_jax.jac),
            hess_inv=np.array(minres_jax.hess_inv),
            nfev=minres_jax.nfev,
            njev=minres_jax.njev,
            nit=minres_jax.nit,
        )
        print(minres_sci.fun, minres_sci.x)

        return minres_sci

def basinhopping(func, x0, niter=100, T=1.0, stepsize=0.5,
                 minimizer_kwargs=None, take_step=None, accept_test=None,
                 callback=None, interval=50, disp=False, niter_success=None,
                 seed=None, *, target_accept_rate=0.5, stepwise_factor=0.9):

    if target_accept_rate <= 0. or target_accept_rate >= 1.:
        raise ValueError('target_accept_rate has to be in range (0, 1)')
    if stepwise_factor <= 0. or stepwise_factor >= 1.:
        raise ValueError('stepwise_factor has to be in range (0, 1)')

    x0 = np.array(x0)

    # set up the np.random generator
    rng = check_random_state(seed)

    # set up minimizer
    if minimizer_kwargs is None:
        minimizer_kwargs = dict()
    wrapped_minimizer = JaxMinimizerWrapper(jax.scipy.optimize.minimize, func,
                                            **minimizer_kwargs)

    # set up step-taking algorithm
    if take_step is not None:
        if not callable(take_step):
            raise TypeError("take_step must be callable")
        # if take_step.stepsize exists then use AdaptiveStepsize to control
        # take_step.stepsize
        if hasattr(take_step, "stepsize"):
            take_step_wrapped = AdaptiveStepsize(
                take_step, interval=interval,
                accept_rate=target_accept_rate,
                factor=stepwise_factor,
                verbose=disp)
        else:
            take_step_wrapped = take_step
    else:
        # use default
        displace = RandomDisplacement(stepsize=stepsize, random_gen=rng)
        take_step_wrapped = AdaptiveStepsize(displace, interval=interval,
                                             accept_rate=target_accept_rate,
                                             factor=stepwise_factor,
                                             verbose=disp)

    # set up accept tests
    accept_tests = []
    if accept_test is not None:
        if not callable(accept_test):
            raise TypeError("accept_test must be callable")
        accept_tests = [accept_test]

    # use default
    metropolis = Metropolis(T, random_gen=rng)
    accept_tests.append(metropolis)

    if niter_success is None:
        niter_success = niter + 2

    bh = JaxBasinHoppingRunner(x0, wrapped_minimizer, take_step_wrapped,
                               accept_tests, disp=disp)

    # The wrapped minimizer is called once during construction of
    # BasinHoppingRunner, so run the callback
    if callable(callback):
        callback(bh.storage.minres.x, bh.storage.minres.fun, True)

    # start main iteration loop
    count, i = 0, 0
    message = ["requested number of basinhopping iterations completed"
               " successfully"]
    for i in range(niter):
        new_global_min = bh.one_cycle()

        if callable(callback):
            # should we pass a copy of x?
            val = callback(bh.xtrial, bh.energy_trial, bh.accept)
            if val is not None:
                if val:
                    message = ["callback function requested stop early by"
                               "returning True"]
                    break

        count += 1
        if new_global_min:
            count = 0
        elif count > niter_success:
            message = ["success condition satisfied"]
            break

    # prepare return object
    res = bh.res
    res.lowest_optimization_result = bh.storage.get_lowest()
    res.x = np.copy(res.lowest_optimization_result.x)
    res.fun = res.lowest_optimization_result.fun
    res.message = message
    res.nit = i + 1
    res.success = res.lowest_optimization_result.success
    return res