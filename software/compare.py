import warnings
import tqdm
import scipy

import numpy as np
import pandas as pd

import scipy.stats as st

def log_like_iid_gamma(log_params, n):
    """Log likelihood for i.i.d. Gamma measurements with
    input being logarithm of parameters.

    Parameters
    ----------
    log_params : array
        Logarithm of the parameters alpha and b.
    n : array
        Array of counts.

    Returns
    -------
    output : float
        Log-likelihood.
    """
    # Extract parameters and exponentiate
    log_alpha, log_beta = log_params
    alpha = np.exp(log_alpha)
    beta = np.exp(log_beta)
    
    # Return log likelihood for gamma
    return np.sum(st.gamma.logpdf(n, alpha, scale=1/beta))

def mle_iid_gamma(n):
    """Perform maximum likelihood estimates for parameters for i.i.d.
    gamma measurements, parametrized by alpha, b=1/beta"""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        # Optimize log likelihood function for gamma
        res = scipy.optimize.minimize(
            fun=lambda log_params, n: -log_like_iid_gamma(log_params, n),
            x0=np.array([1, 1]),
            args=(n,),
            method='BFGS',
            tol=1e-3
        )
    
    # Return result
    if res.success:
        return res.x
    else:
        raise RuntimeError('Convergence failed with message', res.message)

def log_like_iid_poisson(log_params, n):
    """Log likelihood for i.i.d. convoluted poisson measurements with
    input being logarithm of parameters.

    Parameters
    ----------
    log_params : array
        Logarithm of the parameters alpha and b.
    n : array
        Array of counts.

    Returns
    -------
    output : float
        Log-likelihood.
    """
    
    # Extract parameters and exponentiate
    log_b1, log_b2 = log_params
    b1 = np.exp(log_b1)
    b2 = np.exp(log_b2)
    alpha = 2
    
    # Return results
    if abs(b2 - b1) < 0.01:
        # Acts as two step gamma when beta1 is similar to beta 2
        return np.sum(st.gamma.logpdf(n, alpha, scale=1/b1))
    elif b2 > b1:
        # Log-sum-exp when beta2 > beta1
        return np.sum(log_b1 + log_b2 - np.log(b2-b1) - b2*n)
    else:
        # Log-sum-exp when beta1 > beta2
        return np.sum(log_b1 + log_b2 - np.log(b2-b1) - b1*n)

def mle_iid_poisson(n):
    """Perform maximum likelihood estimates for parameters for i.i.d.
    convoluted poisson measurements, parametrized by alpha, beta1, beta2"""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        # Optimize log likelihood function for convoluted poisson
        res = scipy.optimize.minimize(
            fun=lambda log_params, n: -log_like_iid_poisson(log_params, n),
            x0=np.array([0.01, 0.005]),
            args=(n,),
            method='BFGS',
            tol=1e-3
        )
    
    # Return result
    if res.success:
        return res.x
    else:
        raise RuntimeError('Convergence failed with message', res.message)

rg = np.random.default_rng()

def draw_bs_sample(data):
    """Draw a bootstrap sample from a 1D data set."""
    return rg.choice(data, size=len(data))


def draw_bs_reps_mle(mle_fun, data, args=(), size=1, progress_bar=False):
    """Draw nonparametric bootstrap replicates of maximum likelihood estimator.

    Parameters
    ----------
    mle_fun : function
        Function with call signature mle_fun(data, *args) that computes
        a MLE for the parameters
    data : one-dimemsional Numpy array
        Array of measurements
    args : tuple, default ()
        Arguments to be passed to `mle_fun()`.
    size : int, default 1
        Number of bootstrap replicates to draw.
    progress_bar : bool, default False
        Whether or not to display progress bar.

    Returns
    -------
    output : numpy array
        Bootstrap replicates of MLEs.
    """
    if progress_bar:
        iterator = tqdm.tqdm(range(size))
    else:
        iterator = range(size)

    return np.array([mle_fun(draw_bs_sample(data)) for _ in iterator])