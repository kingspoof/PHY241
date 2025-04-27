import numpy as np
from scipy.optimize import minimize, dual_annealing
from config import *
from scipy.special import gammaln

# Perform a binned maximum likelihood fit to the histogram to extract estimates ˆτπ and ˆτμ for 
# the two mean lifetimes, as well as uncertainties on your two estimates. Compare the results 
# with the values that you put into the simulation


def binned_maximum_likelihood_fit(counts, bin_edges, initial_guess=[1, 1]):
    """
    Perform a binned maximum likelihood fit to the histogram to extract estimates for the mean lifetimes.
    """

    # compute the bin centers and bin widths to do the minimization faster
    n = np.sum(counts)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    bin_width = bin_edges[1] - bin_edges[0]
    
    
    print(binned_maximum_likelihood([MUON_MEAN_LIFETIME, PION_MEAN_LIFETIME], counts, bin_centers, bin_width, n))
    
    #result = minimize(
    #    binned_maximum_likelihood,
    #    x0=initial_guess,
    #    args=(counts, bin_centers, bin_width, n),
    #    method='L-BFGS-B',
    #    bounds=[(1e-9, 1e-7), (1e-7, 1e-5)]
    #)
    
    result = dual_annealing(
        binned_maximum_likelihood,
        bounds=[(1e-9, 1e-7), (1e-7, 1e-5)],
        args=(counts, bin_centers, bin_width, n),
        maxiter=10000,
    )
    
    next_parameters = result.x
    result = minimize(
        binned_maximum_likelihood,
        next_parameters,
        method='L-BFGS-B',
        bounds=[(1e-9, 1e-7), (1e-7, 1e-5)],
        args=(counts, bin_centers, bin_width, n),
    )
    
    # Assuming 'result' contains the optimization result
    best_fit_values = result.x
    hess_inv = result.hess_inv  # Inverse Hessian

    # Since it's a 'LbfgsInvHessProduct', convert it to a matrix (if needed) and estimate uncertainties
    uncertainties = np.sqrt(np.diag(hess_inv.todense()))  # If it's sparse, convert to dense matrix first

    # Print the values with their respective uncertainties
    for i, (value, uncertainty) in enumerate(zip(best_fit_values, uncertainties)):
        print(f"Parameter {i+1}: {value:.4e} ± {uncertainty:.4e}")


    return result

    
    #eturn muon_mean_lifetime, pion_mean_lifetime, muon_mean_lifetime_uncertainty, pion_mean_lifetime_uncertainty


def binned_maximum_likelihood(params, counts, bin_centers, bin_width, total_counts):

    muon_mean_lifetime, pion_mean_lifetime = params

    # estimate the lifetimes using the given lifetimes as well as the bin centers and bin widths
    estimated_lifetime_counts = estimate_lifetime_counts(
        bin_centers,
        bin_width,
        total_counts,
        muon_mean_lifetime,
        pion_mean_lifetime
    )

    # calculate the negative log-likelihood
    nll = negative_log_likelihood(counts, estimated_lifetime_counts)

    return nll


def estimate_lifetime_counts(bin_centers, bin_width, total_counts, muon_mean_lifetime, pion_mean_lifetime):
    """
    Estimate the expected counts in each bin based on the mean lifetimes.
    """

    estimated_counts = N(bin_centers, total_counts, muon_mean_lifetime, pion_mean_lifetime) * bin_width
    estimated_counts = np.clip(estimated_counts, 1e-15, None) 
    
    return estimated_counts

def negative_log_likelihood(counts, estimated_counts):
    """
    Calculates the negative log-likelihood between the actual counts and the estimated counts.
    """
    
    nll = - np.sum(counts * np.log(estimated_counts) - estimated_counts - gammaln(sum(counts)))
    return nll

