import numpy as np
from scipy.optimize import minimize, dual_annealing, newton
from config import *
from scipy.special import gammaln
import matplotlib.pyplot as plt


from mcmc import markov_chain_monte_carlo

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
    
    
    global_result = dual_annealing(
        binned_maximum_likelihood,
        bounds=[(1e-9, 1e-7), (1e-7, 1e-5)],
        args=(counts, bin_centers, bin_width, n),
        maxiter=10000,
    )

    # use a finer grid around the global minimum to get a better estimate of the parameters
    local_range = 0.01
    bounds = [
        ((1 - local_range) * global_result.x[0], (1 + local_range) * global_result.x[0]),
        ((1 - local_range) * global_result.x[1], (1 + local_range) * global_result.x[1])
    ]
    
    local_result = minimize(
        binned_maximum_likelihood,
        x0=global_result.x,
        args=(counts, bin_centers, bin_width, n),
        method='trust-constr',
        bounds=bounds,
        options={'gtol': 1e-10, 'xtol': 1e-10}
    )
    
    # lastly find the absolute best minimum using markov chain monte carlo
    min_params, min_nll, _ = markov_chain_monte_carlo(
        binned_maximum_likelihood,
        local_result.x,
        data=(counts, bin_centers, bin_width, n),
        covariance_matrix=np.diag([1, 1]),
        max_iterations=100000,
        step_multiplier=1e-10
    )
    
    muon_estimate, pion_estimate = min_params
    muon_uncertainty, pion_uncertainty = get_uncertainties(
        min_params,
        counts,
        bin_centers,
        bin_width,
        n,
        min_nll
    )
    
    return muon_estimate, pion_estimate, muon_uncertainty, pion_uncertainty

def get_uncertainties(estimate, counts, bin_centers, bin_width, n, nll_min):
    muon_estimate, pion_estimate = estimate
    best_params = estimate
    delta = 1.15
    
    # the uncertainties are given by the nll where it changes by the delta abount
    def nll_muon(muon_lifetime):
        return binned_maximum_likelihood(
            (muon_lifetime, best_params[1]),
            counts,
            bin_centers,
            bin_width,
            n
        ) - nll_min - delta
    
    def nll_pion(pion_lifetime):
        return binned_maximum_likelihood(
            (best_params[0], pion_lifetime),
            counts,
            bin_centers,
            bin_width,
            n
        ) - nll_min - delta
    
    # now we can use newtons method to find the roots of the nll functions
    muon_uncertainty = newton(nll_muon, muon_estimate * 1.2, maxiter=1000)
    pion_uncertainty = newton(nll_pion, pion_estimate * 1.2, maxiter=1000)
    
    return muon_uncertainty, pion_uncertainty
    
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
    
    nll = - np.sum(counts * np.log(estimated_counts) - estimated_counts - gammaln(counts + 1))
    return nll

