import numpy as np
from scipy.optimize import minimize, dual_annealing, newton, curve_fit
from config import *
from scipy.special import gammaln
import matplotlib.pyplot as plt


from mcmc import markov_chain_monte_carlo

# Perform a binned maximum likelihood fit to the histogram to extract estimates ˆτπ and ˆτμ for 
# the two mean lifetimes, as well as uncertainties on your two estimates. Compare the results 
# with the values that you put into the simulation

def binned_least_squares_fit(counts, bin_edges, initial_guess, bounds=(-np.inf, np.inf)):

    n = np.sum(counts)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    bin_width = bin_edges[1] - bin_edges[0]

    def N_wrapper(bin_centers, muon_mean_lifetime, pion_mean_lifetime):
        counts = estimate_lifetime_counts(bin_centers, bin_width, n, muon_mean_lifetime, pion_mean_lifetime)
        return counts
    
    popt, pcov = curve_fit(N_wrapper, bin_centers, counts, p0=initial_guess, bounds=bounds)

    MUON_estimate_squares, PION_estimate_squares = popt
    MUON_uncer_squares, PION_uncer_squares = np.sqrt(np.diag(pcov))

    return MUON_estimate_squares, PION_estimate_squares, MUON_uncer_squares, PION_uncer_squares

def binned_maximum_likelihood_fit_2(counts, bin_edges, initial_guess):

    n = np.sum(counts)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    bin_width = bin_edges[1] - bin_edges[0]

    MUON_estimate_like, PION_estimate_like = minimize(binned_maximum_likelihood, [2e-8, 2e-6], args=(counts, bin_centers, bin_width, n)).x

    MUON_uncer_like, PION_uncer_like = 0, 0

    return MUON_estimate_like, PION_estimate_like, MUON_uncer_like, PION_uncer_like

def binned_maximum_likelihood_fit(counts, bin_edges, initial_guess=[1, 1]):
    """
    Perform a binned maximum likelihood fit to the histogram to extract estimates for the mean lifetimes.
    """

    # compute the bin centers and bin widths to do the minimization faster
    n = np.sum(counts)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    bin_width = bin_edges[1] - bin_edges[0]
    bounds = [(1e-7, 1e-5), (1e-9, 1e-7)]
    
    
    global_result = dual_annealing(
        binned_maximum_likelihood,
        bounds=[(1e-7, 1e-5), (1e-9, 1e-7)], #bounds=[(1e-9, 1e-7), (1e-7, 1e-5)] die sind falsch rum 
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
        max_iterations=1000,
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
    
    print(get_uncertainties_2d(min_params,
        counts,
        bin_centers,
        bin_width,
        n,
        min_nll))
    
    muon_pull = pull(muon_estimate, MUON_MEAN_LIFETIME, muon_uncertainty)
    pion_pull = pull(pion_estimate, PION_MEAN_LIFETIME, pion_uncertainty)
    
    return muon_estimate, muon_uncertainty, muon_pull, pion_estimate, pion_uncertainty, pion_pull

def get_uncertainties(estimate, counts, bin_centers, bin_width, n, nll_min):
    muon_estimate, pion_estimate = estimate
    best_params = estimate
    delta = 0.5
    
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


def get_uncertainties_2d(estimate, counts, bin_centers, bin_width, n, nll_min):
    delta = 0.5  # 1-sigma for 2D likelihood

    def nll(params):
        return binned_maximum_likelihood(params, counts, bin_centers, bin_width, n)

    # grid search around best estimate to find contour where NLL = nll_min + delta
    n_points = 2500
    muon_vals = np.linspace(estimate[0] * 0.8, estimate[0] * 1.2, n_points)
    pion_vals = np.linspace(estimate[1] * 0.8, estimate[1] * 1.2, n_points)
    
    contour_points = []

    for mu in muon_vals:
        for pi in pion_vals:
            params = (mu, pi)
            current_nll = nll(params)
            if abs(current_nll - nll_min - delta) < 0.05:  # rough threshold
                contour_points.append(params)

    contour_points = np.array(contour_points)
    if contour_points.size == 0:
        raise RuntimeError("No valid points found on the ΔNLL=0.5 contour.")

    muon_uncertainty = np.ptp(contour_points[:, 0]) / 2
    pion_uncertainty = np.ptp(contour_points[:, 1]) / 2

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
    # estimated_counts = np.clip(estimated_counts, 1e-15, None) 
    
    return estimated_counts

def negative_log_likelihood(counts, estimated_counts):
    """
    Calculates the negative log-likelihood between the actual counts and the estimated counts.
    """
    
    # nll = - np.sum(counts * np.log(estimated_counts) - estimated_counts - gammaln(counts + 1))
    nll = - np.sum(counts * np.log(estimated_counts))
    
    return nll

def pull(reconstructed_quantity, generated_quantity, uncertainty):
    pull = (reconstructed_quantity - generated_quantity) / uncertainty
    return pull
