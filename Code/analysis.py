import numpy as np
from config import *

# Perform a binned maximum likelihood fit to the histogram to extract estimates ˆτπ and ˆτμ for 
# the two mean lifetimes, as well as uncertainties on your two estimates. Compare the results 
# with the values that you put into the simulation


def binned_maximum_likelihood(params, counts, bin_edges):
    """
    Perform a binned maximum likelihood fit to the histogram to extract estimates for the mean lifetimes.
    """

    muon_mean_lifetime, pion_mean_lifetime = params

    n = np.sum(counts)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    bin_width = bin_edges[1:] - bin_edges[:-1]

    # calculate the expected counts for each bin using the N function
    estimated_counts = estimate_lifetime_counts(bin_edges, n, muon_mean_lifetime, pion_mean_lifetime)

    # calculate the negative log-likelihood
    nll = negative_log_likelihood(counts, estimated_counts)

    return nll

def estimate_lifetime_counts(bin_edges, total_counts, muon_mean_lifetime, pion_mean_lifetime):
    """
    Estimate the counts for each bin using the N function.
    
    Parameters:
    bin_edges (np.ndarray): The edges of the bins.
    total_counts (int): The total number of counts.
    muon_mean_lifetime (float): The mean lifetime of the muon.
    pion_mean_lifetime (float): The mean lifetime of the pion.
    
    Returns:
    np.ndarray: Estimated counts for each bin.
    """
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    bin_width = bin_edges[1:] - bin_edges[:-1]
    
    estimated_counts = np.zeros_like(bin_centers, dtype=float)
    
    for i, center in enumerate(bin_centers):
        estimated_counts[i] = N(center, total_counts, muon_mean_lifetime, pion_mean_lifetime) * bin_width[i]
    
    return estimated_counts

def negative_log_likelihood(counts, estimated_counts):
    """
    Calculates the negative log-likelihood between the actual counts and the estimated counts.
    """

    estimated_counts = np.clip(estimated_counts, 1e-15, None) # replace 0 with something very small, so we don't get an error
    nll = - np.sum(counts * np.log(estimated_counts) - estimated_counts)
    return nll

