from numpy import exp, nan_to_num, inf
from scipy.optimize import newton

PION_MEAN_LIFETIME = 2.6033e-8
MUON_MEAN_LIFETIME = 2.1969811e-6


def N(t, N0):
    """
    Equation 1 from the documentation
    here N0 is a normalization constant
    """
    first_part = N0 / (MUON_MEAN_LIFETIME - PION_MEAN_LIFETIME)
    second_part = exp(- t / MUON_MEAN_LIFETIME) - exp(- t / PION_MEAN_LIFETIME)
    
    return first_part * second_part

def N_CDF(x,N0):
    """
    The CDF from the Equation 1 from the documentation
    here N0 is a normalization constant
    """
    t1 = MUON_MEAN_LIFETIME
    t2 = PION_MEAN_LIFETIME

    first_part = N0 / (t1 - t2)
    second_part = - (t1 * exp(x / t2) - t2 * exp(x / t1)) * exp(- x / t2 - x / t1) - t2 + t1

    second_part = nan_to_num(second_part, nan=inf)

    return first_part * second_part

def N_Integrated(t, N0):
    top_part = N0 * (PION_MEAN_LIFETIME * exp(-t / PION_MEAN_LIFETIME) - MUON_MEAN_LIFETIME * exp(- t / PION_MEAN_LIFETIME))
    bottom_part = MUON_MEAN_LIFETIME - PION_MEAN_LIFETIME

    return top_part / bottom_part

def N_Integrated_Inverted_Numerical(x, N0):

    def function_wrapper(t):
        """
        Wrapper used for finding any root of a given function
        """
        return N_Integrated(t, N0) - x
    
    root = newton(function_wrapper, 0)
    return root
