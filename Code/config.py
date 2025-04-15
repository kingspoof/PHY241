from numpy import exp, log, array
from numpy.random import random
from scipy.optimize import newton

PION_MEAN_LIFETIME = 2.6033e-8
MUON_MEAN_LIFETIME = 2.1969811e-6

T_MAX = MUON_MEAN_LIFETIME * PION_MEAN_LIFETIME * log(MUON_MEAN_LIFETIME/PION_MEAN_LIFETIME) / (MUON_MEAN_LIFETIME - PION_MEAN_LIFETIME)


def N(t, N0):
    """
    Equation 1 from the documentation
    here N0 is a normalization constant
    """
    first_part = N0 / (MUON_MEAN_LIFETIME - PION_MEAN_LIFETIME)
    second_part = exp(- t / MUON_MEAN_LIFETIME) - exp(- t / PION_MEAN_LIFETIME)
    
    return first_part * second_part

# def N_Wrapper_Distribution(ts, N0, t_end=2e-5):
#     """
#     Function that is always higher than Equation 1 from the documentation
#     here N0 is a normalization constant
#     """
#     N_MAX = N(T_MAX, N0) + 1
#     SLOPE = - N_MAX / (t_end - T_MAX)

#     Ns = array([N_MAX if t < T_MAX*2 else SLOPE*(t-T_MAX*2)+N_MAX for t in ts])

#     return Ns

def Random_number_from_distribution(N0, t_end, N_MAX):
    t_rand = random() * t_end
    N_rand = random() * N_MAX

    while N_rand > N(t_rand, N0):
        t_rand = random() * t_end
        N_rand = random() * N_MAX

    return [t_rand, N_rand]

def Random_Numbers_from_Dist(N0, t_end):
    N_MAX = N(T_MAX, N0)

    points = array([Random_number_from_distribution(N0, t_end, N_MAX) for _ in range(N0)])

    return points

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
