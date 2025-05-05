from numpy import exp

MUON_MEAN_LIFETIME = 2.2e-6
PION_MEAN_LIFETIME = 2.6033e-8

def N(t, N0, muon_mean_lifetime=MUON_MEAN_LIFETIME, pion_mean_lifetime=PION_MEAN_LIFETIME):
    """
    Equation 1 from the documentation
    here N0 is a normalization constant
    """
    
    first_part = N0 / (muon_mean_lifetime - pion_mean_lifetime)
    second_part = exp(- t / muon_mean_lifetime) - exp(- t / pion_mean_lifetime)
    
    # some combinations if muon and pion lifetimes will result in invalid values for the count
    return first_part * second_part
