from numpy import exp, log, array
from numpy.random import random, normal
from config import *

T_MAX = MUON_MEAN_LIFETIME * PION_MEAN_LIFETIME * log(MUON_MEAN_LIFETIME/PION_MEAN_LIFETIME) / (MUON_MEAN_LIFETIME - PION_MEAN_LIFETIME)


def Random_number_from_distribution(N0, t_end, N_MAX):
    t_rand = random() * t_end
    N_rand = random() * N_MAX

    while N_rand > N(t_rand, N0):
        t_rand = random() * t_end
        N_rand = random() * N_MAX

    return [t_rand, N_rand]    

def simulate_decay_times(N0, t_end):
    N_MAX = N(T_MAX, N0)

    points = array([Random_number_from_distribution(N0, t_end, N_MAX) for _ in range(N0)])

    return points

def smear_decay_times(decay_times, sigma, mu=0):
    times = decay_times[:,0]
    noise = normal(loc=mu, scale=sigma, size=times.shape)
    decay_times[:,0] = decay_times[:,0] + noise
    return decay_times