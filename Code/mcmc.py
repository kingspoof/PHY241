import numpy as np


def markov_chain_monte_carlo(
    likelihood_function,
    parameters,
    data,
    covariance_matrix,
    max_iterations=1000,
    acceprance_depressor=0.1,
    step_multiplier=1.0,
):
    nll_list = []
    param_list = []
    acceptance_list = []
    
    current_params = parameters
    current_likelihood = likelihood_function(current_params, *data)
    
    
    for _ in range(max_iterations):
        
        # propose a new position
        new_params = current_params + np.random.multivariate_normal(mean=[0,0], cov=covariance_matrix) * step_multiplier
        new_likelihood = likelihood_function(new_params, *data)
        
        if acceptance_method(new_likelihood, current_likelihood, acceprance_depressor):
            acceptance_list.append(1)
            current_params, current_likelihood = new_params, new_likelihood
        else:
            acceptance_list.append(0)
            
        nll_list.append(current_likelihood)
        param_list.append(current_params)
    
    min_index = np.argmin(nll_list)
    return param_list[min_index], nll_list[min_index], param_list



def acceptance_method(new, old, acceptance_depressor):
    if(new < old):
        return True

    acceptance = (new / old) * acceptance_depressor
    return np.random.rand() < acceptance
    