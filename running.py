import numpy as np

global_mapping_ct_function = {
    'uniform': lambda size: np.random.uniform(size=size),
    'normal': lambda size: np.random.normal(size=size),
    'poisson': lambda size: np.random.poisson(size=size),
    'exponential': lambda size: np.random.exponential(size=size),
    'binomial': lambda size: np.random.binomial(n=1, p=0.5, size=size)
}