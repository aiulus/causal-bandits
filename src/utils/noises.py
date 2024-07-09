import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats.distributions import norm, bernoulli, beta, gamma, expon
import os
import json
import argparse
import SCM
import re

# Set target destination for .json files containing graph structures
PATH_GRAPHS = "../../outputs/graphs"
PATH_SCM = "../../outputs/SCMs"
PATH_PLOTS = "../../outputs/plots"
PATH_DATA = "../../outputs/data"

def generate_distributions(variables, distr_type: str, params=None):
    # TODO: make sure that 'params' is generated as a list of lists [[]]
    # TODO: currently no way of passing a 'params' array in a manageable way
    p_n = {}

    # If the parameters are not specified, use standard values
    if params is None:
        params = {
            'gaussian': {'mu': 0, 'sigma': 1},
            'bernoulli': {'p': 0.5},
            'exp': {'lam': 1.0}
        }

    # TODO: too nested
    for x_i in variables:
        if distr_type == 'gaussian':
            mu, sigma = params['gaussian']['mu'], params['gaussian']['sigma']
            p_n[x_i] = lambda x, mu=mu, sigma=sigma: norm.rvs(mu, sigma, size=x)
        elif distr_type == 'bernoulli':
            p = params['bernoulli']['p']
            p_n[x_i] = lambda x, p=p: bernoulli.rvs(p, size=x)
        elif distr_type == 'exp':
            lam = params['exp']['lam']
            p_n[x_i] = lambda x, lam=lam: expon.rvs(scale=1 / lam, size=x)
        else:
            raise ValueError(f"Unsupported distribution type:{distr_type}")

    return p_n

def parse_noise_string(noise_str):
    dist_type_map = {
        'N': 'gaussian',
        'Exp': 'exponential',
        'Ber': 'bernoulli'
    }

    pattern = r'([A-Za-z]+)\(([^)]+)\)'
    match = re.match(pattern, noise_str)
    if not match:
        raise ValueError(f"Invalid distribution format: {noise_str}")

    noise_type, params = match.groups()
    params = [float(x) for x in params.split(',')]

    if noise_type not in dist_type_map:
        raise ValueError(f"Unsupported distrbution type: {noise_type}")

    return {"type": dist_type_map[noise_type], "params": params}

def parse_noise(noise):
    """
    Parse noise distribution strings into a dictionary format.
    Example: "N(0,1) --> {"type": "gaussian", "params": [0,1]}
    """

    noise_dict = {}

    if isinstance(noise, list):
        for i, noise_str in enumerate(noise):
            noise_dict[i] = parse_noise_string(noise_str)
    else:
        noise_dict[0] = parse_noise_string(noise)

    return noise_dict
