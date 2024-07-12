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
# Command line strings for the surrently supported set of distributions
DISTS = ['N', 'Exp', 'Ber']


def generate_distributions(noise_dict):
    p_n = {}

    # TODO: too nested
    for node, noise_spec in noise_dict.items():
        dist_type = noise_spec['type']
        params = noise_spec['params']

        if dist_type == 'gaussian':
            p_n[node] = lambda x, mu=params[0], sigma=params[1]: norm.rvs(mu, sigma, size=x)
        elif dist_type == 'bernoulli':
            p_n[node] = lambda x, p=params[0]: bernoulli.rvs(p, size=x)
        elif dist_type == 'exp':
            p_n[node] = lambda x, lam=params[0]: expon.rvs(scale=1 / lam, size=x)
        else:
            raise ValueError(f"Unsupported distribution type:{dist_type}")

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

    return dist_type_map[noise_type], params


def parse_noise(noise, nodes):
    """
    Parse noise distribution strings into a dictionary format.
    Example: "N(0,1) --> {"X_{node_id}": ("N", 0, 1)}
    """

    if isinstance(noise, str):
        noise = [noise]

    num_nodes = len(nodes)
    counts = {distr_str: noise.count(distr_str) for distr_str in DISTS}
    arg_count = len(noise)

    if arg_count == 1:
        noise = noise * num_nodes
    elif arg_count != num_nodes:
        raise ValueError(f"Expected either 1 or {num_nodes} noise distributions, but got {arg_count}: \n {noise}")

    noise_dict = {node: parse_noise_string(noise_str) for node, noise_str in zip(nodes, noise)}

    return noise_dict