from scipy.stats.distributions import norm, bernoulli, beta, gamma, expon
import re

import io_mgmt

config = io_mgmt.configuration_loader()
PATH_GRAPHS = config['PATH_GRAPHS']
PATH_SCM = config['PATH_SCMs']
PATH_PLOTS = config['PATH_PLOTS']
MAX_DEGREE = config['MAX_POLYNOMIAL_DEGREE']
COEFFS = config['COEFFICIENTS']
DISTS = config['DISTS']

def generate_distribution(noise_term):

    dist_type = noise_term[0]
    params = noise_term[1:]

    if dist_type == 'gaussian':
        return lambda x, mu=params[0], sigma=params[1]: norm.rvs(mu, sigma, size=x)
    elif dist_type == 'bernoulli':
        return lambda x, p=params[0]: bernoulli.rvs(p, size=x)
    elif dist_type == 'exponential':
        return lambda x, lam=params[0]: expon.rvs(scale=1 / lam, size=x)
    else:
        raise ValueError(f"Unsupported distribution type:{dist_type}")




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