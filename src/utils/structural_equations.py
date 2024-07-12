import argparse
import json
import sys
import os

import networkx as nx
import numpy as np
from scipy.stats import norm, bernoulli, expon
import matplotlib.pyplot as plt
import re

sys.path.insert(0, 'C:/Users/aybuk/Git/causal-bandits/src/utils')
import graph_generator, plots

# Set target destination for .json files containing graph structures
PATH_GRAPHS = "../../outputs/graphs"
PATH_SCM = "../../outputs/SCMs"
PATH_PLOTS = "../../outputs/plots"
MAX_DEGREE = 3  # For polynomial function generation
# Set of coefficients to choose from
# PRIMES = [-11, -7, -5, -3, -2, 2, 3, 5, 7, 11]
PRIMES = [-2, -1.5, -1, -0.5, 0.5, 1, 1.5, 2]

# TODO: noises should appear as arguments in lambdas
def generate_linear_function(parents, noise, coeffs):
    """

    :param parents: immediate predecessors (pa(X_i)) of node X_i in graph G
    :param coeffs: coefficient vector of length |pa(X_i)|
    :return: Linear function f_i(pa(X_i), N_i) = f(pa(X_i)) + N_i as a string.
    """
    terms = [f"{coeffs[i]} * {parent}" for i, parent in enumerate(parents)]
    terms.append(noise)
    function = f"lambda {', '.join(parents)}: " + " + ".join(terms)

    return function


def generate_polynomial(parents, noise, coeffs, degrees):
    """
    Generate f_i(pa(X_i), N_i) = polynomial(pa(X_i), a_j, e_j) + N_i
    :param parents: pa(X_i)
    :param coeffs: a_i
    :param degrees: e_i in f_i(pa(X_i), N_i) = N_i + sum(a_j * X_i^{e_j} for j in pa(X_i))
    :return: Linear function f_i(pa(X_i), N_i) = f(pa(X_i)) + N_i as a string.
    """
    terms = [f"{coeffs[i]}*{parent}**{degrees[i]}" for i, parent in enumerate(parents)]
    terms.append(noise)
    function = f"lambda {', '.join(parents)}: " + " + ".join(terms)
    return function


def generate_functions(graph, noise_vars, funct_type='linear'):
    # TODO
    """

    :param graph:
    :param noise_vars: Noise variables N_i specified as f"lambda ... : ..."
    :param funct_type:
    :return:
    """
    functions = {}
    functions.keys()
    for node in graph.nodes:
        parents = list(graph.predecessors(node))
        if funct_type == 'linear':
            # Randomly pick the coefficients
            coeffs = np.random.choice(PRIMES, size=len(parents))
            # coeffs = np.random.randn(len(parents))
            # functions[node] = generate_linear_function(parents, noise_vars[node], coeffs)
            functions[node] = generate_linear_function(parents, f"N_{node}", coeffs)
        elif funct_type == 'polynomial':
            degrees = np.random.randint(1, MAX_DEGREE + 1, size=len(parents))
            coeffs = np.random.choice(PRIMES, size=len(parents))
            # coeffs = np.random.randn(len(parents))
            # functions[node] = generate_polynomial(parents, noise_vars[node], coeffs, degrees)
            functions[node] = generate_polynomial(parents, f"N_{node}", coeffs, degrees)
        else:
            raise ValueError(f"Unsupported function type: {funct_type}. Use 'linear' or 'polynomial'.")

    return functions


def parse_functions(func_dict):
    """
    Parse strings into lambda functions.
    """
    parsed_functions = {}
    for key, func_str in func_dict.items():
        parsed_functions[key] = eval(func_str)
    return parsed_functions
