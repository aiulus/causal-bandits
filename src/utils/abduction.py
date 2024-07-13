import numpy as np
import scipy.stats as stats

import SCM, noises

"""
Defines methods related to constructing an observationally constrained SCM.
"""

noise_dict = {
    "X1": [
        "gaussian",
        2.0,
        5.0
    ],
    "X2": [
        "gaussian",
        2.0,
        5.0
    ],
    "X3": [
        "gaussian",
        2.0,
        5.0
    ],
    "Y": [
        "gaussian",
        2.0,
        5.0
    ]
}


def rejection_sampling(scm, observations, noise_term):
    """
    Computes a conditional joint noise distribution N|(X'=x') where x' is a set of observations.
    :param scm:
    :param observations:
    :param noise_dict:
    :return:
    """
    accepted_samples = []
    n_samples = len(observations)

    noise_pdf = noises.generate_distribution(noise_term)  # Map string representation to lambda function
    norm = 1  # Initialize normalization constant

    # TODO


def generate_random_observations():
    return 0
