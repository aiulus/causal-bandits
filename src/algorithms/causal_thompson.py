import numpy as np
from scipy.stats import beta
import sys
sys.path.insert(0, 'C:/Users/aybuk/Git/causal-bandits/src/utils')

"""
Implements the Causal Thompson Sampling algorithm as defined in
@inproceedings{NIPS2015_795c7a7a,
 author = {Bareinboim, Elias and Forney, Andrew and Pearl, Judea},
 booktitle = {Advances in Neural Information Processing Systems},
 editor = {C. Cortes and N. Lawrence and D. Lee and M. Sugiyama and R. Garnett},
 pages = {},
 publisher = {Curran Associates, Inc.},
 title = {Bandits with Unobserved Confounders: A Causal Approach},
 url = {https://proceedings.neurips.cc/paper_files/paper/2015/file/795c7a7a5ec6b460ec00c5841019b9e9-Paper.pdf},
 volume = {28},
 year = {2015}
}
"""
def causal_thompson_sampling(bandit, T, p_obs):
    """
    Runs Causal Thompson Sampling on a given Causal Bandit instance

    :param bandit: An instance of Causal_MAB
    :param T: Time horizon (number of rounds to run the simulation)
    :param p_obs: Observed probability distribution for the arms
    :return: Actions, Rewards, Probabilities, Conditions
    """
    # TODO: Algorithm description

    """
    Initialize arrays to track the number of successes and failures for each arm and context.
    Initialized with ones to use Beta(1,1) as a non-informative prior.
    """
    s = np.array([1, 1], [1, 1])
    f = np.array([1, 1], [1, 1])

    """
    Retrieve and normalize probabilities of contexts"""
    p_X = s[sum(p_obs[0]) / sum(p_obs), sum(p_obs[1]) / sum(p_obs)]
    """
    Compute the conditional probability of Y = 1 given context 0/1:
        p_Y_X[0]: Probabilty of Y = 1 given context 0
        p_Y_X[1]: Probabilty of Y = 1 given context 1
    """
    p_Y_X = [p_obs[0, 1] / sum(p_obs[0]), p_obs[1, 1] /sum(p_obs[1])]

    # Initialize vector for tracking the number of times each context is observed
    z_count = [0, 0]