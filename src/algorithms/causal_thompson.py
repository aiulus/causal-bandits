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

    # Seed P(y | do(X), z) with observations
    s[0, 0] = p_obs[0, 1]
    s[1, 1] = p_obs[1, 1]
    f[0, 0] = p_obs[0, 0]
    f[1, 1] = p_obs[1, 0]

    actions = np.zeros(T)
    rewards = np.zeros(T)
    probs = np.zeros(T)
    conds = np.zeros((T, 4))
    theta = np.zeros((2, 4))
    for i in range(2):
        for j in range(4):
            interventions = {'X1': i, 'X2': j // 2}
            theta[i, j] = bandit.expected_reward(interventions)

    for t in range(T):
        b, d, z = np.random.randint(0, 2, size=3)
        z_prime = 1 - z
        covar_index = b + d * 2
        conds[t, covar_index] +=1

        z_count[z] += 1
        p_Z = [z_count[0] / sum(z_count), z_count[1] / sum(z_count)]
        p_YdoX_Z = np.array([[s[0, 0] / (s[0, 0] + f[0, 0]), s[0, 1] / (s[0, 1] + f[0, 1])],
                           [s[1, 0] / (s[1, 0] + f[1, 0]), s[1, 1] / (s[1, 1] + f[1, 1])]])

        # Q1 = E(y_x' | x) [Counter-intuition]
        q1 = p_YdoX_Z[z, z_prime]
        # Q2 = E(y_x | x) [intuition]
        q2 = p_Y_X[z]

        # Weighting
        bias = abs(q1 - q2)
        w = np.ones(2)
        weight = 1 if np.isnan(bias) else 1 - bias
        if q1 > q2:
            w[z] = weight
        else:
            w[z_prime] = weight

        # Find optimal action
        theta_hat = [beta.rvs(s[z, 0], f[z, 0]) * w[0], beta.rvs(s[z, 1], f[z, 1]) * w[1]]
        a_opt = np.argmax(theta_hat)
        theta = theta[a_opt, covar_index]

        reward = np.random.rand() <= theta

        s[z, a_opt] += reward
        f[z, a_opt] += 1 - reward

        actions[t] = a_opt
        rewards[t] = reward
        best_action = np.argmax([theta[0, covar_index], theta[1, covar_index]])
        probs[t] = a_opt == best_action

    return actions, rewards, probs, conds