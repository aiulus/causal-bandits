import json

import numpy as np
from scipy.stats import beta, dirichlet
import sys, argparse
sys.path.insert(0, 'C:/Users/aybuk/Git/causal-bandits/src/utils')
from src.utils import MAB, SCM
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

"""
Online Causal Thompson Sampling Algorithm based on
@inproceedings{Sachidananda2017OnlineLF,
  title={Online Learning for Causal Bandits},
  author={Vin Sachidananda and Emma Brunskill},
  year={2017},
  url={https://api.semanticscholar.org/CorpusID:26215248}
}
"""
def online_causal_TS(bandit, T, pa_Y):
    """

    :param bandit: An instance of Causal_MAB
    :param T: Time horizon
    :param pa_Y: Set containing parent nodes of the reward variable
    """
    # TODO: Describe the algorithm
    N = len(pa_Y)

    # Initialize Beta and Dirichlet distributions
    beta_params = np.ones((2**N, 2)) # Beta(1,1)
    dirichlet_params = np.ones((2, 2^N)) # Dirichlet(1)

    # Initialize vectors to keep track of successes and failures
    S = np.zeros(2**N)
    F = np.zeros(2**N)

    # Initialize lists to track actions and rewards for each time step
    actions = []
    rewards = []

    # Convert a state list of parent variables to an index
    def state_to_index(state):
        return sum([state[i] * (2**i) for i in range(len(state))])

    for t in range(T):
        mu_a = np.zeros(2) # For storing the expected reward for each action
        for a in range(2):
            # TODO: refactor variable names
            pa_Y_Z = dirichlet.rvs(dirichlet_params[a])[0]
            p_Y_given_pa_Y_Z = [beta.rvs(beta_params[k, 0], beta_params[k, 1]) for k in range(2**N)]
            # Calculate the expected reward for action 'a' by summing over the states of the parent variables
            mu_a[a] = np.sum([p_Y_given_pa_Y_Z[k] * pa_Y_Z[k] for k in range(2 ** N)])

        # Choose the action yielding the maximum expected reward
        a_t = np.argmax(mu_a)

        # Sample the observed values for the parent variables
        X_c = bandit.get_observed_values()
        # Determine the state of the parent variables after the intervention
        Z_k = set(X_c) | {a_t}
        # Simulate the reward based on the expected reward for the chosen action
        Y_t = np.random.binomial(1, mu_a[a_t])

        # Update the Dirichlet distribution for the chosen action and state
        dirichlet_params[a_t, list(Z_k)] += 1

        # Update the Beta distribution based on the observed reward
        if Y_t == 1:
            S[list(Z_k)] += 1
        else:
            F[list(Z_k)] += 1

        # Update Beta parameters for each state in Z_k
        for k in list(Z_k):
            beta_params[k] = [S[k] + 1, F[k] + 1]

        actions.append(a_t)
        rewards.append(Y_t)

    return actions, rewards

def main():
    parser = argparse.ArgumentParser(description="Run a causal bandit problem with Thompson Sampling.")
    parser.add_argument('--reward', type=str, required=True, help="Reward variable in SCM.")
    parser.add_argument('--rounds', type=int, default=1000, help="Time horizon.")
    parser.add_argument('--algorithm', type=str, choices=['C-TS', 'OC-TS'])
    parser.add_argument('--obs', nargs='+', type=float, required=True, help='Observed probability distribution for the arms.')
    parser.add_argument('--pa_Y', nargs='+', help="Set containing the parents nodes of the reward variable.")
    parser.add_argument('--json', type=str, help="Path to the JSON file describing the SCM.")

    args = parser.parse_args()

    if args.json:
        try:
            with open(args.json, 'r') as file:
                scm_json = json.load(file)
        except FileNotFoundError:
            print(f"Error: This file {args.json} was not found.")
            return
        except json.JSONDecodeError:
            print("Error: Failed to decode JSON from the provided file.")

    scm = SCM(scm_json)
    Y = args.reward
    T = args.rounds
    P_X = np.array(args.obs).reshape(2, 2)

    bandit = MAB.CausalBandit(scm, Y)

    if args.algorithm == 'C-TS':
        act, rew, probs, cond = causal_thompson_sampling(bandit, P_X)
    elif args.algorithm == 'OC-TS':
        if not args.pa_Y:
            raise ValueError("Parent nodes of the reward variable must be provided for Online Causal Thompson Sampling.")
        pa_Y = args.pa_Y
        act, rew = online_causal_TS(bandit, T, pa_Y)
    else:
        raise ValueError("Unsupported algorithm type.")

    print("Actions taken: ", act)
    print("Rewards obtained: ", rew)

if __name__ == "__main__":
    main()
