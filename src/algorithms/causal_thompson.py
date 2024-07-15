import json
import numpy as np
from scipy.stats import beta, dirichlet
import sys
import argparse

# Assuming the required modules are present in the utils folder
sys.path.insert(0, 'C:/Users/aybuk/Git/causal-bandits/src/utils')
from src.utils import MAB, SCM


class CausalThompsonSampling:
    def __init__(self, bandit, T, p_obs):
        self.bandit = bandit
        self.T = T
        self.p_obs = p_obs

        # Initialize arrays for successes and failures
        self.s = np.array([1, 1])
        self.f = np.array([1, 1])

        # Initialize probabilities
        self.p_X = np.sum(self.p_obs, axis=0) / np.sum(self.p_obs)
        self.p_Y_X = np.array([self.p_obs[0, 1] / np.sum(self.p_obs[0]), self.p_obs[1, 1] / np.sum(self.p_obs[1])])

        self.z_count = [0, 0]
        self.rewards = np.zeros(T)
        self.cumulative_rewards = np.zeros(T)

    def run(self):
        # Seed P(y | do(X), z) with observations
        self.s[0] = self.p_obs[0, 1]
        self.s[1] = self.p_obs[1, 1]
        self.f[0] = self.p_obs[0, 0]
        self.f[1] = self.p_obs[1, 0]

        for t in range(self.T):
            z = np.random.randint(0, 2)
            z_prime = 1 - z
            self.z_count[z] += 1

            p_Z = self.z_count / np.sum(self.z_count)
            p_YdoX_Z = np.array([
                [self.s[0] / (self.s[0] + self.f[0]), self.s[1] / (self.s[1] + self.f[1])],
                [self.s[1] / (self.s[1] + self.f[1]), self.s[0] / (self.s[0] + self.f[0])]
            ])

            q1 = p_YdoX_Z[z, z_prime]
            q2 = self.p_Y_X[z]

            bias = abs(q1 - q2)
            weight = 1 if np.isnan(bias) else 1 - bias
            w = np.ones(2)
            w[z] = weight

            theta_hat = [beta.rvs(self.s[0], self.f[0]) * w[0], beta.rvs(self.s[1], self.f[1]) * w[1]]
            a_opt = np.argmax(theta_hat)
            reward = np.random.rand() <= theta_hat[a_opt]

            self.s[a_opt] += reward
            self.f[a_opt] += 1 - reward

            self.rewards[t] = reward
            self.cumulative_rewards[t] = self.rewards[:t + 1].sum()

        return self.rewards, self.cumulative_rewards


class OnlineCausalThompsonSampling:
    def __init__(self, bandit, T):
        self.bandit = bandit
        self.T = T

        self.pa_Y = [node for node in bandit.scm.G.predecessors('Y')]
        self.N = len(list(self.pa_Y))

        self.beta_params = np.ones((2 ** self.N, 2))  # Beta(1,1)
        self.dirichlet_params = np.ones((2, 2 ** self.N))  # Dirichlet(1)

        self.S = np.zeros(2 ** self.N)
        self.F = np.zeros(2 ** self.N)

        self.rewards = np.zeros(T)
        self.cumulative_rewards = np.zeros(T)

    def state_to_index(self, state):
        return sum([state[i] * (2 ** i) for i in range(len(state))])

    def run(self):
        for t in range(self.T):
            mu_a = np.zeros(2)  # For storing the expected reward for each action
            for a in range(2):
                pa_Y_Z = dirichlet.rvs(self.dirichlet_params[a])[0]
                p_Y_given_pa_Y_Z = [beta.rvs(self.beta_params[k, 0], self.beta_params[k, 1]) for k in range(2 ** self.N)]
                mu_a[a] = np.sum([p_Y_given_pa_Y_Z[k] * pa_Y_Z[k] for k in range(2 ** self.N)])

            a_t = np.argmax(mu_a)
            X_c = self.bandit.get_observed_values()
            Z_k = set(X_c) | {a_t}
            Y_t = np.random.binomial(1, mu_a[a_t])

            self.dirichlet_params[a_t, list(Z_k)] += 1
            if Y_t == 1:
                self.S[list(Z_k)] += 1
            else:
                self.F[list(Z_k)] += 1

            for k in list(Z_k):
                self.beta_params[k] = [self.S[k] + 1, self.F[k] + 1]

            self.rewards[t] = Y_t
            self.cumulative_rewards[t] = self.rewards[:t + 1].sum()

        return self.rewards, self.cumulative_rewards


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
        cts = CausalThompsonSampling(bandit, T, P_X)
        rewards, cumulative_rewards = cts.run()
    elif args.algorithm == 'OC-TS':
        if not args.pa_Y:
            raise ValueError("Parent nodes of the reward variable must be provided for Online Causal Thompson Sampling.")
        octs = OnlineCausalThompsonSampling(bandit, T)
        rewards, cumulative_rewards = octs.run()
    else:
        raise ValueError("Unsupported algorithm type.")

    print("Rewards obtained: ", rewards)
    print("Cumulative rewards: ", cumulative_rewards)


if __name__ == "__main__":
    main()
