import sys
from pathlib import Path

import numpy as np
from scipy.stats import beta, invgamma, norm

from src.utils import MAB, SCM

path_root = Path(__file__).parents[2]
sys.path.append(str(path_root))


class ThompsonSamplingBernoulli:
    def __init__(self, n_arms, n_rounds, bandit):
        self.n_arms = n_arms
        self.n_rounds = n_rounds
        self.bandit = bandit

    def run(self):
        if self.n_arms != self.bandit.get_arms():
            raise ValueError("Number of arms does not match with the given MAB instance.")

        successes = np.zeros(self.n_arms)
        failures = np.zeros(self.n_arms)
        rewards = np.zeros(self.n_rounds)
        cumulative_rewards = np.zeros(self.n_rounds)

        for round in range(self.n_rounds):
            sampled_values = [beta.rvs(a=1 + successes[i], b=1 + failures[i]) for i in range(self.n_arms)]
            a_opt = np.argmax(sampled_values)
            reward = self.bandit.pull_arm(a_opt)

            if reward == 1:
                successes[a_opt] += 1
            else:
                failures[a_opt] += 1

            rewards[round] = reward
            cumulative_rewards[round] = rewards[:round + 1].sum()

        print(f"Total reward after {self.n_rounds} rounds: {rewards.sum()}")
        print(f"Successes: {successes}")
        print(f"Failures: {failures}")

        return rewards, cumulative_rewards


class ThompsonSamplingGaussian:
    def __init__(self, n_arms, n_rounds, bandit):
        self.n_arms = n_arms
        self.n_rounds = n_rounds
        self.bandit = bandit

    def run(self):
        alpha = np.ones(self.n_arms)
        beta = np.ones(self.n_arms)
        mu = np.zeros(self.n_arms)
        lamb = np.ones(self.n_arms)

        rewards = np.zeros(self.n_rounds)
        cumulative_rewards = np.zeros(self.n_rounds)

        for round in range(self.n_rounds):
            sampled_means = np.zeros(self.n_arms)

            for i in range(self.n_arms):
                sampled_var = invgamma.rvs(a=alpha[i], scale=beta[i])
                sampled_means[i] = norm.rvs(loc=mu[i], scale=np.sqrt(sampled_var / lamb[i]))

            a_opt = np.argmax(sampled_means)
            reward = self.bandit.pull_arm(a_opt)

            lamb[a_opt] += 1
            mu[a_opt] = (mu[a_opt] * (lamb[a_opt] - 1) + reward) / lamb[a_opt]
            alpha[a_opt] += 0.5
            beta[a_opt] += 0.5 * (reward - mu[a_opt]) ** 2 / lamb[a_opt]

            rewards[round] = reward
            cumulative_rewards[round] = rewards[:round + 1].sum()

        print(f"Total reward after {self.n_rounds} rounds: {rewards.sum()}")
        print(f"Sampled means: {sampled_means}")

        return rewards, cumulative_rewards


class ThompsonSamplingLinear:
    def __init__(self, n_arms, n_rounds, bandit):
        self.n_arms = n_arms
        self.n_rounds = n_rounds
        self.bandit = bandit

    def run(self):
        context_dim = self.bandit.context_dim
        alpha = 1.0
        beta = 1.0
        A = np.eye(context_dim) / alpha
        b = np.zeros(context_dim)

        rewards = np.zeros(self.n_rounds)
        cumulative_rewards = np.zeros(self.n_rounds)

        for round in range(self.n_rounds):
            theta_sample = np.random.multivariate_normal(np.linalg.solve(A, b), np.linalg.inv(A))
            sampled_rewards = np.zeros(self.n_arms)

            for i in range(self.n_arms):
                context = self.bandit.get_context(i)
                sampled_rewards[i] = np.dot(context, theta_sample)

            a_opt = np.argmax(sampled_rewards)
            reward = self.bandit.pull_arm(a_opt)

            A += np.outer(context, context) / beta
            b += context * reward / beta

            rewards[round] = reward
            cumulative_rewards[round] = rewards[:round + 1].sum()

        print(f"Total reward after {self.n_rounds} rounds: {rewards.sum()}")
        return rewards, cumulative_rewards

def UCB1(bandit, n_rounds):
    """

    :param bandit:
    :param n_rounds:
    :return: Total reward after all rounds, counts of pulls for each arm
    """
    n_arms = bandit.get_arms()
    counts = np.zeros(n_arms) # Number of times each arm has been played
    means = np.zeros(n_arms) # Mean reward for each arm
    rewards = np.zeros(n_rounds) # Rewards obtained in each round
    cumulative_rewards = np.zeros(n_rounds)

    for t in range(n_rounds):
        if t < n_arms:
            # Play each arm once if time horizon allows for that
            a_t = t
        else:
            # Compute upper confidence bounds
            ucb_vals = means + np.sqrt((2 * np.log(t)) / counts)
            a_t = np.argmax(ucb_vals)

        # Pull the chosen arm
        r_t = bandit.pull_arm(a_t)

        # Update counts and mean values
        counts[a_t] += 1
        n = counts[a_t]
        mu_t = means[a_t]
        means[a_t] = ((n - 1) / n) * mu_t + (1 / n) * r_t

        rewards[t] = r_t
        cumulative_rewards[t] = rewards[:t + 1].sum()

    print(f"Total reward after {n_rounds} rounds: {rewards.sum()}")
    print(f"Number of times each arm was played: {counts}")

    return rewards, cumulative_rewards

