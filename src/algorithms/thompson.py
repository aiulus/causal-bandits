from abc import ABC

import numpy as np
from scipy.stats import beta, invgamma, norm
import argparse
import sys
from pathlib import Path

from src.utils import MAB

path_root = Path(__file__).parents[2]
sys.path.append(str(path_root))


# TODO: REDUNDANT // REMOVE
class Thompson:
    def __init__(self, n_arms, n_rounds, bandit, budget):
        self.n_arms = n_arms
        self.n_rounds = n_rounds
        self.bandit = bandit
        self.budget = budget
        self.costs_per_arm = bandit.costs
        self.remaining_budget = budget
        self.rewards = np.zeros(self.n_rounds)
        self.cumulative_rewards = np.zeros(self.n_rounds)

    def actions_left(self):
        return self.budget is None or self.remaining_budget >= min(self.bandit.costs)

    def pull_arm(self, arm_index):
        reward = self.bandit.pull_arm(arm_index)
        if self.budget is not None and self.actions_left():
            self.remaining_budget -= self.costs[arm_index]
        return reward

    def run(self):
        if not self.actions_left():
            print("Insufficient budget to run the simulation.")
            return self.rewards, self.cumulative_rewards
        t_max = 0
        for t in range(self.n_rounds):
            t_max += 1
            if not self.actions_left():
                break
            reward, _ = self.simulate_single_pull()
            self.rewards[t] = reward
            self.cumulative_rewards[t] = self.rewards[:t + 1].sum()
        print(f"Total reward after {t_max} rounds: {self.rewards.sum()}")
        return self.rewards, self.cumulative_rewards

    def simulate_single_pull(self):
        raise NotImplementedError("Single-action simulation must be implemented by the calling subclass.")

    def set_budget(self, budget):
        self.budget = budget


class ThompsonSamplingBernoulli(Thompson):
    def __init__(self, n_arms, n_rounds, bandit, budget):
        super().__init__(n_arms, n_rounds, bandit, budget)
        self.successes = np.zeros(self.n_arms)
        self.failures = np.zeros(self.n_arms)
        self.rewards = np.zeros(self.n_rounds)
        self.cumulative_rewards = np.zeros(self.n_rounds)

    def actions_left(self):
        return self.budget is None or self.remaining_budget >= min(self.costs_per_arm)

    def pull_arm(self, arm_index):
        reward = self.bandit.pull_arm(arm_index)
        if reward is not None:
            if self.budget is not None:
                self.remaining_budget -= self.costs_per_arm[arm_index]
        return reward

    def simulate_single_pull(self):
        if not self.actions_left():
            return None, None

        sampled_values = [beta.rvs(a=1 + self.successes[i], b=1 + self.failures[i]) for i in range(self.n_arms)]
        a_opt = np.argmax(sampled_values)
        reward = self.bandit.pull_arm(a_opt)

        if reward == 1:
            self.successes[a_opt] += 1
        else:
            self.failures[a_opt] += 1

        return reward, a_opt


class ThompsonSamplingGaussian(Thompson):
    def __init__(self, n_arms, n_rounds, bandit, budget):
        super().__init__(n_arms, n_rounds, bandit, budget)
        self.alpha = np.ones(self.n_arms)
        self.beta = np.ones(self.n_arms)
        self.mu = np.zeros(self.n_arms)
        self.lamb = np.ones(self.n_arms)

    def simulate_single_pull(self):
        if not self.actions_left():
            return None, None
        sampled_means = np.zeros(self.n_arms)

        for i in range(self.n_arms):
            sampled_var = invgamma.rvs(a=self.alpha[i], scale=self.beta[i])
            sampled_means[i] = norm.rvs(loc=self.mu[i], scale=np.sqrt(sampled_var / self.lamb[i]))

        a_opt = np.argmax(sampled_means)
        print(f"ARM PLAYED: {a_opt}")
        reward = self.bandit.pull_arm(a_opt)
        print(f"REWARD: {reward}")  # Debug statement

        if reward is not None:
            self.lamb[a_opt] += 1
            self.mu[a_opt] = (self.mu[a_opt] * (self.lamb[a_opt] - 1) + reward) / self.lamb[a_opt]
            self.alpha[a_opt] += 0.5
            self.beta[a_opt] += 0.5 * (reward - self.mu[a_opt]) ** 2 / self.lamb[a_opt]

        return reward, a_opt


class ThompsonSamplingLinear(Thompson, ABC):
    def __init__(self, n_arms, n_rounds, bandit, budget=None):
        super().__init__(n_arms, n_rounds, bandit, budget)

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


def main():
    parser = argparse.ArgumentParser(description="Run a multi-armed bandit problem with Thompson Sampling.")
    parser.add_argument('--arms', type=int, required=True, help="Number of arms in the bandit problem")
    parser.add_argument('--rounds', type=int, default=1000, help="Number of rounds to run the simulation.")
    parser.add_argument('--bandit_type', type=str, choices=['bernoulli', 'gaussian', 'linear'], required=True,
                        help="Type of bandit problem, currently supported: Bernoulli, Gaussian, Linear")
    parser.add_argument('--p', nargs='+', type=float, help="True probabilities of each arm (for Bernoulli bandit).")
    parser.add_argument('--means', nargs='+', type=float, help="True means of each arm (for Gaussian bandit).")
    parser.add_argument('--vars', nargs='+', type=float, help="True variances of each arm (for Gaussian bandit).")
    parser.add_argument('--theta', nargs='+', type=float, help="True parameter vector for the Linear bandit.")
    parser.add_argument('--noise-level', type=float, default=0.1,
                        help="Standard deviation of the Gaussian noise (for Linear bandit).")
    parser.add_argument('--context-dim', type=int, help="Dimensionality of the context vectors (for Linear bandit).")

    args = parser.parse_args()

    try:
        if args.bandit_type == 'bernoulli':
            bandit = MAB.Bernoulli_MAB(args.arms, args.p)
            ts = ThompsonSamplingBernoulli(args.arms, args.rounds, bandit)
        elif args.bandit_type == 'gaussian':
            if not args.means or not args.vars:
                raise ValueError("True means and true variances must be provided for Gaussian bandit.")
            bandit = MAB.Gaussian_MAB(args.arms, args.means, args.vars)
            ts = ThompsonSamplingGaussian(args.arms, args.rounds, bandit)
        elif args.bandit_type == 'linear':
            if not args.theta or not args.context_dim:
                raise ValueError("True theta and context dimension must be provided for Linear bandit.")
            bandit = MAB.Linear_MAB(args.arms, args.context_dim, np.array(args.theta), args.noise_level)
            ts = ThompsonSamplingLinear(args.arms, args.rounds, bandit)
        else:
            raise ValueError("Unsupported Bandit Type.")

        rewards, cumulative_rewards = ts.run()

    except Exception as e:
        print(f"Error: {e}")
        parser.print_help()


if __name__ == "__main__":
    main()
