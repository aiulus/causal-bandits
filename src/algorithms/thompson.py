import numpy as np
from scipy.stats import beta, invgamma, norm
import argparse
import sys
from pathlib import Path

from src.utils import MAB


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


def main():
    parser = argparse.ArgumentParser(description="Run a multi-armed bandit problem with Thompson Sampling.")
    parser.add_argument('--arms', type=int, required=True, help="Number of arms in the bandit problem")
    parser.add_argument('--rounds', type=int, default=1000, help="Number of rounds to run the simulation.")
    parser.add_argument('--bandit_type', type=str, choices=['bernoulli', 'gaussian', 'linear'], required=True, help="Type of bandit problem, currently supported: Bernoulli, Gaussian, Linear")
    parser.add_argument('--p', nargs='+', type=float, help="True probabilities of each arm (for Bernoulli bandit).")
    parser.add_argument('--means', nargs='+', type=float, help="True means of each arm (for Gaussian bandit).")
    parser.add_argument('--vars', nargs='+', type=float, help="True variances of each arm (for Gaussian bandit).")
    parser.add_argument('--theta', nargs='+', type=float, help="True parameter vector for the Linear bandit.")
    parser.add_argument('--noise-level', type=float, default=0.1, help="Standard deviation of the Gaussian noise (for Linear bandit).")
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
