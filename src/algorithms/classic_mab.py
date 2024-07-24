from abc import ABC, abstractmethod
import numpy as np
from scipy.stats import beta, invgamma, norm
import sys
from pathlib import Path

path_root = Path(__file__).parents[2]
sys.path.append(str(path_root))


class BanditAlgorithm(ABC):
    def __init__(self, n_arms, T, bandit, budget):
        self.n_arms = n_arms
        self.T = T
        self.bandit = bandit
        self.budget = budget
        self.costs = bandit.costs if bandit.costs and bandit.budget else [0] * n_arms
        self.remaining_budget = budget
        self.rewards = np.zeros(self.T)
        self.cumulative_rewards = np.zeros(self.T)
        self.counts = np.zeros(n_arms)

    def actions_left(self):
        return self.budget is None or self.remaining_budget >= min(self.bandit.costs)

    def pull_arm(self, arm_index):
        reward = self.bandit.pull_arm(arm_index)
        if self.budget is not None and self.actions_left():
            self.remaining_budget -= self.costs[arm_index]
        return reward

    @abstractmethod
    def simulate_single_pull(self):
        pass

    def run(self):
        if not self.actions_left():
            print("Insufficient budget to run the simulation.")
            return self.rewards, self.cumulative_rewards

        for t in range(self.T):
            if not self.actions_left():
                break
            reward, _ = self.simulate_single_pull()
            self.rewards[t] = reward
            self.cumulative_rewards[t] = self.rewards[:t + 1].sum()

        print(f"Total reward after (<=) {self.T} rounds: {self.rewards.sum()}")
        print(f"Number of times each arm was played: {self.counts}")
        return self.rewards, self.cumulative_rewards


class ThompsonSamplingBernoulli(BanditAlgorithm):
    def __init__(self, n_arms, T, bandit, budget):
        super().__init__(n_arms, T, bandit, budget)
        self.successes = np.zeros(self.n_arms)
        self.failures = np.zeros(self.n_arms)

    def simulate_single_pull(self):
        sampled_values = [beta.rvs(a=1 + self.successes[i], b=1 + self.failures[i]) for i in range(self.n_arms)]
        a_opt = np.argmax(sampled_values)
        reward = self.bandit.pull_arm(a_opt)
        self.counts[a_opt] += 1

        if reward == 1:
            self.successes[a_opt] += 1
        else:
            self.failures[a_opt] += 1

        return reward, a_opt


class ThompsonSamplingGaussian(BanditAlgorithm):
    def __init__(self, n_arms, T, bandit, budget):
        super().__init__(n_arms, T, bandit, budget)
        self.alpha = np.ones(self.n_arms)
        self.beta = np.ones(self.n_arms)
        self.mu = np.zeros(self.n_arms)
        self.lamb = np.ones(self.n_arms)

    def simulate_single_pull(self):
        sampled_means = np.zeros(self.n_arms)

        for i in range(self.n_arms):
            sampled_var = invgamma.rvs(a=self.alpha[i], scale=self.beta[i])
            sampled_means[i] = norm.rvs(loc=self.mu[i], scale=np.sqrt(sampled_var / self.lamb[i]))

        a_opt = np.argmax(sampled_means)
        print(f"ARM PLAYED: {a_opt}")  # Debug statement
        reward = self.bandit.pull_arm(a_opt)
        print(f"REWARD: {reward}")  # Debug statement

        if reward is not None:
            self.lamb[a_opt] += 1
            self.mu[a_opt] = (self.mu[a_opt] * (self.lamb[a_opt] - 1) + reward) / self.lamb[a_opt]
            self.alpha[a_opt] += 0.5
            self.beta[a_opt] += 0.5 * (reward - self.mu[a_opt]) ** 2 / self.lamb[a_opt]

        return reward, a_opt


class ThompsonSamplingLinear(BanditAlgorithm, ABC):
    def __init__(self, n_arms, T, bandit, budget=None):
        super().__init__(n_arms, T, bandit, budget)

    def run(self):
        context_dim = self.bandit.context_dim
        alpha = 1.0
        beta = 1.0
        A = np.eye(context_dim) / alpha
        b = np.zeros(context_dim)

        rewards = np.zeros(self.T)
        cumulative_rewards = np.zeros(self.T)

        for round in range(self.T):
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

        return rewards, cumulative_rewards


class UCB1(BanditAlgorithm):
    def __init__(self, n_arms, T, bandit, budget):
        super().__init__(n_arms, T, bandit, budget)
        self.means = np.zeros(n_arms)
        self.counter = 0

    def update_counts_and_means(self, arm_index, reward):
        self.counts[arm_index] += 1
        n = self.counts[arm_index]
        mu_t = self.means[arm_index]
        if reward is not None:
            self.means[arm_index] = ((n - 1) / n) * mu_t + (1 / n) * reward

    # TODO
    def simulate_single_pull(self):
        t = int(np.sum(self.counts) + 1)  # Current round
        if t < self.n_arms:
            a_t = t - 1  # Play each arm once if time horizon allows for that
        else:
            ucb_vals = self.means + np.sqrt((2 * np.log(t)) / self.counts)
            a_t = np.argmax(ucb_vals)

        r_t = self.pull_arm(a_t)
        if r_t is not None:
            self.update_counts_and_means(a_t, r_t)

        return r_t, a_t
