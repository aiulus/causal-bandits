import numpy as np
import sys, argparse

sys.path.insert(0, 'C:/Users/aybuk/Git/causal-bandits/src/utils')
from src.utils import MAB, SCM


# TODO: REDUNDANT // REMOVE
def UCB1(bandit, n_rounds):
    """
S
    :param bandit:
    :param n_rounds:
    :return: Total reward after all rounds, counts of pulls for each arm
    """
    n_arms = bandit.get_arms()
    counts = np.zeros(n_arms)  # Number of times each arm has been played
    means = np.zeros(n_arms)  # Mean reward for each arm
    rewards = np.zeros(n_rounds)  # Rewards obtained in each round
    cumulative_rewards = np.zeros(n_rounds)

    for t in range(n_rounds):
        if t < n_arms:
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


class UCB1:
    def __init__(self, n_arms, n_rounds, bandit, budget):
        self.n_arms = n_arms
        self.n_rounds = n_rounds
        self.bandit = bandit
        self.budget = budget
        self.remaining_budget = budget
        self.costs = bandit.costs if bandit.costs else [1] * n_arms
        self.rewards = np.zeros(self.n_rounds)
        self.cumulative_rewards = np.zeros(n_rounds)
        self.counts = np.zeros(n_arms)
        self.means = np.zeros(n_arms)
        self.counter = 0

    def actions_left(self):
        return self.budget is None or self.remaining_budget >= min(self.costs)

    def pull_arm(self, arm_index):
        reward = self.bandit.pull_arm(arm_index)
        if reward is not None and self.actions_left():
            self.remaining_budget -= self.costs[arm_index]
        return reward

    def update_counts_and_means(self, arm_index, reward):
        self.counts[arm_index] += 1
        n = self.counts[arm_index]
        mu_t = self.means[arm_index]
        if reward is not None:
            self.means[arm_index] = ((n - 1) / n) * mu_t + (1 / n) * reward

    def run(self):
        if not self.actions_left():
            print(f"Insufficient budget ({self.remaining_budget}) to run the simulation"
                  f" (min. {min(self.costs)} needed.)")
            return self.rewards, self.cumulative_rewards

        for t in range(self.n_rounds):
            self.counter += 1
            if not self.actions_left():
                break
            if t < self.n_arms:
                a_t = t  # Play each arm once if time horizon allows for that
            else:
                ucb_vals = self.means + np.sqrt((2 * np.log(t)) / self.counts)
                a_t = np.argmax(ucb_vals)

            r_t = self.pull_arm(a_t)
            if r_t is not None:
                self.update_counts_and_means(a_t, r_t)
                self.rewards[t] = r_t
                self.cumulative_rewards[t] = self.rewards[:t + 1].sum()

        print(f"Total reward after {self.counter} rounds: {self.rewards.sum()}")
        print(f"Number of times each arm was played: {self.counts}")
        return self.rewards, self.cumulative_rewards
