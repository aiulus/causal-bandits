import numpy as np
import sys, argparse
sys.path.insert(0, 'C:/Users/aybuk/Git/causal-bandits/src/utils')
from src.utils import MAB, SCM

def UCB1(bandit, rounds):
    """

    :param bandit:
    :param rounds:
    :return: Total reward after all rounds, counts of pulls for each arm
    """
    n_arms = bandit.get_arms()
    counts = np.zeros(n_arms) # Number of times each arm has been played
    means = np.zeros(n_arms) # Mean reward for each arm
    rewards = np.zeros(rounds) # Rewards obtained in each round

    for t in range(rounds):
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

    print(f"Total reward after {rounds} rounds: {rewards.sum()}")
    print(f"Number of times each arm was played: {counts}")

    return rewards, counts