import numpy as np
from scipy.stats import beta
import argparse
import sys

sys.path.insert(0, 'C:/Users/aybuk/Git/causal-bandits/src/utils')
def thompson_sampling(n_arms, n_rounds, bandit):
    """
    Runs Thompson Sampling on a given Multi-Armed Bandit instance.

    :param n_arms: Number of arms
    :param n_rounds: Number of runs to run the simulation
    :param bandit: Instance of Multi-Armed Bandit
    :return:
    """
    if n_arms != bandit.get_arms():
        raise ValueError("Number of arms does not match with the given MAB instance.")

    # Initialize return values
    successes = np.zeros(n_arms)
    failures = np.zeros(n_arms)
    rewards = np.zeros(n_rounds)

    for round in range(n_rounds):
        """
        Sample from the beta distribution for each arm -
        Initialization Beta(1, 1) chosen as a non-informative prior
        """
        sampled_values = [beta.rvs(a = 1 + successes[i], b = 1 + failures[i]) for i in range(n_arms)]
        # Select the arm with the highest sampled value
        a_opt = np.argmax(sampled_values)
        # Pull the chosen arm and get the reward
        reward = bandit.pull_arm(a_opt)
        # Update track record based on the result of the Bernoulli trial
        if reward == 1:
            successes[a_opt] += 1
        else:
            failures[a_opt] += 1
        rewards[round] = reward

    print(f"Total reward after {n_rounds} rounds: {rewards.sum()}")
    print(f"Successes: {successes}")
    print(f"Failures: {failures}")

    return rewards, successes, failures

from src.utils import MAB
def main():
    """
    Main function to run the Multi-Armed Bandit problem with Thompson Sampling
    :return:
    """

    parser = argparse.ArgumentParser(description="Run a multi-armed bandit problem with Thompson Sampling.")
    parser.add_argument('--arms', type=int, required=True, help="Number of arms in the bandit problem")
    parser.add_argument('--p', nargs='+', type=float, required=True, help="True probabilities of each arm.")
    parser.add_argument('--rounds', type=int, default=1000, help="Number of rounds to run the simulation.")
    parser.add_argument('--bandit_type', type=str, choices=['bernoulli'], required=True, help="Type of bandit problem: {Bernoulli, ...}")

    args = parser.parse_args()

    try:
        if args.bandit_type == 'bernoulli':
            bandit = MAB.Bernoulli_MAB(args.arms, args.p)
        else:
            raise ValueError("Unsupported Bandit Type.")
        thompson_sampling(args.arms, args.rounds, bandit)
    except Exception as e:
        print(f"Error: {e}")
        parser.print_help()

if __name__ == "__main__":
    main()
