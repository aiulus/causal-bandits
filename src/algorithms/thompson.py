import numpy as np
from scipy.stats import beta, invgamma, norm
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

def thompson_sampling_gaussian(n_arms, n_rounds, bandit):
    """
    Runs Thompson Sampling for Gaussian rewards on a given bandit instance with Gaussian rewards.

    Algorithm: 1- Use a Normal-Gamma prior for the mean and precision/variance of the normal distribution
    2- Sample the precision from a Gamma distribution, the mean from a normal distribution 3- Choose the arm
    with the highest sampled mean 4- Update the posterior parameters based on the observed rewards.

    :param n_arms: Number of arms.
    :param n_rounds: Number of rounds to run the simulation.
    :param bandit: Instance of Gaussian_MAB
    :return:
    """
    # Initialize the parameters for Normal-Gamma distribution
    alpha = np.ones(n_arms)
    beta = np.ones(n_arms)
    mu = np.zeros(n_arms)
    lamb = np.ones(n_arms)

    # Initialize list to keep track of the rewards obtained in each run.
    rewards = np.zeros(n_rounds)

    for round in range(n_rounds):
        sampled_means = np.zeros(n_arms)

        for i in range(n_arms):
            sampled_var = invgamma.rvs(a = alpha[i], scale = beta[i])
            sampled_means[i] = norm.rvs(loc=mu[i], scale=np.sqrt(sampled_var / lamb[i]))

        a_opt = np.argmax(sampled_means)
        reward = bandit.pull_arm(a_opt)

        # Update the parameters of the Normal-Gamma distribution
        lamb[a_opt] += 1
        mu[a_opt] = (mu[a_opt] * (lamb[a_opt] - 1) + reward) / lamb[a_opt]
        alpha[a_opt] += 0.5
        beta[a_opt] += 0.5 * (reward - mu[a_opt])**2 / lamb[a_opt]

        rewards[round] = reward

    print(f"Total reward after {n_rounds} rounds: {rewards.sum()}")
    print(f"Sampled means: {sampled_means}")

    return rewards

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
    parser.add_argument('--bandit_type', type=str, choices=['bernoulli', 'gaussian'], required=True, help="Type of bandit problem, currently supported: Bernoulli, Gaussian")
    parser.add_argument('--means', nargs='+', type=float, help="True means of each arm.")
    parser.add_argument('--vars', nargs='+', type=float, help="True variances of each arm.")

    args = parser.parse_args()

    try:
        if args.bandit_type == 'bernoulli':
            bandit = MAB.Bernoulli_MAB(args.arms, args.p)
        elif args.bandit_type == 'gaussian':
            if not args.means or not args.vars:
                raise ValueError("True means and true variances must be provided for Gaussian bandit.")
            bandit = MAB.Gaussian_MAB(args.n_arms, args.means, args.vars)
        else:
            raise ValueError("Unsupported Bandit Type.")
        thompson_sampling_gaussian(args.arms, args.rounds, bandit)
    except Exception as e:
        print(f"Error: {e}")
        parser.print_help()

if __name__ == "__main__":
    main()
