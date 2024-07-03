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

def main():
    parser = argparse.ArgumentParser(description="Run a multi-armed bandit problem with UCB.")
    parser.add_argument('--arms', type=int, required=True, help="Number of arms in the bandit problem.")
    parser.add_argument('--rounds', type=int, default=1000, help="Number of times to run the simulation.")
    parser.add_argument('--bandit-type', type=str, choices=['bernoulli', 'gaussian'], required="True",
                        help="Type of bandit problem - currently supported: Bernoulli, Gaussian")
    # For Bernoulli_MAB
    parser.add_argument('--p', nargs='+', type=float, help="Success probabilities for each arm (for Bernoulli bandit).")
    # For Gaussian_MAB
    parser.add_argument('--means', nargs='+', type=float, help="True means of each arm (for Gaussian bandit).")
    parser.add_argument('--vars', nargs='+', type=float, help="True variances of each arm (for Gaussian bandit).")

    args = parser.parse_args()

    try:
        if args.bandit_type == 'bernoulli':
            if not args.p:
                raise ValueError("Probabilities of success must be provided for Bernoulli bandit.")
            bandit = MAB.Bernoulli_MAB(args.arms, args.p)
        elif args.bandit_type == 'gaussian':
            if not args.means or not args.vars:
                raise ValueError("True means and true variances must be provided for Gaussian bandit.")
            bandit = MAB.Gaussian_MAB(args.arms, args.means, args.vars)
        else:
            raise ValueError("Unsupported Bandit Type. Currently supported: ['bernoulli', 'gaussian']")
        UCB1(bandit, args.rounds)
    except Exception as e:
        print(f"Error: {e}")
        parser.print_help()

if __name__ == "__main__":
    main()