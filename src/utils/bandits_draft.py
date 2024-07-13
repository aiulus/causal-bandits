from pathlib import Path
import sys
import argparse
import numpy as np

import SCM, MAB, plots, io_mgmt
import src.algorithms.thompson as thompson
import src.algorithms.ucb as UCB
import src.algorithms.causal_thompson as causal_thompson

path_root = Path(__file__).parents[2]
sys.path.append(str(path_root))
# print(sys.path)

#bandit = MAB.Bernoulli_MAB(10, 0.5)
# bandit.n_arms


def run_thompson(bandit, bandit_type, n_iterations):
    n_arms = bandit.get_arms()

    thompson_methods = {
        'bernoulli': thompson.thompson_sampling,
        'gaussian': thompson.thompson_sampling_gaussian,
        'linear': thompson.thompson_sampling_linear
    }

    if bandit_type not in thompson_methods:
        raise ValueError(f"Unsupported bandit type for Thompson Sampling : {bandit_type}.\n"
                         f"Currently supported:['linear', 'gaussian', 'bernoulli']")

    return thompson_methods[bandit_type](n_arms, n_iterations, bandit)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scm_path', type=str, help="Specify the path for the underlying SCM via --scm_path. "
                                                     "Required for Causal Multi-Armed Bandit.")
    parser.add_argument('--bandit_type', type=str, required=True, choices=['causal', 'linear', 'gaussian', 'bernoulli'],
                        help="Please specify the bandit type. Currenty supported: ['causal', 'linear', 'gaussian', 'bernoulli']")
    parser.add_argument('--n_arms', type=int, help="Please specify the number of arms for the bandit with '--n_arms'.")
    parser.add_argument('--T', type=int, required=True, help="Time horizon / #iterations for the bandit algorithm.")
    parser.add_argument('--p', type=float, nargs='+',
                        help="Probability vector for Bernoulli bandit, must match the number of arms.")
    parser.add_argument('--mu', type=float, nargs='+',
                        help="Means for Gaussian bandit, must match the number of arms.")
    parser.add_argument('--sigma_sqr', type=float, nargs='+',
                        help="Variances for Gaussian bandit, must match the number of arms.")
    parser.add_argument('--context_dim', type=int, help="Context dimension for Linear bandit.")
    parser.add_argument('--theta', type=float, nargs='+', help="Theta vector for Linear bandit.")
    parser.add_argument('--epsilon', type=float, default=0.1,
                        help="Epsilon (noise standard deviation) for Linear bandit.")
    parser.add_argument('--algorithm', type=str, choices=['thompson', 'ucb'],
                        help="Please specify the type of algorithm to use. Currently supported: ['thompson', 'ucb']")
    parser.add_argument('--save', action='store_true')

    args = parser.parse_args()

    MAB.validate_bandit_args(args)

    bandit_kwargs = {
        'n_arms': args.n_arms,
        'p_true': args.p,
        'means': args.mu,
        'variances': args.sigma_sqr,
        'context_dim': args.context_dim,
        'theta': np.array(args.theta),
        'epsilon': args.epsilon,
        'scm': SCM.load_scm(args.scm_path) if args.scm_path else None,
        'reward_variable': 'reward_variable'  # Replace with actual reward variable name if necessary
    }

    bandit = MAB.create_bandit(args.bandit_type, **{k: v for k, v in bandit_kwargs.items() if v is not None})

    if args.algorithm == 'thompson':
        rewards = run_thompson(bandit, args.bandit_type, args.T)
    elif args.algorithm == 'ucb':
        rewards = UCB.UCB1(bandit, args.T)

    io_mgmt.save_rewards_to_csv(rewards)
    plots.plot_rewards(rewards, args.algorithm, args.bandit_type, args.save)

if __name__ == '__main__':
    main()