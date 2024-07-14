from pathlib import Path
import sys
import argparse
import numpy as np

import SCM, MAB, plots, io_mgmt
import src.algorithms.thompson as thompson
import src.algorithms.ucb as UCB
import src.algorithms.causal_thompson as causal_thompson

config = io_mgmt.configuration_loader()
PATH_REWARDS = config['PATH_REWARDS']

path_root = Path(__file__).parents[2]
sys.path.append(str(path_root))


# print(sys.path)

# bandit = MAB.Bernoulli_MAB(10, 0.5)
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
    parser.add_argument('--reward_type', type=str, required=True, choices=['cumulative', 'simple'],
                        help="Please specify the reward type. Currently supported: ['cumulative', 'simple']")
    parser.add_argument('--save', action='store_true')

    args = parser.parse_args()

    MAB.validate_bandit_args(args)

    bandit_kwargs = {
        'n_arms': args.n_arms,
        'p_true': args.p,
        'means': args.mu if args.mu else None,
        'variances': args.sigma_sqr if args.sigma_sqr else None,
        'context_dim': args.context_dim if args.context_dim else None,
        'theta': np.array(args.theta) if args.theta else None,
        'epsilon': args.epsilon if args.epsilon else None,
        'scm': SCM.load_scm(args.scm_path) if args.scm_path else None,
        'reward_variable': 'reward_variable'  # Replace with actual reward variable name if necessary
    }
    bandit = MAB.create_bandit(args.bandit_type, **{k: v for k, v in bandit_kwargs.items() if v is not None})

    if args.algorithm == 'thompson':
        rewards_simple, rewards_cumulative = run_thompson(bandit, args.bandit_type, args.T)
    elif args.algorithm == 'ucb':
        rewards_simple, rewards_cumulative = UCB.UCB1(bandit, args.T)

    rewards_filename = PATH_REWARDS + f"{args.algorithm}_{args.n_arms}-armed_{args.bandit_type}_bandit_T{args.T}_{args.reward_type}-rewards"
    if args.reward_type == 'simple':
        rewards = rewards_simple
    elif args.reward_type == 'cumulative':
        rewards = rewards_cumulative
    io_mgmt.save_rewards_to_csv(rewards, rewards_filename)
    plots.plot_rewards(rewards, args.algorithm, args.bandit_type, args.save)


if __name__ == '__main__':
    main()
