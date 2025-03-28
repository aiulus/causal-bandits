from pathlib import Path
import sys
import argparse
import numpy as np

import SCM, MAB, plots, io_mgmt
# import src.algorithms.thompson as thompson
# import src.utils.thompson_draft as thompson
# import src.algorithms.ucb as UCB
import src.algorithms.causal_mab as causalMAB
import src.algorithms.classic_mab as classicMAB

config = io_mgmt.configuration_loader()
PATH_REWARDS = config['PATH_REWARDS']
PATH_REGRET = config['PATH_REGRET']

path_root = Path(__file__).parents[2]
sys.path.append(str(path_root))


class BanditConfig:
    def __init__(self, args):
        self.scm_path = args.scm_path
        self.bandit_type = args.bandit_type
        self.n_arms = args.n_arms
        self.T = args.T
        self.p = args.p
        self.mu = args.mu
        self.sigma_sqr = args.sigma_sqr
        self.context_dim = args.context_dim
        self.theta = args.theta
        self.epsilon = args.epsilon
        self.algorithm = args.algorithm
        self.eval_mode = args.eval_mode
        self.obs = args.obs
        self.save = args.save
        self.budget = args.budget
        self.costs = args.costs
        self.do_values = args.do_values

    def to_dict(self):
        return {
            'scm_path': self.scm_path,
            'bandit_type': self.bandit_type,
            'n_arms': self.n_arms,
            'T': self.T,
            'p': self.p,
            'mu': self.mu,
            'sigma_sqr': self.sigma_sqr,
            'context_dim': self.context_dim,
            'theta': self.theta,
            'epsilon': self.epsilon,
            'algorithm': self.algorithm,
            'eval_mode': self.eval_mode,
            'obs': self.obs,
            'save': self.save,
            'budget': self.budget,
            'costs': self.costs,
            'do_values': self.do_values
        }


def run_algorithm(bandit, config):
    algorithms = {
        'thompson': {
            'bernoulli': classicMAB.ThompsonSamplingBernoulli,
            'gaussian': classicMAB.ThompsonSamplingGaussian,
            'linear': classicMAB.ThompsonSamplingLinear
        },
        'ucb': classicMAB.UCB1,
        'causal_ts': causalMAB.CausalThompsonSampling,
        'online_causal_ts': causalMAB.OnlineCausalThompsonSampling,
        'random_causal': causalMAB.RandomCausalBandit,
        'epsilon_greedy': causalMAB.EpsilonGreedyCausalMAB,
        'causal_ucb': causalMAB.CausalUCB,
        'causal_rs': causalMAB.CausalRejectionSampling
    }
    algo = config.algorithm
    if algo == 'thompson':
        bandit_type = config.bandit_type
        if bandit_type not in algorithms['thompson']:
            raise ValueError(f"Unsupported bandit type for Thompson Sampling : {bandit_type}.\n"
                             f"Currently supported:['linear', 'gaussian', 'bernoulli']")
        algo_class = algorithms['thompson'][bandit_type]
        algo_instance = algo_class(config.n_arms, config.T, bandit, config.budget)
    elif algo == 'ucb':
        algo_class = algorithms['ucb']
        algo_instance = algo_class(config.n_arms, config.T, bandit, config.budget)
    else:
        algo_class = algorithms[algo]
        # algo_instance = algo_class(bandit, args.T, args.budget, args.costs, args.do_values)
        algo_instance = algo_class(bandit, config.T, config.budget, config.costs, config.do_values)
    return algo_instance.run()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scm_path', type=str, help="Specify the path for the underlying SCM via --scm_path. "
                                                     "Required for Causal Multi-Armed Bandit.")
    parser.add_argument('--bandit_type', type=str, required=True, choices=['causal', 'linear', 'gaussian', 'bernoulli'],
                        help="Please specify the bandit type. Currenty supported: ['causal', 'linear', 'gaussian', 'bernoulli']")
    parser.add_argument('--n_arms', type=int, required=True,
                        help="Please specify the number of arms for the bandit with '--n_arms'.")
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
    parser.add_argument('--algorithm', type=str, required=True,
                        choices=['thompson', 'ucb', 'causal_ts', 'online_causal_ts', 'random_causal', 'epsilon_greedy',
                                 'causal_ucb', 'causal_rs'],
                        help="Please specify the type of algorithm to use. Currently supported: "
                             "['thompson', 'ucb', 'causal_ts', 'online_causal_ts', 'random_causal', 'epsilon_greedy', "
                             " 'causal_ucb']")
    parser.add_argument('--eval_mode', type=str, required=True, choices=['cumulative_reward', 'simple_reward', 'cumulative_regret', 'simple_regret'],
                        help="Please specify the reward type. Currently supported: ['cumulative', 'simple']")
    # TODO: Need to parse, e.g., N(0,1) to a string representation of the corresponding lambda function (pdf)
    #  before passing it to the algorithm
    parser.add_argument('--obs', nargs='+', type=str,
                        help="Observed probability distributions of the arms. Required for Causal Thompson Sampling. ('C-TS')")
    # TODO: Add the functionality to change the reward variable's distribution
    parser.add_argument('--save', action='store_true')
    parser.add_argument('--budget', type=float, default=0.01, help="Specify the budget for the bandit.")
    parser.add_argument('--costs', type=float, nargs='+', default=[0.0],
                        help="Specify the costs of pulling each arm.")
    parser.add_argument('--do_values', type=float, nargs='+', default=[0],
                        help="Specify default values for interventions (required for Causal Bandits).")


    args = parser.parse_args()

    MAB.validate_bandit_args(args)

    # print(f"Args: {args}")  # Debug statement
    config = BanditConfig(args)

    bandit_kwargs = {
        'n_arms': config.n_arms,
        'p_true': config.p,
        'means': config.mu if config.mu else None,
        'variances': config.sigma_sqr if config.sigma_sqr else None,
        'context_dim': config.context_dim if config.context_dim else None,
        'theta': np.array(config.theta) if config.theta else None,
        'epsilon': config.epsilon if config.epsilon else None,
        'scm': SCM.SCM(config.scm_path) if config.scm_path else None,
        'reward_variable': 'Y',
        'budget': config.budget,
        'costs': config.costs,
        'do_values': config.do_values,
        'T': config.T
    }

    bandit = MAB.create_bandit(args.bandit_type, **{k: v for k, v in bandit_kwargs.items() if v is not None})
    debug_dict = {k: v for k, v in bandit_kwargs.items() if v is not None}
    print(f" Created bandit instance with: {debug_dict}")  # Debug statement

    rewards_simple, rewards_cumulative, regrets_simple, regrets_cumulative = run_algorithm(bandit, config)

    if "reward" in args.eval_mode:
        type2 = "reward"
        if "simple" in args.eval_mode:
            data = rewards_simple
            type1 = "simple"
        elif args.eval_mode.count("cumulative") == 1:
            data = rewards_cumulative
            type1 = "cumulative"
        else:
            raise ValueError(f"Unsupported evaluation mode: {args.eval_mode}")
    if "regret" in args.eval_mode:
        type2 = "regrets"
        if "simple" in args.eval_mode:
            data = regrets_simple
            type1 = "simple"
        elif "cumulative" in args.eval_mode:
            data = regrets_cumulative
            type1 = "cumulative"
        else:
            raise ValueError(f"Unsupported evaluation mode: {args.eval_mode}")

    data_filename = PATH_REWARDS + f"/{args.algorithm}_{args.n_arms}-armed_{args.bandit_type}_bandit_T{args.T}_{type1}-{type2}"
    io_mgmt.save_rewards_to_csv(data, data_filename)
    plots.plot_outputs(data, args.algorithm, args.bandit_type, data_filename, args.save)


if __name__ == '__main__':
    main()
