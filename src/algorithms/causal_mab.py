import abc
from abc import abstractmethod
import numpy as np
from scipy.stats import beta, dirichlet
import sys
from pathlib import Path
import json
import argparse

path_root = Path(__file__).parents[2]
sys.path.append(str(path_root))
from src.utils import MAB, SCM


class CausalBanditAlgorithm:
    """
    Initializes the algorithm with the relevant properties of the CausalBandit (MAB.py) object that's passed
    to __init__().
    """

    def __init__(self, bandit, n_iterations, budget, costs_per_arm, interv_values):
        if interv_values is None:
            interv_values = [0]  # Currently not in use
        self.bandit = bandit
        print(f"Bandit object assigned. Properties: n_arms={self.bandit.n_arms}, n_iterations={n_iterations}")
        print(f"BUDGET: {self.bandit.budget}")
        self.scm = self.bandit.scm
        print(f"SCM object created. Nodes: {self.scm.G.nodes}\n Edges: {self.scm.G.edges}\n ")  # Debug statement
        self.G = self.scm.G
        self.n_arms = self.bandit.n_arms
        print(f"Setting n_arms={self.n_arms}")  # Debug statement
        self.budget = budget if budget else 0.01
        print(f"Setting budget={self.budget}")  # Debug statement
        self.remaining_budget = budget
        self.costs_per_arm = costs_per_arm
        print(f"Setting costs per arm as {self.costs_per_arm}")  # Debug statement
        self.n_iterations = n_iterations
        print(f"Setting T={self.n_iterations}")  # Debug statement
        self.intervention_values = interv_values * self.n_arms if len(interv_values) == 1 else interv_values

    def actions_left(self):
        # If the budget option is not set, it defaults to 0.01
        return self.budget == 0.01 or self.remaining_budget >= min(self.costs_per_arm)

    # TODO: Currenty strategy for modeling pulling an arm is an atomic intervention with value 0, do(X_i = 0),
    #  regardless of the arm's distribution
    def pull_arm(self, arm_index, value=None):
        if value is not None:
            interventions = {arm_index: value}
        else:
            interventions = None

        self.bandit.pull_arm(interventions)
        reward = self.bandit.get_reward()

        if self.budget is not None:
            self.remaining_budget -= self.costs_per_arm[arm_index]

        return reward

    @abstractmethod
    def select_arm(self):
        """
        Corresponds to the specific strategy - must be implemented by the subclass.
        """
        pass

    def run(self):
        print(f"Iterations: {self.n_iterations}\n Object type: {type(self.n_iterations)}")
        cumulative_rewards = np.zeros(self.n_iterations)
        rewards = np.zeros(self.n_iterations)

        for t in range(self.n_iterations):
            if not self.actions_left():
                break

            # TODO: Use self.intervention_values to randomly pick the value to set the arm to from a list of predefined values
            # Select an arm (and value) to pull
            a_t = self.select_arm()
            r_t = self.pull_arm(a_t, self.intervention_values[a_t])

            rewards[t] = r_t
            cumulative_rewards[t] = rewards[:t + 1].sum()

        return rewards, cumulative_rewards


class RandomCausalBandit(CausalBanditAlgorithm):
    def __init__(self, bandit, n_iterations, budget, costs_per_arm, interv_values):
        super().__init__(bandit, n_iterations, budget, costs_per_arm, interv_values)

    def select_arm(self):
        arm_index = np.random.choice(self.n_arms)
        value = np.random.choice(self.intervention_values[arm_index] + [None])
        return arm_index, value


class EpsilonGreedyCausalMAB(CausalBanditAlgorithm):
    def __init__(self, bandit, n_iterations, budget, costs_per_arm, interv_values, epsilon=0.1):
        super().__init__(bandit, costs_per_arm, n_iterations, budget, interv_values)
        self.epsilon = epsilon
        self.arm_values = np.zeros(self.n_arms)
        self.arm_counts = np.zeros(self.n_arms)

    def select_arm(self):
        if np.random.rand() < self.epsilon:
            arm_index = np.random.choice(self.n_arms)  # Exploration
        else:
            arm_index = np.argmax(self.arm_values)  # Exploitation

        value = np.random.choice(self.intervention_values[arm_index] + [None])
        return arm_index, value

    def pull_arm(self, arm_index, value=None):
        reward = super().pull_arm(arm_index, value)
        self.arm_counts[arm_index] += 1
        n = self.arm_counts[arm_index]
        value_estimate = self.arm_values[arm_index]
        self.arm_values[arm_index] = ((n - 1) / n) * value_estimate + (1 / n) * reward
        return reward


class CausalThompsonSampling(CausalBanditAlgorithm):
    def __init__(self, bandit, n_iterations, budget, costs_per_arm, interv_values):
        super().__init__(bandit, costs_per_arm, n_iterations, budget, interv_values)
        self.successes = np.zeros(self.n_arms)
        self.failures = np.zeros(self.n_arms)

    def select_arm(self):
        sampled_values = [beta.rvs(a=1 + self.successes[i], b=1 + self.failures[i]) for i in range(self.n_arms)]
        arm_index = np.argmax(sampled_values)
        value = np.random.choice(self.intervention_values[arm_index] + [None])
        return arm_index, value

    def pull_arm(self, arm_index, value=None):
        reward = super().pull_arm(arm_index, value)
        if reward == 1:
            self.successes[arm_index] += 1
        else:
            self.failures[arm_index] += 1
        return reward


class CausalUCB(CausalBanditAlgorithm):
    def __init__(self, bandit, n_iterations, budget, costs_per_arm, interv_values):
        super().__init__(bandit, costs_per_arm, n_iterations, budget, interv_values)
        self.counts = np.zeros(self.n_arms)
        self.values = np.zeros(self.n_arms)

    def select_arm(self):
        total_counts = np.sum(self.counts)
        ucb_values = self.values + np.sqrt((2 * np.log(total_counts + 1)) / (self.counts + 1))
        arm_index = np.argmax(ucb_values)
        value = np.random.choice(self.intervention_values[arm_index] + [None])
        return arm_index, value

    def pull_arm(self, arm_index, value=None):
        reward = super().pull_arm(arm_index, value)
        self.counts[arm_index] += 1
        n = self.counts[arm_index]
        value_estimate = self.values[arm_index]
        self.values[arm_index] = (n - 1 / n) * value_estimate + (1 / n) * reward
        return reward


class CausalRejectionSampling(CausalBanditAlgorithm):
    def __init__(self, bandit, n_iterations, budget, costs_per_arm, interv_values):
        super().__init__(bandit, costs_per_arm, n_iterations, budget, interv_values)
        self.rewards = np.zeros((self.n_arms, n_iterations // self.n_arms))
        self.arm_indices = np.arange(self.n_arms)

    def select_arm(self):
        if len(self.arm_indices) == 1:
            arm_index = self.arm_indices[0]
        else:
            round_rewards = self.rewards[:, :self.n_iterations // self.n_arms]
            mean_rewards = np.mean(round_rewards, axis=1)
            arm_index = np.argmax(mean_rewards)
        value = np.random.choice(self.intervention_values[arm_index] + [None])
        return arm_index, value

    def pull_arm(self, arm_index, value=None):
        reward = super().pull_arm(arm_index, value)
        self.rewards[arm_index, len(self.rewards[arm_index][self.rewards[arm_index > 0]])] = reward
        if len(self.rewards[arm_index][self.rewards[arm_index] > 0]) == (self.n_iterations // self.n_arms):
            self.arm_indices = np.delete(self.arm_indices, np.where(self.arm_indices == arm_index))
        return reward


class OnlineCausalThompsonSampling(CausalBanditAlgorithm):
    def __init__(self, bandit, n_iterations, budget, costs_per_arm, interv_values):
        super().__init__(bandit, costs_per_arm, n_iterations, budget, interv_values)
        self.successes = np.zeros(self.n_arms)
        self.failures = np.zeros(self.n_arms)

    # TODO: Currently same as 'CausalThompsonSampling'
    def select_arm(self):
        sampled_values = [beta.rvs(a=1 + self.successes[i], b=1 + self.failures[i]) for i in range(self.n_arms)]
        arm_index = np.argmax(sampled_values)
        value = np.random.choice(self.intervention_values[arm_index] + [None])
        return arm_index, value


class BudgetedCumulativeRegret:
    def __init__(self, bandit, budget, costs_per_arm):
        self.G = bandit.scm.G
        self.n_arms = bandit.n_arms
        self.budget = budget
        self.costs_per_arm = costs_per_arm

    def _initial_procedure(self):
        # TODO
        return None


def main():
    parser = argparse.ArgumentParser(description="Run a causal bandit problem with Thompson Sampling.")
    parser.add_argument('--reward', type=str, required=True, help="Reward variable in SCM.")
    parser.add_argument('--rounds', type=int, default=1000, help="Time horizon.")
    parser.add_argument('--algorithm', type=str, choices=['C-TS', 'OC-TS'])
    parser.add_argument('--obs', nargs='+', type=float, required=True,
                        help='Observed probability distribution for the arms.')
    parser.add_argument('--pa_Y', nargs='+', help="Set containing the parents nodes of the reward variable.")
    parser.add_argument('--json', type=str, help="Path to the JSON file describing the SCM.")

    args = parser.parse_args()

    if args.json:
        try:
            with open(args.json, 'r') as file:
                scm_json = json.load(file)
        except FileNotFoundError:
            print(f"Error: This file {args.json} was not found.")
            return
        except json.JSONDecodeError:
            print("Error: Failed to decode JSON from the provided file.")

    scm = SCM(scm_json)
    Y = args.reward
    T = args.rounds
    P_X = np.array(args.obs).reshape(2, 2)

    bandit = MAB.CausalBandit(scm, Y)

    if args.algorithm == 'C-TS':
        cts = CausalThompsonSampling(bandit, T, P_X)
        rewards, cumulative_rewards = cts.run()
    elif args.algorithm == 'OC-TS':
        if not args.pa_Y:
            raise ValueError(
                "Parent nodes of the reward variable must be provided for Online Causal Thompson Sampling.")
        octs = OnlineCausalThompsonSampling(bandit, T)
        rewards, cumulative_rewards = octs.run()
    else:
        raise ValueError("Unsupported algorithm type.")

    print("Rewards obtained: ", rewards)
    print("Cumulative rewards: ", cumulative_rewards)


if __name__ == "__main__":
    main()
