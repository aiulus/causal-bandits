import numpy as np

class Bernoulli_MAB:
    def __init__(self, n_arms, p_true):
        """
        :param n_arms: Number of arms
        :param p_true: Probability of success for each arm
        """
        if n_arms != p_true:
            raise ValueError("Number of arms must match the length of the probability vector.")
        if any(p < 0 or p > 1 for p in p_true):
            raise ValueError("All probabilities must be between 0 and 1.")

        self.n_arms = n_arms
        self.p_true = p_true

    # Simulates pulling an arm
    def pull_arm(self, arm_index):
        """
        :param arm_index: Index of the arm to pull
        :return: Reward (1 if successful, 0 otherwise)
        """
        if arm_index < 0 or arm_index >= self.n_arms:
            raise IndexError("Arm index out of bounds.")
        # Simulates a Bernoulli Trial with a probability of success as specified in p_true during initialization
        return np.random.rand() < self.p_true[arm_index]

    def get_arms(self):
        return self.n_arms
    def get_p(self):
        return self.p_true

class Gaussian_MAB:
    def __init__(self, n_arms, means, variances):
        """
        Defines a Multi-Armed Bandit instance where each arm provides rewards drawn from a normal distribution with
        a true mean and a true variance unknown to the agent.

        :param n_arms: Number of arms in the bandit instance.
        :param means: List of true means of the normal distributions for each arm.
        :param variances: List of true variances of the normal distributions for each arm.
        """
        if n_arms != len(means) or n_arms != len(variances):
            raise ValueError("Number of arms must match the length of the true means and true variances.")
        if any(var < 0 for var in variances):
            raise ValueError("All variances must be non-negative.")

        self.n_arms = n_arms
        self.means = means
        self.variances = variances

    def pull_arm(self, arm_index):
        """
        Simulates pulling an arm and returns the reward.

        :param arm_index: Index of the arm to pull
        :return: Reward drawn from the normal distribution of the specified arm.
        """

        if arm_index < 0 or arm_index >= self.n_arms:
            raise IndexError("Arm index out of bounds.")

        # Simulates pulling an arm
        return np.random.normal(self.means[arm_index], np.sqrt(self.variances[arm_index]))

    def get_arms(self):
        return self.n_arms

