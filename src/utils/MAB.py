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
