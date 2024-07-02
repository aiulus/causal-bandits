import numpy as np
import sys

sys.path.insert(0, 'C:/Users/aybuk/Git/causal-bandits/src/utils')

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

class Linear_MAB:
    def __init__(self, n_arms, context_dim, theta, epsilon=0.1):
        """
        Defines a Linear Bandit instance where each arm is associated with a d-dimensional real-valued
        feature vector. The reward r_A for pulling arm A is given by: r_A = x_A^T.θ + Ɛ,
        where θ theta is a d-dimensional parameter vector unknown to the agent and Ɛ is noise.

        :param n_arms: Number of arms
        :param context_dim: Dimensionality of the context vectors
        :param theta: True parameter vector for the linear model
        :param epsilon: Standard deviation of the Gaussian noise added to the rewards
        """
        self.n_arms = n_arms
        self.context_dim = context_dim
        self.theta = theta
        self.epsilon = epsilon
        # Initialize context/feature vectors with random values drawn from a standard normal distribution
        self.contexts = np.random.randn(n_arms, context_dim)

    def pull_arm(self, arm_index):
        """
        Simulates pulling an arm and returns a reward.

        :param arm_index: Index of the arm to pull
        :return: Reward (dot-product of context and true theta + noise)
        """
        if arm_index < 0 or arm_index >= self.n_arms:
            raise IndexError("Arm index out of bounds.")
        # Adding some Gaussian Noise
        noise = np.random.normal(0, self.epsilon)

        # Return r_A = x_A^T.θ + Ɛ
        return np.dot(self.contexts[arm_index], self.theta) + noise

    def get_arms(self):
        return self.n_arms
    def get_context(self, arm_index):
        if arm_index < 0 or arm_index >= self.n_arms:
            raise IndexError("Arm index out of bounds.")
        return self.contexts[arm_index]

class CausalBandit:
    def __init__(self, scm, reward_variable):
        """

        :param scm: Structural Causal Model
        :param reward_variable: The variable in graph G that represents the reward (Y)
        """
        self.scm = scm
        self.reward_variable = reward_variable

    def intervene(self, interventions):
        """
        Perform an atomic intervention by setting the specified variables to the given values.
        :param interventions: Dictionary with {variable_index : value_to_set_variable}
        """
        self.scm.intervene(interventions)
        self.scm.sample(1) # Updata the SCM with the intervention

    def get_reward(self):
        """
        Get the reward value based on the current observe values.
        :return: Reward value
        """
        observed_values = self.scm.sample(1)
        return observed_values[self.reward_variable][0]

    def get_observed_values(self):
        """
        Get the current observed values of all variables
        :return: Dictionary containing current observed values of all variables
        """
        return self.scm.sample(1)

    def expected_reward(self, interventions, sample_size=1000):
        """
        Calculate the expected reward for a given set of interventions
        :param interventions: Dictionary with keys as variables and values as the values to set
        :return: Expected reward value
        """
        self.intervene(interventions)
        rewards = [self.get_reward() for _ in range(sample_size)]
        return np.mean(rewards)