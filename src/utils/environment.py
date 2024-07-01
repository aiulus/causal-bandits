import numpy as np

class StandardEnvironment:
    def __init__(self, n_arms, reward_function):
        self.n_arms = n_arms
        self.probabilities = np.random.rand(n_arms)
        self.reward_function = reward_function

    def get_reward(self, arm):
        return self.reward_function(arm, self.probabilities)
def bernoulli_reward(arm, probabilities):
    return 1 if np.random.rand() < probabilities[arm] else 0

def gaussian_reward(arm, means, std_devs):
    return np.random.normal(means[arm], std_devs[arm])

class NoCausalStructure:
    def modify_probabilities(self, probabilities):
        return probabilities

class SimpleCausalStructure:
    def __init__(self, dependencies):
        self.dependencies = dependencies

    def modify_probabilities(self, probabilities):
        # Modify the probabilities based on the causal structure
        modified_probabilities = probabilities.copy()
        for arm, dep in self.dependencies.items():
            modified_probabilities[arm] *= probabilities[dep]
        return modified_probabilities

