import json
from environment.environment import StandardEnvironment
from environment.reward_functions import bernoulli_reward, gaussian_reward
from environment.causal_structures import NoCausalStructure, SimpleCausalStructure
from algorithms.ucb import UCBAgent
from algorithms.thompson_sampling import ThompsonSamplingAgent
from algorithms.causal_bandit import CausalBanditAgent
from algorithms.pomis_causal_bandit import POMISCausalBanditAgent

def load_config(config_file):
    with open(config_file, 'r') as file:
        config = json.load(file)
    return config

def create_environment(config):
    reward_function = globals()[config["reward_function"]]
    if config["environment"] == "StandardEnvironment":
        return StandardEnvironment(config["n_arms"], reward_function)
    # Add more environment types as needed
    raise ValueError("Unknown environment type")

def create_causal_structure(config):
    if config["causal_structure"] == "None":
        return NoCausalStructure()
    if config["causal_structure"] == "SimpleCausalStructure":
        dependencies = {}  # Define dependencies as needed
        return SimpleCausalStructure(dependencies)
    # Add more causal structure types as needed
    raise ValueError("Unknown causal structure type")

def create_agent(config, n_arms):
    if config["algorithm"] == "UCBAgent":
        return UCBAgent(n_arms)
    if config["algorithm"] == "ThompsonSamplingAgent":
        return ThompsonSamplingAgent(n_arms)
    if config["algorithm"] == "CausalBanditAgent":
        return CausalBanditAgent(n_arms)
    if config["algorithm"] == "POMISCausalBanditAgent":
        return POMISCausalBanditAgent(n_arms)
    raise ValueError("Unknown algorithm type")

def run_experiment(config_file):
    config = load_config(config_file)
    environment = create_environment(config)
    causal_structure = create_causal_structure(config)
    agent = create_agent(config, config["n_arms"])
    environment.probabilities = causal_structure.modify_probabilities(environment.probabilities)
    experiment = Experiment(environment, agent, config["n_rounds"])
    regret = experiment.run()
    return regret

if __name__ == "__main__":
    config_file = "config/config_example.json"
    regret = run_experiment(config_file)
    print(regret)
