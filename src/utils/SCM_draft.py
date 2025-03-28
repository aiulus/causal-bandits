import networkx as nx
import json
import matplotlib.pyplot as plt
import sys
import os
from pathlib import Path

from src.utils import io_mgmt

path_root = Path(__file__).parents[2]
sys.path.append(str(path_root))

config = io_mgmt.configuration_loader()
PATH_GRAPHS = config['PATH_GRAPHS']
PATH_SCM = config['PATH_SCMs']
PATH_PLOTS = config['PATH_PLOTS']
MAX_DEGREE = config['MAX_POLYNOMIAL_DEGREE']
COEFFS = config['COEFFICIENTS']
DISTS = config['DISTS']


def parse_scm(input_data):
    if isinstance(input_data, str):
        with open(input_data, 'r') as file:
            data = json.load(file)
    elif isinstance(input_data, dict):
        data = input_data
    else:
        raise ValueError("Input must be a JSON string or a dictionary.")

    nodes = data['nodes']
    functions = data['functions']
    noise = data['noise']

    G = nx.DiGraph()
    G.add_nodes_from(data['nodes'])
    G.add_edges_from(data['edges'])

    return nodes, G, functions, noise


class BaseSCM:
    def __init__(self, input_data):
        self.nodes, self.G, self.F, self.N = parse_scm(input_data)
        self.interventions = []
        self.state_history = []

    def save_state(self):
        state = {
            'F': self.F.copy(),
            'N': self.N.copy(),
            'interventions': self.interventions.copy()
        }
        self.state_history.append(state)

    def restore_state(self):
        if self.state_history:
            state = self.state_history.pop()
            self.F = state['F']
            self.N = state['N']
            self.interventions = state['interventions']

    def intervene(self, interventions):
        self.save_state()
        for variable, func in interventions.items():
            lambda_string = f"lambda _: {func}"
            self.interventions[variable] = func
            self.F[variable] = lambda_string

    def sample(self, n_samples, mode='observational', interventions=None):
        raise NotImplementedError("Sampling method must be implemented by the SCM subclass.")

    def visualize(self):
        if nx.is_planar(self.G):
            pos = nx.planar_layout(self.G)
        else:
            pos = nx.bfs_layout(self.G)
        nx.draw(self.G, pos, with_labels=True, node_size=1000, node_color='lightblue', font_size=10, font_weight='bold')
        plt.show()

    def save_to_json(self, filename):
        os.makedirs(PATH_SCM, exist_ok=True)
        scm_data = {
            "nodes": [node for node in self.nodes],
            "edges": [edge for edge in self.G.edges],
            "functions": {k: str(v) for k, v in self.F.items()},
            # TODO: Save only the type(s) of distribution for the noises
            "noise": self.N
        }
        file_path = os.path.join(PATH_SCM, filename)
        with open(file_path, 'w') as f:
            json.dump(scm_data, f, indent=2)
        print(f"SCM saved to {file_path}")

    @staticmethod
    def func_to_str(func):
        return func

    @staticmethod
    def str_to_func(func_str):
        return eval(func_str)

