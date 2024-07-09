import argparse
import json
import sys
import os

import networkx as nx
import numpy as np
from scipy.stats import norm, bernoulli, expon
import matplotlib.pyplot as plt
import re

sys.path.insert(0, 'C:/Users/aybuk/Git/causal-bandits/src/utils')
import graph_generator, plots

# Set target destination for .json files containing graph structures
PATH_GRAPHS = "../../outputs/graphs"
PATH_SCM = "../../outputs/SCMs"
PATH_PLOTS = "../../outputs/plots"
MAX_DEGREE = 3  # For polynomial function generation
# Set of coefficients to choose from
PRIMES = [-11, -7, -5, -3, -2, 2, 3, 5, 7, 11]

def parse_scm(input):
    if isinstance(input, str):
        # JSON input
        data = json.loads(input)
    elif isinstance(input, dict):
        # Dictionary input
        data = input
    else:
        raise ValueError("Input must be a JSON string or a dictionary")

    nodes = data['nodes']
    functions = data['functions']
    noise = data['noise']

    G = nx.DiGraph()
    G.add_nodes_from(data['nodes'])
    G.add_edges_from(data['edges'])

    return nodes, G, functions, noise

def parse_noise_string(noise_str):
    dist_type_map = {
        'N': 'gaussian',
        'Exp': 'exponential',
        'Ber': 'bernoulli'
    }

    pattern = r'([A-Za-z]+)\(([^)]+)\)'
    match = re.match(pattern, noise_str)
    if not match:
        raise ValueError(f"Invalid distribution format: {noise_str}")

    noise_type, params = match.groups()
    params = [float(x) for x in params.split(',')]

    if noise_type not in dist_type_map:
        raise ValueError(f"Unsupported distrbution type: {noise_type}")

    return {"type": dist_type_map[noise_type], "params": params}

def parse_noise(noise):
    """
    Parse noise distribution strings into a dictionary format.
    Example: "N(0,1) --> {"type": "gaussian", "params": [0,1]}
    """

    noise_dict = {}

    if isinstance(noise, list):
        for i, noise_str in enumerate(noise):
            noise_dict[i] = parse_noise_string(noise_str)
    else:
        dist_type, params = noise[0].lower(), [float(x) for x in noise[2:].split(',').strip(')').strip(' ')]
        noise_dict[dist_type] = {"type": dist_type, "params": params}
    return noise_dict


def parse_interventions(interventions):
    """
    Parse intervention strings like 'do(X_i=a)' into a dictionary.
    Example: "do(X1=0)" --> {"X1": 0}
    """
    interventions_dict = {}
    for intervention in interventions:
        var, val = intervention.replace('do(', '').replace(')', '').split('=')
        interventions_dict[var.strip()] = float(val.strip())

    return interventions_dict


def generate_distributions(variables, distr_type: str, params=None):
    # TODO: make sure that 'params' is generated as a list of lists [[]]
    # TODO: currently no way of passing a 'params' array in a manageable way
    p_n = {}

    # If the parameters are not specified, use standard values
    if params is None:
        params = {
            'gaussian': {'mu': 0, 'sigma': 1},
            'bernoulli': {'p': 0.5},
            'exp': {'lam': 1.0}
        }

    # TODO: too nested
    for x_i in variables:
        if distr_type == 'gaussian':
            mu, sigma = params['gaussian']['mu'], params['gaussian']['sigma']
            p_n[x_i] = lambda x, mu=mu, sigma=sigma: norm.rvs(mu, sigma, size=x)
        elif distr_type == 'bernoulli':
            p = params['bernoulli']['p']
            p_n[x_i] = lambda x, p=p: bernoulli.rvs(p, size=x)
        elif distr_type == 'exp':
            lam = params['exp']['lam']
            p_n[x_i] = lambda x, lam=lam: expon.rvs(scale=1 / lam, size=x)
        else:
            raise ValueError(f"Unsupported distribution type:{distr_type}")

    return p_n


def generate_linear_function(parents, noise, coeffs):
    """

    :param parents: immediate predecessors (pa(X_i)) of node X_i in graph G
    :param coeffs: coefficient vector of length |pa(X_i)|
    :return: Linear function f_i(pa(X_i), N_i) = f(pa(X_i)) + N_i as a string.
    """
    terms = [f"{coeffs[i]} * {parent}" for i, parent in enumerate(parents)]
    terms.append(noise)
    function = f"lambda {', '.join(parents)}: " + " + ".join(terms)

    return function


def generate_polynomial(parents, noise, coeffs, degrees):
    """
    Generate f_i(pa(X_i), N_i) = polynomial(pa(X_i), a_j, e_j) + N_i
    :param parents: pa(X_i)
    :param coeffs: a_i
    :param degrees: e_i in f_i(pa(X_i), N_i) = N_i + sum(a_j * X_i^{e_j} for j in pa(X_i))
    :return: Linear function f_i(pa(X_i), N_i) = f(pa(X_i)) + N_i as a string.
    """
    terms = [f"{coeffs[i]}*{parent}**{degrees[i]}" for i, parent in enumerate(parents)]
    terms.append(noise)
    function = f"lambda {', '.join(parents)}: " + " + ".join(terms)
    return function


def generate_functions(graph, noise_vars, funct_type='linear'):
    # TODO
    """

    :param graph:
    :param noise_vars: Noise variables N_i specified as f"lambda ... : ..."
    :param funct_type:
    :return:
    """
    functions = {}
    functions.keys()
    for node in graph.nodes:
        parents = list(graph.predecessors(node))
        if funct_type == 'linear':
            # Randomly pick the coefficients
            coeffs = np.random.choice(PRIMES, size=len(parents))
            # coeffs = np.random.randn(len(parents))
            # functions[node] = generate_linear_function(parents, noise_vars[node], coeffs)
            functions[node] = generate_linear_function(parents, f"N_{node}", coeffs)
        elif funct_type == 'polynomial':
            degrees = np.random.randint(1, MAX_DEGREE + 1, size=len(parents))
            coeffs = np.random.choice(PRIMES, size=len(parents))
            # coeffs = np.random.randn(len(parents))
            # functions[node] = generate_polynomial(parents, noise_vars[node], coeffs, degrees)
            functions[node] = generate_polynomial(parents, f"N_{node}", coeffs, degrees)
        else:
            raise ValueError("Unsupported function type. Use 'linear' or 'polynomial'.")

    return functions


# TODO: Extend to other than just fully-observed SCM's
class SCM:
    def __init__(self, input):
        self.nodes, self.G, self.F, self.N = parse_scm(input)
        self.interventions = {}

    def intervene(self, interventions):
        """Perform interventions on multiple variables.

        Interventions can be perfect (constant value) or soft (stochastic function).
        """
        for variable, func in interventions.items():
            self.interventions[variable] = func

    def sample(self, n_samples=1):
        """Generate random samples from the SCM"""
        data = {X_j: np.zeros(n_samples) for X_j in self.G.nodes}

        # Apply interventions
        for X_j, f_j in self.interventions.items():
            # For soft interventions
            if callable(f_j):
                data[X_j] = f_j(n_samples)
            # For perfect interventions
            else:
                data[X_j] = np.full(n_samples, f_j)

        # TODO: Check if the construction of 'data' object makes sense
        # Generate samples in topological node order
        for j, X_j in enumerate(nx.topological_sort(self.G)):
            if X_j in self.interventions:
                continue
            # Generate noise
            # noise = eval(self.N[j])
            noise = parse_noise(self.N)
            # Propagate noise
            f_j = eval(self.F[j])
            pa_j = list(self.G.predecessors(j))
            parent_data = np.array([data[parent] for parent in pa_j])
            if parent_data.size == 0:
                parent_data = np.zeros((0, n_samples))
            else:
                parent_data = parent_data.T
            # Additive noise
            data[X_j] = f_j(*parent_data.T) + noise

        return data

    def abduction(self, L1):
        """Infer the values of the exogenous variables given observational outputs"""
        noise_data = {}
        for X_j in self.G.nodes:
            f_j = eval(self.F[X_j])
            pa_j = list(self.G.predecessors(X_j))
            parents_data = [L1[parent] for parent in pa_j]
            inferred_noise = L1[X_j] - f_j(*parents_data)
            noise_data[X_j] = inferred_noise
        return noise_data

    def counterfactual(self, L1, interventions, n_samples):
        """Compute counterfactual distribution given L1-outputs and an intervention."""
        # Step 1: Abduction - Update the noise distribution given the observations
        noise_data = self.abduction(L1)

        # Step 2: Action - Intervene within the observationally constrained SCM
        self.intervene(interventions)
        L2 = self.sample(n_samples)

        # Step 3: Prediction - Generate samples in the modified model
        L3 = {node: np.zeros(n_samples) for node in self.G.nodes}
        for X_j in nx.topological_sort(self.G):
            if X_j in L2:
                L3[X_j] = L2[X_j]
                continue

            N_j = noise_data[X_j]
            f_j = eval(self.F[X_j])
            pa_j = list(self.G.predecessors(X_j))
            parents_data = [L3[parent] for parent in pa_j]
            L3[X_j] = f_j(*parents_data) + noise_data

        return L3

    def visualize(self):
        pos = nx.spring_layout(self.G)
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

    @classmethod
    def load_from_json(cls, file_path):
        with open(file_path, 'r') as f:
            data = json.load(f)
        data['functions'] = {k: cls.str_to_func(v) for k, v in data['functions'].items()}
        data['noise'] = {k: cls.str_to_func(v) for k, v in data['noise'].items()}
        return cls(data)

    @staticmethod
    def func_to_str(func):
        return func

    @staticmethod
    def str_to_func(func_str):
        return eval(func_str)


def load_graph(filepath):
    with open(filepath, 'r') as f:
        data = json.load(f)
    G = nx.DiGraph()
    G.add_nodes_from(data['nodes'])
    G.add_edges_from(data['edges'])
    return G


def main():
    parser = argparse.ArgumentParser("Structural Causal Model (SCM) operations.")
    parser.add_argument("--graph_type", choices=['chain', 'parallel', 'random'],
                        help="Type of graph structure to generate. Currently supported: ['chain', 'parallel', 'random']")
    # TODO: help info
    parser.add_argument('--noise_types', default='N(0,1)', type=str, nargs='+')
    # parser.add_argument('--noise_type', type=str, nargs='+', default='gaussian',
    #                    choices=['gaussian', 'bernoulli', 'exponential'],
    #                    help="Specify the type of noise distributions. Currently "
    #                         "supported: ['gaussian', 'bernoulli', 'exponential']")
    # TODO: the list must be reshaped from (n_params * n_variables, 1) to (n_params, n_variables) --> dependency: generate_distributions()
    parser.add_argument('--funct_type', type=str, nargs='+', default='linear', choices=['linear', 'polynomial'],
                        help="Specify the function family "
                             "to be used in structural "
                             "equations. Currently "
                             "supported: ['linear', "
                             "'polynomial']")
    parser.add_argument("--n", type=int, required=True, help="Number of (non-reward) nodes in the graph.")
    # Required for --graph_type random
    parser.add_argument("--p", type=int, help="Denseness of the graph / prob. of including any potential edge.")
    parser.add_argument("--pa_n", type=int, default=1, help="Cardinality of pa_Y in G.")
    parser.add_argument("--vstr", type=int, help="Desired number of v-structures in the causal graph.")
    parser.add_argument("--conf", type=int, help="Desired number of confounding variables in the causal graph.")
    parser.add_argument("--intervene", type=str, help="JSON string representing interventions to perform.")
    parser.add_argument("--plot", action='store_true')
    # TODO: Currently no method for re-assigning default source/target paths
    parser.add_argument("--path_graphs", type=str, default=PATH_GRAPHS, help="Path to save/load graph specifications.")
    parser.add_argument("--path_scm", type=str, default=PATH_SCM, help="Path to save/load SCM specifications.")
    parser.add_argument("--path_plots", type=str, default=PATH_PLOTS, help="Path to save the plots.")

    args = parser.parse_args()

    save_path = f"SCM_n{args.n}_{args.graph_type}-graph_{args.funct_type}-functions.json"

    if args.plot:
        plots.draw_scm(save_path)
        return

    if args.noise_types is not None:
        noises = args.noise_types
        if len(noises) != 1 and len(noises) != args.n + 1:
            raise ValueError("Specify either exactly one noise distribution or |X| - many !")

    # TODO: file names for random graphs differ from chain, parallel
    # graph_type = f"random_pa{args.pa_n}_conf{args.conf}_vstr{args.vstr}"
    graph_type = f"{args.graph_type}_graph_N{args.n}"
    file_path = f"{PATH_GRAPHS}/{graph_type}.json"
    if args.graph_type == 'random':
        graph_type = f"random_graph_N{args.n}_paY_{args.pa_n}_p_{args.p}"
        file_path = f"{PATH_GRAPHS}/{graph_type}"
    try:
        graph = load_graph(file_path)
        print("Successfully loaded the graph file.")
    except (FileNotFoundError, UnicodeDecodeError):
        print(f"No such file: {file_path}")
        generate_graph_args = [
            '--graph_type', f"{args.graph_type}",
            '--n', f"{args.n}",
            '--p', f"{args.n}",
            '--pa_n', f"{args.pa_n}",
            # '--vstr', f"{args.vstr}",
            # '--conf', f"{args.conf}",
            '--save'
        ]
        print("Trying again...")
        sys.argv = ['graph_generator.py'] + generate_graph_args
        graph_generator.main()

        graph = load_graph(file_path)

    # TODO: Check if args.n or args.n + 1
    # TODO: Generation of actual distribution functions first during sampling
    # noises = generate_distributions(graph.nodes, args.noise_types, args.noise_params)
    # TODO: noises should be specified not only by the names but N(0, 1), Exp(2), Geo(0.25), Ber(0.5), etc.
    # TODO: 'noises' should be a dictionary indexed by node names
    noises = args.noise_types
    if len(args.noise_types) == 1:
        noises = np.repeat(args.noise_types, args.n + 1)
    functions = generate_functions(graph, noises, args.funct_type)

    scm_data = {
        "nodes": graph.nodes,
        "edges": graph.edges,
        "functions": {k: SCM.func_to_str(v) for k, v in functions.items()},
        "noise": noises
    }
    scm = SCM.SCM(scm_data)

    # f"{PATH_SCM}/SCM_N5_chain_graph_linear_functions_gaussian_noises.json"
    # save_path = PATH_SCM
    scm.save_to_json(save_path)




if __name__ == '__main__':
    main()

"""
Example JSON input
        json_input = '''
{
    "nodes": ["X1", "X2", "Y"],
    "edges": [["X1", "Y"], ["X2", "Y"]],
    "functions": {
        "Y": "lambda x1, x2: 2*x1 + 3*x2"
    },
    "noise": {
        "X1": "np.random.normal(0, 1)",
        "X2": "np.random.normal(0, 1)",
        "Y": "np.random.normal(0, 1)"
    }
}
'''    
"""
