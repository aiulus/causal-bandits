import argparse
import ast
import json
import types

import networkx as nx
import numpy as np
from scipy.stats import norm, bernoulli, expon
import matplotlib.pyplot as plt
import sys
import os

sys.path.insert(0, 'C:/Users/aybuk/Git/causal-bandits/src/utils')
import graph_generator

# Set target destination for .json files containing graph structures
PATH_GRAPHS = "../../outputs/graphs"
PATH_SCM = "../../outputs/SCMs"
PATH_PLOTS = "../../outputs/plots"
MAX_DEGREE = 3  # For polynomial function generation


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
            coeffs = np.random.randn(len(parents))
            # functions[node] = generate_linear_function(parents, noise_vars[node], coeffs)
            functions[node] = generate_linear_function(parents, f"N_{node}", coeffs)
        elif funct_type == 'polynomial':
            degrees = np.random.randint(1, MAX_DEGREE + 1, size=len(parents))
            coeffs = np.random.randn(len(parents))
            # functions[node] = generate_polynomial(parents, noise_vars[node], coeffs, degrees)
            functions[node] = generate_polynomial(parents, f"N_{node}", coeffs, degrees)
        else:
            raise ValueError("Unsupported function type. Use 'linear' or 'polynomial'.")

    return functions


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

        # Generate samples in topological node order
        for X_j in nx.topological_sort(self.G):
            if X_j in self.interventions:
                continue
            # Generate noise
            noise = eval(self.N[X_j])
            # Propagate noise
            f_j = eval(self.F[X_j])
            pa_j = list(self.G.predecessors(X_j))
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

    def save_to_json(self, file_path):
        os.makedirs(PATH_SCM, exist_ok=True)
        scm_data = {
            "nodes": [node for node in self.nodes],
            "edges": [edge for edge in self.G.edges],
            "functions": {k: str(v) for k, v in self.F.items()},
            # TODO: Save only the type(s) of distribution for the noises
            "noise": self.N
        }
        with open(file_path, 'w') as f:
            json.dump(scm_data, f, indent=2)
        print(f"SCM saved to {file_path}")

    @staticmethod
    def func_to_str(func):
        return func
    @staticmethod
    def str_to_func(func_str):
        return eval(func_str)
    @classmethod
    def load_from_json(cls, file_path):
        with open(file_path, 'r') as f:
            data = json.load(f)
        data['functions'] = {k: cls.str_to_func(v) for k, v in data['functions'].items()}
        data['noise'] = {k: cls.str_to_func(v) for k, v in data['noise'].items()}
        return cls(data)
    def draw_scm(self):
        G = nx.DiGraph()
        G.add_nodes_from(self.G.nodes)
        G.add_edges_from(self.G.edges)

        pos = nx.planar_layout(self.G)
        # Draw the regular nodes
        nx.draw_networkx_nodes(self.G, pos, node_color='none', node_size=500)
        nx.draw_networkx_labels(self.G, pos, font_color='black', font_size=10)
        # Draw the edges
        nx.draw_networkx_edges(self.G, pos, edge_color='black')

        # Draw noise nodes
        noise_nodes = []
        noise_labels = {}
        index = 1
        # TODO: check this first
        while index < len(self.N):
            dist_type = self.N[index]
            if dist_type == 'N':
                noise_labels[f"N_{index//4 + 1}"] = f"N_{index//4 + 1} ~ Exp({self.N[index+1]}, {self.N[index+2]}"
                index += 3
            elif dist_type == 'Exp':
                noise_labels[f"N_{index // 4 + 1}"] = f"N_{index // 4 + 1} ~ Exp({self.N[index + 1]})"
                index += 2
            elif dist_type == 'Ber':
                noise_labels[f"N_{index // 4 + 1}"] = f"N_{index // 4 + 1} ~ Ber({self.N[index + 1]})"
                index += 2
            noise_nodes.append(f"N_{index//4}")

        # Add noise nodes to the graph
        for i, node in enumerate(self.nodes):
            noise_node = f"N_{i + 1}"
            G.add_node(noise_node)
            G.add_edge(noise_node, node)
            pos[noise_node] = (pos[node][0], pos[node][1] + 0.1)

        # Draw the noise nodes
        nx.draw_networkx_nodes(G, pos, nodelist=noise_nodes, node_shape='o', node_color='black', node_size=500, alpha=0.5)
        nx.draw_networkx_edges(G, pos, edgelist=[(f"N_{i+1}", self.nodes[i]) for i in range(len(self.nodes))], style='dashed', edge_color='black')
        nx.draw_networkx_labels(G, pos, labels=noise_labels, font_color='black', font_size=10)

        # Display the functions next to the graph
        functions = self.F
        function_text = "\n".join([f"{k}: {v}" for k, v in functions.items()])
        plt.text(1.05, 0.5, function_text, ha='left', va='center', transform=plt.gca().transAxes)

        # Show/save the plot
        plt.title("Structural Causal Model")
        os.makedirs(PATH_PLOTS, exist_ok=True)
        plot_filename = os.path.join(PATH_PLOTS, f"SCM_n{len(self.nodes)}_{self.N}-noises.png")
        plt.savefig(plot_filename, bbox_inches='tight')
        # plt.show()
        plt.close()
        print(f"Plot saved to {plot_filename}")




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
    parser.add_argument('--noise_types', type=str, nargs='+')
    #parser.add_argument('--noise_type', type=str, nargs='+', default='gaussian',
    #                    choices=['gaussian', 'bernoulli', 'exponential'],
    #                    help="Specify the type of noise distributions. Currently "
    #                         "supported: ['gaussian', 'bernoulli', 'exponential']")
    # TODO: the list must be reshaped from (n_params * n_variables, 1) to (n_params, n_variables) --> dependency: generate_distributions()
    parser.add_argument('--noise_params', type=float, nargs='+',
                        help="Specify a list of float parameters for the noise distributions."
                             "Length of the provided list must match num_params x num_variables")
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

    args = parser.parse_args()

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
            #'--vstr', f"{args.vstr}",
            #'--conf', f"{args.conf}",
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
    scm = SCM(scm_data)
    save_path = f"{PATH_SCM}/SCM_n{args.n}_{args.graph_type}-graph_{args.funct_type}-functions_{args.noise_types}-noises.json"
    # f"{PATH_SCM}/SCM_N5_chain_graph_linear_functions_gaussian_noises.json"
    # save_path = PATH_SCM
    scm.save_to_json(save_path)

    if args.plot:
        scm.draw_scm()


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
