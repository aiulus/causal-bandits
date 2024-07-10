import argparse
import json
import os
import sys

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

# from src.utils import noises, plots, structural_equations, graph_generator
import noises, plots, structural_equations, graph_generator

sys.path.insert(0, 'C:/Users/aybuk/Git/causal-bandits/src/utils')


# Set target destination for .json files containing graph structuress
PATH_GRAPHS = "../../outputs/graphs"
PATH_SCM = "../../outputs/SCMs"
PATH_PLOTS = "../../outputs/plots"
MAX_DEGREE = 3  # For polynomial function generation
# Set of coefficients to choose from
PRIMES = [-11, -7, -5, -3, -2, 2, 3, 5, 7, 11]
# Command line strings for the surrently supported set of distributions
DISTS = ['N', 'Exp', 'Ber']


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

    def get_structural_equations(self):
        F = {}

        noise = noises.parse_noise(self.N)
        noise_dists = noises.generate_distributions(noise)

        # Make sure that each node is parsed after its parents
        for j, X_j in enumerate(nx.topological_sort(self.G)):
            x_j = self.nodes[j]
            pa_j_list = [node for node in self.G.predecessors(x_j)]  # Get the list of parents of j
            n_j = noise_dists.get(j)  # Get the noise term

            if len(pa_j_list) == 0:  # If X_j is a source node, f_j(pa_j, N_j) = N_j
                F[x_j] = n_j
                continue

            # Evaluate the structural equation
            fj_str = list(self.F.items())[j]
            f_j = eval(fj_str[1])

            # Update F // Additive noises
            F[x_j] = lambda x: f_j(x) + n_j(x)

        return F

    def sample(self, n_samples=1):
        """Generate random samples from the SCM by independently sampling noise variables in topological order of the
        causal graph, and recursively propagating the noise distributions according to the structural equations."""

        # Initialize a dictionary {X_j: data ~ P_Xj s.t. |data| = n_samples}
        data = {X_j: np.zeros(n_samples) for X_j in self.G.nodes}

        # Get structural equations
        F = self.get_structural_equations()

        # Make sure that each node is parsed after its parents
        for j, X_j in enumerate(nx.topological_sort(self.G)):
            # TODO: still single arg not populated to |V| - many

            # Generate noise samples for the current node
            # noise_samples = noise_dists.get(j)(n_samples)

            x_j = self.nodes[j]

            f_j = F.get(x_j)

            data[x_j] = f_j(n_samples)

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
    parser.add_argument('--noise_types', default='N(0,1)', type=str, nargs='+', help='--noise_types')
    # parser.add_argument('--noise_type', type=str, nargs='+', default='gaussian',
    #                    choices=['gaussian', 'bernoulli', 'exponential'],
    #                    help="Specify the type of noise distributions. Currently "
    #                         "supported: ['gaussian', 'bernoulli', 'exponential']")
    # TODO: the list must be reshaped from (n_params * n_variables, 1) to (n_params, n_variables) --> dependency: generate_distributions()
    parser.add_argument('--funct_type', type=str, default='linear', choices=['linear', 'polynomial'],
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
        # counts = {distr_str: args.noise_types.count(distr_str) for distr_str in DISTS}
        # arg_count = sum(counts.values())
        arg_count = len(args.noise_types)
        if arg_count != 1 and arg_count != args.n + 1:
            raise ValueError(f"Provided: {args.noise_types}. Invalid number of noise terms: {arg_count}\n"
                             f"Specify either exactly one noise distribution or |X| - many !")

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

    noise_list = [f'{dist}' for dist in args.noise_types]
    if len(noise_list) == 1:
        noise_list *= len(graph.nodes)

    noises_dict = noises.parse_noise(noise_list, list(graph.nodes))
    functions = structural_equations.generate_functions(graph, noises_dict, args.funct_type)

    scm_data = {
        "nodes": graph.nodes,
        "edges": graph.edges,
        "functions": {k: SCM.func_to_str(v) for k, v in functions.items()},
        "noise": {node: (dist_type, *params) for node, (dist_type, params) in noises_dict.items()}
    }
    scm = SCM(scm_data)

    # f"{PATH_SCM}/SCM_N5_chain_graph_linear_functions_gaussian_noises.json"
    # save_path = PATH_SCM
    scm.save_to_json(save_path)


if __name__ == '__main__':
    main()
