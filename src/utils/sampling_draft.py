import argparse
import ast
import json
import csv
import os
import sys
import re
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import noises, plots, structural_equations, graph_generator, SCM, io_mgmt

sys.path.insert(0, 'C:/Users/aybuk/Git/causal-bandits/src/utils')

config = io_mgmt.configuration_loader()
PATH_GRAPHS = config['PATH_GRAPHS']
PATH_SCM = config['PATH_SCMs']
PATH_PLOTS = config['PATH_PLOTS']
MAX_DEGREE = config['MAX_POLYNOMIAL_DEGREE']
COEFFS = config['COEFFICIENTS']
DISTS = config['DISTS']

# file_path = "..\..\outputs\SCMs\SCM_n7_chain-graph_polynomial-functions.json"
# file_path = "..\..\outputs\SCMs\SCM_n5_chain-graph_polynomial-functions.json"
# file_path = "..\..\outputs\SCMs\SCM_n7_parallel-graph_linear-functions.json"
# file_path = "..\..\outputs\SCMs\SCM_n7_random-graph_polynomial-functions.json"
# file_path = "..\..\outputs\SCMs\SCM_n7_random-graph_polynomial-functions.json"
#file_path = "..\..\outputs\SCMs\SCM_n10_parallel-graph_polynomial-functions.json"
# file_path = "..\..\outputs\SCMs\SCM_n7_random-graph_polynomial-functions.json"
# file_path = "..\..\outputs\SCMs\SCM_n7_random-graph_polynomial-functions_['N(0,1)']_noises.json"
# file_path = "..\..\outputs\SCMs\SCM_n7_random-graph_polynomial-functions_['N(2,10)']_noises_p0.3.json" # TODO: Didn't work
file_path = "..\..\outputs\SCMs\SCM_n7_chain-graph_polynomial-functions_['N(2,0.5)']_noises_pNone.json"
data_savepath = "..\..\outputs\data\DATA_SCM_n7_chain-graph_polynomial-functions_['N(2,0.5)']_noises_pNone.csv"


def evaluate_structural_equation(function_string, data_dict, noise_dict):
    match_single_arg = re.search(r'lambda\s+(\w+)\s*:', function_string)
    match_multiple_args = re.search(r'lambda\s*\(([^)]*)\)\s*:', function_string)
    match = re.search(r'lambda\s*([^:]+)\s*:', function_string)
    if match_multiple_args:
        input_vars = match_multiple_args.group(1).split(',')
    elif match_single_arg:
        input_vars = [match_single_arg.group(1)]
    elif match:
        input_vars = match.group(1).split(',')
    else:
        raise ValueError(f"Invalid lambda function format: {function_string}")
    # Clean up and map the input variables to data_dict entries
    input_vars = [var.strip() for var in input_vars]

    # Replace 'N_Xi" with the corresponding noise data vectors
    noise_vars = re.findall(r'N\w+', function_string)
    noise_vars = [var[2:] for var in noise_vars]

    print(f"Noise variables: {noise_vars}")  # Debug statement
    print(f"Input variables: {input_vars}")  # Debug statement

    # TODO: ATTENTION! Currently assuming additive noises. Due to storage preferences,
    #  the noises don't appear in the function strings contained in the .json representation of the SCM.

    function_string = re.sub(r'\s*\+\s*N_\w+', '', function_string)  # Remove terms like '+ N_Xi'

    # Define the lambda function
    SE_lambda = eval(function_string)

    # Prepare the arguments
    args = [data_dict[var] for var in input_vars]

    # Evaluate the function element-wise on the input variables
    result = SE_lambda(*args)

    return result


def save_to_csv(dict, path):
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        for key, value in dict.items():
            value_str = ','.join(map(str, value))
            writer.writerow([key, value_str])

def csv_to_dict(path):
    data = {}

    with open(path, mode='r') as f:
        # reader = csv.DictReader(f)
        reader = csv.reader(f)
        for row in reader:
            node = row[0]
            values = list(map(float, row[1].split(',')))
            data[node] = values

    return data


def main():
    parser = argparse.ArgumentParser(description="Generating L1, L2, L3 data from .json files representing SCM's.")
    # parser.add_argument('--file_name', required=True, help="Please specify the name (without path!) of the SCM file.")
   # parser.add_argument('--mode', required=True, choices=['l1', 'l2', 'l3'],
    #                    help="Please provide the distribution type: "
     #                        "'l1' for observational, 'l2' for interventional, 'l3' for counterfactual data.")
    parser.add_argument('--N', type=int, default=1000, help="Number of samples to generate")
    # Required for the option --mode l2 and --mode l3
    parser.add_argument('--do', type=str, nargs='+',
                        help="Please specify the interventions to be performed. Example: "
                             "'--do X_i 0' sets variable X_i to zero.")
    # TODO: not a very practical way of passing info
    parser.add_argument('--observations_path', type=str,
                        help="Provide the path to the JSON file that contains the observations"
                             "to be used for constructing the observationally constrained SCM in mode 'l3'.")
    parser.add_argument('--variables', nargs='+', help="Variables to visualize.")
    parser.add_argument('--plot', action='store_true')

    args = parser.parse_args()

    if args.plot:
        dict = csv_to_dict(data_savepath)
        plots.plot_distributions_from_dict(dict)
        return



    scm = SCM.SCM(file_path)
    print(f"SCM variables: {scm.nodes}\n")
    print(f"Graph ndoes: {scm.G.nodes}\n")
    print(f"Edges: {scm.G.edges}\n")
    print(f"Functions: {scm.F}\n")
    print(f"Noises: {scm.N}")

    G = nx.DiGraph()

    n_samples = 1000

    noise_data = {}

    for X_j in scm.G.nodes:
        print(f"Parsing noise for {X_j}")
        n_j_str = scm.N[X_j]
        print(f"Obtained string representation of N_{X_j}: {n_j_str}")
        samples = noises.generate_distribution(n_j_str)(n_samples)
        # plots.plot_samples(samples, f"20 Samples ~ {file_path}")
        noise_data[X_j] = samples

    print(f"Noise data: {noise_data}")

    data = {}

    for X_j in nx.topological_sort(scm.G):
        print(f"Next node is: {X_j}")
        if scm.G.in_degree[X_j] == 0:
            data[X_j] = noise_data[X_j]
        else:
            f_j_str = scm.F[X_j]
            print(f"Now evaluating {f_j_str}...")
            samples = evaluate_structural_equation(f_j_str, data, noise_data)
            data[X_j] = samples

        print(f"Sampled data: {data}")
        # TODO: name of the .csv file should contain sample size information
#        file_name = file_path.replace('.json', '.csv')
#        save_name = "DATA" + file_name
#        save_path = os.path.join(PATH_DATA, save_name)
        save_to_csv(data, data_savepath)
        # plots.plot_distributions_from_dict(data)
        print(f"Data saved to {data_savepath}")


if __name__ == '__main__':
    main()

def discarded_get_structural_equations(self):
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

def discarded_sample(self, n_samples=1):
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

#    noise_str = [str(noise_str).replace("'", "") for noise_str in args.noise_types]
#    noise_str = ''.join(map(str, noise_str))