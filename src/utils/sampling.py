import argparse
import ast
import json
import csv
from pathlib import Path
import sys
import re
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import noises, plots, structural_equations, graph_generator, SCM, io_mgmt

path_root = Path(__file__).parents[2]
sys.path.append(str(path_root))

config = io_mgmt.configuration_loader()
PATH_GRAPHS = config['PATH_GRAPHS']
PATH_SCM = config['PATH_SCMs']
PATH_PLOTS = config['PATH_PLOTS']
MAX_DEGREE = config['MAX_POLYNOMIAL_DEGREE']
PATH_DATA = config['PATH_DATA']
DISTS = config['DISTS']


def evaluate_structural_equation(function_string, data_dict, noise_dict):
    match_single_arg = re.search(r'lambda\s+(\w+)\s*:', function_string)
    match_multiple_args = re.search(r'lambda\s*\(([^)]*)\)\s*:', function_string)
    match = re.search(r'lambda\s*([^:]+)\s*:', function_string)
    match_constant_function = re.search(r'lambda\s*_\s*:', function_string)
    if match_multiple_args:
        input_vars = match_multiple_args.group(1).split(',')
    elif match_single_arg:
        input_vars = [match_single_arg.group(1)]
    elif match_constant_function:
        input_vars = []
    elif match:
        input_vars = match.group(1).split(',')
    else:
        raise ValueError(f"Invalid lambda function format: {function_string}")
    # Clean up and map the input variables to data_dict entries
    input_vars = [var.strip() for var in input_vars]

    # Replace 'N_Xi" with the corresponding noise data vectors
    noise_vars = re.findall(r'N\w+', function_string)
    noise_vars = [var[2:] for var in noise_vars]

    # TODO: ATTENTION! Currently assuming additive noises. Due to storage preferences,
    #  the noises don't appear in the function strings contained in the .json representation of the SCM.

    function_string = re.sub(r'\s*\+\s*N_\w+', '', function_string)  # Remove terms like '+ N_Xi'

    # Define the lambda function
    SE_lambda = eval(function_string)

    print(f"DATA DICTIONARY: {data_dict}")  # Debug statement

    # Prepare the arguments
    args = [data_dict[var] for var in input_vars if var != '_']

    result = SE_lambda(*args) if args else SE_lambda('_')

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


def sample_L1(scm, n_samples):
    noise_data = {}
    data = {}

    for X_j in scm.G.nodes:
        n_j_str = scm.N[X_j]

        samples = noises.generate_distribution(n_j_str)(n_samples)
        noise_data[X_j] = samples

    for X_j in nx.topological_sort(scm.G):
        if scm.G.in_degree[X_j] == 0:
            data[X_j] = noise_data[X_j]
        else:
            f_j_str = scm.F[X_j]
            samples = evaluate_structural_equation(f_j_str, data, noise_data)
            data[X_j] = samples

    return data


def sample_L2(scm, n_samples, interventions):
    save_path = "L2_samples_"
    interventions_dict = io_mgmt.parse_interventions(interventions)
    scm.intervene(interventions_dict)
    do_suffix = io_mgmt.make_do_suffix(interventions)
    save_path = save_path + do_suffix + ".json"
    print(f"Attempting to save new SCM to: {save_path}")  # Debug statement
    scm.save_to_json(save_path)
    data_savepath = f"{PATH_DATA}/{save_path}".replace('.json', '.csv')

    noise_data = {}
    print(f"INTERVENTIONS: {interventions_dict}")  # Debug statement

    for X_j in scm.G.nodes:
        if X_j in interventions_dict:
            continue
        print(f"Parsing noise for {X_j}")
        n_j_str = scm.N[X_j]
        print(f"Obtained string representation of N_{X_j}: {n_j_str}")
        samples = noises.generate_distribution(n_j_str)(n_samples)
        noise_data[X_j] = samples

    print(f"Noise data: {noise_data}")

    data = {}

    for X_j in nx.topological_sort(scm.G):
        print(f"Next node is: {X_j}")
        if X_j not in interventions_dict and scm.G.in_degree[X_j] == 0:
            data[X_j] = noise_data[X_j]
        else:
            f_j_str = scm.F[X_j]
            print(f"Now evaluating {f_j_str}...")
            samples = evaluate_structural_equation(f_j_str, data, noise_data)
            if X_j in interventions_dict:
                samples = np.repeat(samples, n_samples)
                data[X_j] = samples
            else:
                data[X_j] = samples + noise_data[X_j]

    print(f"Sampled data: {data}")
    save_to_csv(data, data_savepath)
    print(f"Data saved to {data_savepath}")
    return data


def sample_observational_distribution(scm, n_samples, data_savepath):
    noise_data = {}

    for X_j in scm.G.nodes:
        print(f"Parsing noise for {X_j}")  # Debug statement
        n_j_str = scm.N[X_j]
        print(f"Obtained string representation of N_{X_j}: {n_j_str}")  # Debug statement
        samples = noises.generate_distribution(n_j_str)(n_samples)
        noise_data[X_j] = samples

    data = {}

    for X_j in nx.topological_sort(scm.G):
        if scm.G.in_degree[X_j] == 0:
            data[X_j] = noise_data[X_j]
        else:
            f_j_str = scm.F[X_j]
            print(f"Now evaluating {f_j_str}...")  # Debug statement
            samples = evaluate_structural_equation(f_j_str, data, noise_data)
            data[X_j] = samples

    print(f"Sampled data: {data}")  # Debug statement

    save_to_csv(data, data_savepath)
    print(f"Data saved to {data_savepath}")  # Debug statement

    return data


def sample_interventional_distribution(scm, n_samples, data_savepath, interventions):
    interventions_dict = io_mgmt.parse_interventions(interventions)
    scm.intervene(interventions_dict)
    save_path = data_savepath.strip('.json')
    do_suffix = io_mgmt.make_do_suffix(interventions)
    save_path = save_path + do_suffix + ".json"
    print(f"Attempting to save new SCM to: {save_path}")  # Debug statement
    scm.save_to_json(save_path)
    data_savepath = f"{PATH_DATA}/{save_path}".replace('.json', '.csv')

    noise_data = {}
    print(f"INTERVENTIONS: {interventions_dict}")  # Debug statement

    for X_j in scm.G.nodes:
        if X_j in interventions_dict:
            continue
        print(f"Parsing noise for {X_j}")
        n_j_str = scm.N[X_j]
        samples = noises.generate_distribution(n_j_str)(n_samples)
        noise_data[X_j] = samples

    data = {}

    for X_j in nx.topological_sort(scm.G):
        if X_j not in interventions_dict and scm.G.in_degree[X_j] == 0:
            data[X_j] = noise_data[X_j]
        else:
            f_j_str = scm.F[X_j]
            print(f"Now evaluating {f_j_str}...")
            samples = evaluate_structural_equation(f_j_str, data, noise_data)
            if X_j in interventions_dict:
                samples = np.repeat(samples, n_samples)
                data[X_j] = samples
                print(f"Not using noise for {X_j}")
            else:
                data[X_j] = samples + noise_data[X_j]
                print(f"Additive noise considered for variable {X_j}: {noise_data[X_j]}")

    print(f"Sampled data: {data}")
    save_to_csv(data, data_savepath)
    print(f"Data saved to {data_savepath}")
    return data


def main():
    parser = argparse.ArgumentParser(description="Generating L1, L2, L3 data from .json files representing SCM's.")
    parser.add_argument('--file_name', required=True,
                        help="Please specify the name (not the full path!) of the SCM file.")
    parser.add_argument('--mode', required=True, choices=['l1', 'l2', 'l3'],
                        help="Please provide the distribution type: "
                             "'l1' for observational, 'l2' for interventional, 'l3' for counterfactual data.")
    parser.add_argument('--N', type=int, default=1000, help="Number of samples to generate")
    # Required for the option --mode l2 and --mode l3
    parser.add_argument('--do', type=str, nargs='+',
                        help="Please specify the interventions to be performed. Example: "
                             "'--do (Xi,0)' sets variable X_i to zero. For multiple simultaneous interventions," \
                             "use spaces to seperate the individual do()-operations.")
    # TODO: not a very practical way of passing info
    parser.add_argument('--observations_path', type=str,
                        help="Provide the path to the JSON file that contains the observations"
                             "to be used for constructing the observationally constrained SCM in mode 'l3'.")
    parser.add_argument('--variables', nargs='+', help="Variables to visualize.")
    parser.add_argument('--plot', action='store_true')
    parser.add_argument('--n_samples', type=int, default=1000,
                        help="Specify the number of samples to be generated with '--n_samples'")
    parser.add_argument('--save', action='store_true')

    args = parser.parse_args()
    file_path = f"{PATH_SCM}/{args.file_name}"
    data_savepath = f"{PATH_DATA}/{args.file_name}".replace('.json', '.csv')

    if args.plot:
        dictionary = csv_to_dict(data_savepath)
        fig = plots.plot_distributions_from_dict(dictionary)
        if args.save:
            plot_filename = f"{PATH_PLOTS}/{args.file_name}".replace('.json', '.png')
            fig.savefig(plot_filename)
        return

    scm = SCM.SCM(file_path)
    if args.mode == 'l1':
        sample_observational_distribution(scm, args.n_samples, data_savepath)
    if args.mode == 'l2':
        if args.do is None:
            print(f"Please specify at least one intervention using the option --do (Xj, value)")
            return

        sample_interventional_distribution(scm, args.n_samples, args.file_name, args.do)


if __name__ == '__main__':
    main()
