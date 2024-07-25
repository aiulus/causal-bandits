import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats.distributions import norm, bernoulli, beta, gamma, expon
import os
import json
import argparse
import SCM

# Set target destination for .json files containing graph structures
PATH_GRAPHS = "../../outputs/graphs"
PATH_SCM = "../../outputs/SCMs"
PATH_PLOTS = "../../outputs/plots"
PATH_DATA = "../../outputs/data"


# Draws samples from the observational distribution of the provided SCM
def sample_L1(scm, n_samples):
    return scm.sample_L1(n_samples)


# Draws samples from the interventional distribution of the provided SCM
def sample_L2(scm, interventions, n_samples):
    scm.intervene(interventions)
    return scm.sample_L1(n_samples)


# Draws samples from the counterfactual distribution of the provided SCM
def sample_L3(scm, observations, interventions, n_samples):
    return scm.counterfactual(observations, interventions, n_samples)


def plot_distribution(data, variables, title):
    df = pd.DataFrame(data)
    if len(variables) == 1:
        df[variables[0]].hist(bins=30)
    else:
        pd.plotting.scatter_matrix(df[variables], alpha=0.2)
    plt.title(title)
    file_path = os.path.join(PATH_PLOTS, title + ".png")
    plt.savefig(file_path)
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Run probabilistic inference for a given SCM.")
    parser.add_argument('--file_name', required=True, help="Please specify the name (without path!) of the SCM file.")
    parser.add_argument('--mode', required=True, choices=['l1', 'l2', 'l3'],
                        help="Please provide the distribution type: "
                             "'l1' for observational, 'l2' for interventional, 'l3' for counterfactual data.")
    parser.add_argument('--N', type=int, default=1000, help="Number of samples to generate")
    # Required for the option --mode l2 and --mode l3
    parser.add_argument('--do', type=str, nargs='+',
                        help="Please specify the interventions to be performed. Example: "
                             "'--do X_i 0' sets variable X_i to zero.")
    parser.add_argument('--observations_path', type=str,
                        help="Provide the path to the JSON file that contains the observations"
                             "to be used for constructing the observationally constrained SCM in mode 'l3'.")
    parser.add_argument('--variables', nargs='+', help="Variables to visualize.")
    parser.add_argument('--plot', action='store_true')

    args = parser.parse_args()

    path = os.path.join(PATH_SCM, args.file_name)
    # scm_data = SCM.parse_scm(path)
    # scm = SCM(scm_data)
    with open(path, 'r') as f:
        scm_json = f.read()

    # scm_data = SCM.parse_scm(scm_json)
    scm = SCM.SCM(scm_json)

    if args.mode == 'l1':
        data = sample_L1(scm, args.N)
    elif args.mode == 'l2':
        interventions = json.loads(args.do)
        data = sample_L2(scm, interventions, args.N)
    elif args.mode == 'l3':
        interventions = json.loads(args.do)
        observations = json.loads(args.observations_path)
        data = sample_L3(scm, observations, interventions, args.N)
    else:
        raise ValueError("Unsupported mode. Please choose from ['l1', 'l2', 'l3'].")

    file_name = args.file_name.replace('.json', '.csv')
    save_name = f"{args.mode}".capitalize() + "_data_" + file_name
    file_path = os.path.join(PATH_DATA, save_name)
    data.to_csv(file_path, index=False)
    print(f"Data saved to {file_path}")

    if args.plot:
        plot_distribution(data, args.variables, args.mode)


if __name__ == '__main__':
    main()
