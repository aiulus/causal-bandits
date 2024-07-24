import os
import uuid
import json
import csv
import argparse

import pandas as pd


def append_counter(filename):
    counter = 1
    filename_stripped, file_type = filename.split('.')
    while os.path.exists(f"{filename}/({counter}).{file_type}"):
        counter += 1
    filename_with_counter = f"{filename}/({counter}).{file_type}"

    return filename_with_counter


# Append universally unique identifier
def append_unique_id(filename):
    id = uuid.uuid4()
    filename_stripped, file_type = filename.split('.')
    filename_with_uid = f"{filename}_{id}.{file_type}"

    return filename_with_uid


def scm_args_to_filename(args, file_type, base_path):
    # Specify name components common to all user inputs
    components = [
        f"SCM_n{args.n}",
        f"{args.graph_type}_graph"
    ]

    # Add non-graph related property specifications
    if args.funct_type is not None:
        components.append(f"{args.funct_type}_functions")
    if args.noise_types is not None:
        # stip away apostrophes to avoid misinterpretation of user input
        noise_str = ''.join(map(lambda s: s.replace("'", ""), args.noise_types))
        components.append(f"{noise_str}")

    # Include additional information in the file name when provided
    if args.graph_type == 'random':
        if args.pa_n >= 0:
            components.append(f"paY_{args.pa_n}")
        if args.vstr >= 0:
            components.append(f"vstr_{args.vstr}")
        if args.conf >= 0:
            components.append(f"conf_{args.conf}")
        components.append(f"p{args.p}")

    filename = '_'.join(components) + f".{file_type}"
    return os.path.join(base_path, filename)


def make_do_suffix(do_list):
    suffix = "_do"
    if isinstance(do_list, dict): # FOR DEBUGGING ONLY
        for item in do_list:
            variable, value = item.strip('()').strip(' ').split(',')
            suffix += f"{variable}-{value}"
    return suffix


def parse_interventions(do_list):
    do_dict = {}
    if isinstance(do_list, dict): # FOR DEBUGGING ONLY
        if not isinstance(do_list, list):
            do_list = [do_list]
        for intervention in do_list:
            variable, value = intervention.strip('()').strip(' ').split(',')
            do_dict[variable] = value

    return do_dict


def save_rewards_to_csv(rewards, filename):
    df = pd.DataFrame(rewards)
    filename = filename + ".csv"
    df.to_csv(filename, index=False)
    print(f"Rewards saved to {filename}")


def csv_to_dict(path):
    data = {}
    with open(path, mode='r') as f:
        reader = csv.reader(f)
        for row in reader:
            node = row[0]
            values = list(map(float, row[1].split(',')))
            data[node] = values
    return data


def configuration_loader(config_file="global_variables.json"):
    # TODO: This assumes all scripts that have a dependency on configuration_loader are two levels deep
    config_path = f"../../config/{config_file}"
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config


def process_costs_per_arm(costs, n_arms):
    if isinstance(costs, float):
        costs = [costs]
    if len(costs) == 1:
        costs *= n_arms
    return costs
