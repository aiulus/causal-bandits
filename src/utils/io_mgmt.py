import os
import uuid
import json
import csv
import argparse


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


def args_to_filename(args, file_type, base_path):
    # stip away apostrophes to avoid misinterpretation of user input
    noise_str = ''.join(map(lambda s: s.replace("'", ""), args.noise_types))

    # Specify name components common to all user inputs
    components = [
        f"SCM_n{args.n}",
        f"{args.graph_type}_graph"
    ]

    if args.funct_type is not None:
        components.append(f"{args.funct_type}_functions")
    if args.noise_types is not None:
        components.append(f"{args.noise_types}")

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


def configuration_loader(config_file="global_variables.json"):
    config_path = os.path.join(os.path.dirname(__file__), config_file)
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config
