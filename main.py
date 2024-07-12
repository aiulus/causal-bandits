import argparse
import sys
import src.utils.noises as noises
import src.utils.SCM as SCM
import src.utils.structural_equations as structural_equations
import src.utils.plots as plots
import src.utils.graph_generator as graph_generator
import src.utils.sampling as probabilistic_inference


sys.path.insert(0, 'C:/Users/aybuk/Git/causal-bandits/src/utils')

# Set target destination for .json files containing graph structuress
PATH_GRAPHS = "/outputs/graphs"
PATH_SCM = "outputs/SCMs"
PATH_PLOTS = "/outputs/plots"
MAX_DEGREE = 3  # For polynomial function generation
# Set of coefficients to choose from
PRIMES = [-11, -7, -5, -3, -2, 2, 3, 5, 7, 11]
# Command line strings for the surrently supported set of distributions
DISTS = ['N', 'Exp', 'Ber']

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
        graph = SCM.load_graph(file_path)
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
        print(f"Successfully generated {file_path}")
        graph = SCM.load_graph(file_path)
        print(f"Successfully loaded {file_path}")

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
        "functions": {k: SCM.SCM.func_to_str(v) for k, v in functions.items()},
        "noise": {node: (dist_type, *params) for node, (dist_type, params) in noises_dict.items()}
    }
    scm = SCM.SCM(scm_data)

    # f"{PATH_SCM}/SCM_N5_chain_graph_linear_functions_gaussian_noises.json"
    # save_path = PATH_SCM
    scm.save_to_json(save_path)



if __name__ == '__main__':
    main()

