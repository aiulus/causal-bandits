import argparse
import json
import os
import random
import networkx as nx
import io_mgmt

config = io_mgmt.configuration_loader()

PATH_GRAPHS = config['PATH_GRAPHS']

def generate_chain_graph(n, save=False):
    nodes = [f"X{i}" for i in range(1, n + 1)] + ["Y"]
    edges = [[nodes[i], nodes[i + 1]] for i in range(n)]
    G = {"nodes": nodes, "edges": edges}
    if save:
        save_graph(G, 'chain', n)
    return G


def generate_parallel_graph(n, save=False):
    nodes = [f"X{i}" for i in range(1, n + 1)] + ["Y"]
    edges = [[nodes[i], "Y"] for i in range(n)]
    G = {"nodes": nodes, "edges": edges}
    if save:
        save_graph(G, 'parallel', n)
    return G


def generate_random_dag(n, p):
    """
    Generate a random DAG G=(V,Ɛ), using the Erdős–Rényi model.
    :param save:
    :param n: Number of nodes in the graph.
    :param p: Probability of including one of the binomial(n, 2) potential edges in Ɛ.
    :return: Graph G.
    """
    # TODO: Rename nodes
    # G = nx.erdos_renyi_graph(n, p, directed=True)
    G = nx.DiGraph()
    # Construct the graph in topological order to ensure DAG'ness
    nodes = [f"X{n}" for n in list(range(1, n + 1))]
    nodes.append('Y')
    G.add_nodes_from(nodes)  # Add nodes
    edges = [(u, v) for u in nodes for v in nodes if u < v and random.random() < p]  # Include edges with probability p
    G.add_edges_from(edges)  # Add edges

    return G


def erdos_with_properties(n, p, n_pa_Y, confs, vstr, save=False):
    """
    Generate a random DAG G=(V,Ɛ) with certain properties using the Erdős–Rényi model.

    :param save:
    :param n: Number of nodes in the graph.
    :param p: Probability of including one of the binomial(n, 2) potential edges in Ɛ.
    :param n_pa_Y: Cardinality of the parent set of reward node Y.
    :param confs: Number of confounding variables.
    :param vstr: Number of v-structures.
    :return: Graph G.
    """
    G = generate_random_dag(n, p)
    print(f"ERDOS: Graph successfully created. Checking for other specifications...\n") # Debug statement
    # Ensure that the number of confounders in the graph is same as 'confs'
    if confs != -1:
        print("Applying number of confounders-constraint.\n") # Debug statement
        n_confs = count_confounders(G)
        while n_confs < confs:
            add_confounders(G)
    else:
        print("No confounders specified.\n") # Debug statement
    # Ensure the specified number of v-structures
    if vstr != -1:
        print("Applying v-structures constraint.\n") # Debug statement
        n_vs = count_v_structures(G)
        while n_vs < vstr:
            add_v_structures(G)
    else:
        print("No v-structures specified.\n") # Debug statement

    # TODO: Fix the inconsistecy between #nodes when n_pa_Y exercised (n+1) and when not (n)
    # Ensure a specific number of parent nodes for the reward variable
    if n_pa_Y != -1:
        y = n
        pa_Y = random.sample(G.nodes, n_pa_Y)
        for parent in pa_Y:
            G.add_edge(parent, y)

    print("Mapping graph to dict\n") # Debug statement
    G_dict = {"nodes": list(G.nodes), "edges": list(G.edges)}

    if save:
        graph_type = f"random_graph_N{n}_paY_{n_pa_Y}_p_{p}"
        print(f"Saving {graph_type}...\n")
        save_graph(G_dict, graph_type, n)
        print(f"Random graph saved to {PATH_GRAPHS}/{graph_type}")

    return G_dict


def count_confounders(G):
    """
    O(n^3) - complex in the number of nodes. Don't use with denser graphs.
    :return: Confounder count in the provided graph.
    """
    n_confounders = 0
    # For each node u in the graph
    for u in G.nodes:
        ch_u = list(G.successors(u))
        # Check if any pair of u's children are connected by a directed edge
        for i in range(len(ch_u)):
            for j in range(i + 1, len(ch_u)):
                v, w = ch_u[i], ch_u[j]
                if nx.has_path(G, v, w) or nx.has_path(G, w, v):
                    n_confounders += 1
                    # One such structure suffices to mark u as a confounder
                    break
    return n_confounders


def count_v_structures(G):
    """
        O(n^3) - complex in the number of nodes. Don't use with denser graphs.
        :return: v-structure count in the provided graph.
        """
    n_v = 0
    for v in G.nodes:
        pa_v = list(G.predecessors(v))
        for i in range(len(pa_v)):
            for j in range(i + 1, len(pa_v)):
                u, w, = pa_v[i], pa_v[j]
                if not G.has_edge(u, w) and not G.has_edge(w, u):
                    n_v += 1
    return n_v


def add_confounders(G):
    """
    Add confounders to the graph
    :param G: DAG G=(V,Ɛ)
    :param num_confounders: number of confounders (Nodes Z s.t. Z-->X, Z-->Y, X-->Y)
    """
    nodes = list(G.nodes())
    # Make num_confounders - many randomly selected edges bidirectional, effectively inserting a confounder
    u, v = random.sample(nodes, 2)
    if not G.has_edge(u, v) and not G.has_edge(v, u):
        G.add_edge(u, v)


def add_v_structures(G):
    """
    Add v-structures to the graph.
    :param G: DAG G=(V,Ɛ)
    :param num_v_structures: number of node triples {X,Y,Z} that form a v-structure: X --> Y <-- Z, X -||- Z
    """
    nodes = list(G.nodes())

    u, v, w = random.sample(nodes, 3)
    if not G.has_edge(u, v) and not G.has_edge(v, u) and not G.has_edge(w, v) and not G.has_edge(v, w):
        G.add_edge(u, v)
        G.add_edge(w, v)


def save_graph(graph, graph_type, n):
    os.makedirs(PATH_GRAPHS, exist_ok=True)
    file_path = f"{PATH_GRAPHS}/{graph_type}_graph_N{n}.json"
    with open(file_path, "w") as f:
        json.dump(graph, f, indent=2)
    print(f"{graph_type.capitalize()} graph with {n} nodes saved to {file_path}.")


def main():
    parser = argparse.ArgumentParser(description="Generate graph structures and save as JSON files.")
    parser.add_argument("--graph_type", choices=['chain', 'parallel', 'random'],
                        help="Type of graph structure to generate. Currently supported: ['chain', 'parallel', 'random']")
    parser.add_argument("--n", type=int, required=True, help="Number of (non-reward) nodes in the graph.")
    # Required for option --graph_type random
    parser.add_argument("--p", type=float, help="Denseness of the graph / prob. of including any potential edge.")
    # Required for option --graph_type random
    parser.add_argument("--pa_n", type=int, default=1, help="Cardinality of pa_Y in G.")
    parser.add_argument("--vstr", type=int, help="Desired number of v-structures in the causal graph.")
    parser.add_argument("--conf", type=int, help="Desired number of confounding variables in the causal graph.")
    parser.add_argument("--save", action='store_true')

    args = parser.parse_args()

    if args.graph_type == 'chain':
        generate_chain_graph(args.n, args.save)
    elif args.graph_type == 'parallel':
        generate_parallel_graph(args.n, args.save)
    elif args.graph_type == 'random':
        if args.p is None:
            print("Please specify the probability of including an edge with --p for random graph generation.")
            return
        if args.pa_n is None:
            print("Please specify the cardinality of the parent set for the reward variable Y.")
        # erdos_with_properties(args.n, args.p, args.pa_n, args.conf, args.vstr, args.save)
        erdos_with_properties(args.n, args.p, args.pa_n, args.conf, args.vstr, args.save)
        # graph_type = f"random_pa{args.pa_n}_conf{args.conf}_vstr{args.vstr}"
    else:
        print("Please specify a type of graph. Currently supported: ['chain', 'parallel', 'random']")
        return


if __name__ == "__main__":
    main()
