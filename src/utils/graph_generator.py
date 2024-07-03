import argparse
import json
import os
import random
import networkx as nx

# Set target destination for .json files containing graph structures
PATH_GRAPHS = "../../data/graphs"
def generate_chain_graph(n):
    nodes = [f"X{i}" for i in range(1, n+1)] + ["Y"]
    edges = [[nodes[i], nodes[i+1]] for i in range(n)]
    return {"nodes": nodes, "edges": edges}

def generate_parallel_graph(n):
    nodes = [f"X{i}" for i in range(1, n+1)] + ["Y"]
    edges = [[nodes[i], "Y"] for i in range(n)]
    return {"nodes": nodes, "edges": edges}

def generate_random_dag(n, p):
    """
    Generate a random DAG G=(V,Ɛ), using the Erdős–Rényi model.
    :param n: Number of nodes in the graph.
    :param p: Probability of including one of the binomial(n, 2) potential edges in Ɛ.
    :return: Graph G.
    """
    # G = nx.erdos_renyi_graph(n, p, directed=True)
    G = nx.DiGraph()
    # Construct the graph in topological order to ensure DAG'ness
    nodes = list(range(n))
    G.add_nodes_from(nodes) # Add nodes
    edges = [(u, v) for u in nodes for v in nodes if u < v and random.random() < p] # Include edges with probability p
    G.add_edges_from(edges) # Add edges
    return G

def add_confounders(G, num_confounders):
    """
    Add confounders to the graph
    :param G: DAG G=(V,Ɛ)
    :param num_confounders: number of confounders (Nodes Z s.t. Z-->X, Z-->Y, X-->Y)
    """
    nodes = list(G.nodes())
    # Make num_confounders - many randomly selected edges bidirectional, effectively inserting a confounder
    for _ in range(num_confounders):
        u, v = random.sample(nodes, 2)
        if not G.has_edge(u, v) and not G.has_edge(v, u):
            G.add_edge(u, v)

def add_v_structures(G, num_v_structures):
    """
    Add v-structures to the graph.
    :param G: DAG G=(V,Ɛ)
    :param num_v_structures: number of node triples {X,Y,Z} that form a v-structure: X --> Y <-- Z, X -||- Z
    """
    nodes = list(G.nodes())
    for _ in range(num_v_structures):
        u, v, w = random.sample(nodes, 3)
        if not G.has_edge(u, v) and not G.has_edge(v, u) and not G.has_edge(w, v) and not G.has_edge(v, w):
            G.add_edge(u, v)
            G.add_edge(w, v)

def save_graph(graph, graph_type, n):
    os.makedirs(PATH_GRAPHS, exist_ok=True)
    file_path = f"{PATH_GRAPHS}/{graph_type}_graph_N{n}.json"
    with open(file_path, "w") as f:
        json.dump(graph, f, indent=2)

def main():
    parser = argparse.ArgumentParser(description="Generate graph structures and save as JSON files.")
    parser.add_argument("--graph_type", action="store_true", choices=['chain', 'parallel', 'random'],
                        help="Type of graph structure to generate. Currently supported: ['chain', 'parallel', 'random']")
    parser.add_argument("--n", type=int, required=True, help="Number of (non-reward) nodes in the graph.")
    parser.add_argument("--pa_n", type=int, required=True, default=1, help="Cardinality of pa_Y in G.")
    args = parser.parse_args()

    if args.graph_type == 'chain':
        graph = generate_chain_graph(args.n)
    elif args.graph_type == 'parallel':
        graph = generate_parallel_graph(args.n)
    else:
        print("Please specify a type of graph. Currently supported: ['chain', 'parallel']")
        return

    save_graph(graph, args.graph_type, args.n)
    print(f"{args.graph_type.capitalize()} graph with {args.n} nodes saved to {PATH_GRAPHS}.")

if __name__ == "__main__":
    main()

