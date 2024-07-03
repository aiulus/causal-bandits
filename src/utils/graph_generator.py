import argparse
import json
import os

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

def save_graph(graph, graph_type, n):
    os.makedirs(PATH_GRAPHS, exist_ok=True)
    file_path = f"{PATH_GRAPHS}/{graph_type}_graph_N{n}.json"
    with open(file_path, "w") as f:
        json.dump(graph, f, indent=2)

def main():
    parser = argparse.ArgumentParser(description="Generate graph structures and save as JSON files.")
    parser.add_argument("--graph_type", action="store_true", choices=['chain', 'parallel'],
                        help="Type of graph structure to generate. Currently supported: ['chain', 'parallel']")
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

