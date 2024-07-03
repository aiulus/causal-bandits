import heapq
import json
import networkx as nx

def load_graph_json(path):
    with open(path, 'r') as f:
        data = json.load(f)
    G = nx.Graph()
    G.add_nodes_from(data['nodes'])
    G.add_edges_from(data['edges'])
    return G

def vertex_cover(G):
    cover = set()

    while G.number_of_edges() > 0:
        # Pick the edge with the highest degree
        u, v = max(G.edges(), key=lambda edge: G.degree(edge[0]) + G.degree(edge[1]))
        # Add both endpoints to the cover
        cover.add(u)
        cover.add(v)
        # Remove all edges incident to u or v
        G.remove_node(u)
        G.remove_node(v)

    return cover
# Minimum Spanning Tree (MST) approximation
def MST_Kruskal(G):
    mst = nx.Graph()
    mst.add_nodes_from(G.nodes)
    edges = sorted(G.edges(data=True), key=lambda x: x[2]['weight'])
    uf = nx.utils.UnionFind()

    for u, v, data in edges:
        if uf[u] != uf[v]:
            mst.add_edge(u, v, weight=data['weight'])
            uf.union(u, v)
    return mst

def MST_Prim(G):
    mst = nx.Graph()
    mst.add_nodes_from(G.nodes)
    visited = set()
    start_node = next(iter(G.nodes))
    edges = [(data['weight'], start_node, v) for v, data in G[start_node].items()]
    heapq.heapify(edges)
    visited.add(start_node)

    while edges:
        weight, u, v = heapq.heapify(edges)
        if v not in visited:
            visited.add(v)
            mst.add_edge(u, v, weight=weight)
            for next_v, data in G[v].items():
                if next_v not in visited:
                    heapq.heappush(edges, (data['weight'], v, next_v))
    return mst

# Travelling Salesman Problem (TSP) algorithms
def TSP_Christo(G):
    # Find a MST of G
    T = nx.minimum_spanning_tree(G)

    # Find the vertices with odd degree in T
    odd_nodes = [v for v in T.nodes if T.degree(v) % 2 == 1]

    # Find a minimum-weight perfect matching M in the induced subgraph
    odd_subgraph = G.subgraph(odd_nodes)
    M = nx.algorithms.min_weight_matching(odd_subgraph, maxcardinality=True)

    # Combine T and M to form a multigraph H
    H = nx.MultiGraph(T)
    H.add_edges_from(M)

    # Find a Eulerian circuit in H
    eulerian = list(nx.eulerian_circuit(H))

    # Shortcut the Eulerian circuit to get a Hamiltonian circuit
    visited = set()
    tour = []
    for u, v in eulerian:
        if u not in visited:
            visited.add(u)
            tour.append(u)
    tour.append(tour[0])

    return tour
def TSP_NN(G, start_node=None):
    if start_node is None:
        start_node = list(G.nodes)[0]

    unvisited = set(G.nodes)
    unvisited.remove(start_node)
    tour = [start_node]
    current_node = start_node

    while unvisited:
        next_node = min(unvisited, key=lambda node: G[current_node][node]['weight'])
        unvisited.remove(next_node)
        tour.append(next_node)
        current_node = next_node

    tour.append(start_node)

    return tour
