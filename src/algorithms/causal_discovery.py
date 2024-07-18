import  networkx as nx
from typing import Set, List, Tuple, FrozenSet, AbstractSet

from src.utils import SCM, MAB

def intersect_keep_order(list1, list2):
    if not list2:
        return []
    return [item for item in list1 if item in list2]
class CausalGraph:
    def __init__(self, G):
        self.G = G

    def do(self, interventions, overwrite=False):
        intervened_graph = self.copy()
        for variable in interventions:
            intervened_graph.G.remove_incoming_edges(variable)
        if overwrite:
            self.G = intervened_graph
        return intervened_graph

    def DFS(self, node, visited):
        visited.add(node)
        for neighbor in self.G[node]:
            if neighbor not in visited:
                self.DFS(self.G, neighbor, visited)

    def compute_descendants(self, X_i):
        """
        Uses DFS to compute the descendants of a given node.
        """
        visited = set()
        self.DFS(self.G, X_i, visited)
        visited.remove(X_i)
        return visited

    def compute_ancestors(self, X_i):
        """
        Uses DFS to compute the ancestors of a given node.
        """

        def reverse_graph(graph):
            reversed_graph = {node: set() for node in graph}
            for src in graph:
                for dest in graph[src]:
                    reversed_graph[dest].add(src)
            return reversed_graph

        reversed_graph = reverse_graph(self.G)
        visited = set()
        self.DFS(reversed_graph, X_i, visited)
        visited.remove(X_i)

        return visited

    def get_c_component(self, nodes):
        """
        Computes a subgraph with only bidirected edges
        :param nodes:
        :return:
        """
        # TODO: Not consistent with the current implementation as nx.DiGraph
        bidirected_edges = [(u, v) for u, v, d in self.G.edges(data=True) if d.get('type' == 'bidirected')]
        bidirected_subgraph = self.G.edge_subgraph(bidirected_edges).to_undirected()

        c_component = set()
        for node in nodes:
            if node in bidirected_subgraph:
                c_component.update(nx.node_connected_component(bidirected_subgraph, node))

        return c_component

class POMIS:
    def __init__(self, bandit, reward_variables):
        self.scm = bandit.scm
        self.CG = CausalGraph(bandit.scm.G)
        self.reward = reward_variables if reward_variables else 'Y'
        self.muct = {}
        self.IB = {}

    def compute_muct(self):
        t = {self.reward}
        while True:
            t_new = self.CG.compute_descendants(t)
            t_new = self.CG.get_c_component(t_new)
            if t_new == t:
                break
            t = t_new

        # Minimality check
        t_min = t.copy()
        for node in t:
            t_prime = t - {node}
            if (self.CG.compute_descendants(t_prime) == t_prime and
                self.CG.get_c_component(t_prime) == t_prime):
                t_min.discard(node)

        self.muct = t_min
        return t_min

    def compute_IB(self):
        parents = set()
        if len(self.muct) == 0:
            self.compute_muct()
        t_min = self.muct
        for node in t_min:
            parents.update(self.CG.G.predecessors(node))
        IB = parents - t_min

        self.IB = IB
        return IB

    def compute_MIS(self):
        """
        Find all minimal intervention sets for the reward variable 'Y'.
        - Restricts graph to ancestors of Y.
        - Orders variables for recursive MIS search.
        """
        V_minus_y = self.CG.G.nodes - 'Y'
        G = [node for node in self.CG.compute_ancestors('Y')]
        nodes_sorted = list(nx.topological_sort(G))
        candidates = intersect_keep_order(reversed(nodes_sorted), V_minus_y)

    def subMIS(self, intervention_set: FrozenSet[str], candidates):
        """
        Recursive helper function to compute_MIS().
        :param intervention_set:
        :param candidates:
        :return:
        """
        out = frozenset({intervention_set})
        for i, X_i in enumerate(candidates):
            H = self.CG.do({candidates})[self.CG.compute_ancestors('Y')]
            out |= H.subMIS(intervention_set | {candidates}, intersect_keep_order(candidates[i + 1:], H.nodes))

    def subPOMIS(self, candidates, obs=None):
        if obs is None:
            obs = set()

        out = []

        for i, X_i in enumerate(candidates):
            muct, ib = self.compute_muct(), self.compute_IB()
            obs_new = obs | set(candidates[:i])
            if not (ib & obs_new):
                out.append(ib)
                new_ib = intersect_keep_order(candidates[i + 1:], muct)
                if new_ib:
                    out.extend(self.CG.do(ib).subPOMIS([muct | ib], 'Y', new_ib, obs_new))

        return {frozenset(_) for _ in out}
    def run(self):
        an_y = self.CG.compute_ancestors('Y')
        ancestral_graph = self.G[an_y]
        self.CG.G = ancestral_graph
        muct, ib = self.compute_muct(), self.compute_IB()
        H = self.CG.do(ib, True)[muct, ib]
        nodes_sorted = list(nx.topological_sort(self.CG.G))
        return self.subPOMIS(reversed(nodes_sorted), muct - 'Y') | {frozenset(ib)}
