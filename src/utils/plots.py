import json
import matplotlib.pyplot as plt
import networkx as nx
import os

import io_mgmt

config = io_mgmt.configuration_loader()
PATH_GRAPHS = config['PATH_GRAPHS']
PATH_SCM = config['PATH_SCMs']
PATH_PLOTS = config['PATH_PLOTS']
MAX_DEGREE = config['MAX_POLYNOMIAL_DEGREE']
COEFFS = config['COEFFICIENTS']
DISTS = config['DISTS']


def draw_scm(scm_filename):
    scm_path = os.path.join(PATH_SCM, scm_filename)
    try:
        with open(scm_path, 'r') as f:
            scm_data = json.load(f)
    except FileNotFoundError:
        print(f"The file at {scm_path} does not exist.")
        return

    G = nx.DiGraph()
    G.add_nodes_from(scm_data['nodes'])
    G.add_edges_from(scm_data['edges'])

    # TODO: Best layouts for various types of graphs / possibility to specify the layout / custom layout
    # other good options 'nx.spiral_layout(G)', ' nx.arf_layout(G)'

    try:  # Define the layout
        pos = nx.planar_layout(G)
    except nx.NetworkXException:
        pos = nx.shell_layout(G)

    # Draw the regular nodes
    nx.draw_networkx_nodes(G, pos, node_color='none', edgecolors='black', node_size=1000)
    nx.draw_networkx_labels(G, pos, font_color='black', font_size=10)

    # Draw the edges
    nx.draw_networkx_edges(G, pos, edge_color='black', arrows=True,
                           min_source_margin=14.5, min_target_margin=14.5, arrowsize=15)

    # Draw the noise node
    noise = scm_data['noise']
    noise_nodes = []
    noise_labels = {}

    for i, node in enumerate(scm_data['nodes']):
        noise_node = f"N_{i + 1}"
        noise_nodes.append(noise_node)
        G.add_node(noise_node)
        G.add_edge(noise_node, node)
        pos[noise_node] = (pos[node][0], pos[node][1] + 1)
        noise_labels[node] = noise_node

    # TODO : Create labels for noise nodes
    # Draw the noise nodes
    nx.draw_networkx_nodes(G, pos, nodelist=noise_nodes, node_shape='o', node_color='white',
                           edgecolors='black', node_size=1000, alpha=0.5)
    nx.draw_networkx_edges(G, pos,
                           edgelist=[(f"N_{i + 1}", scm_data['nodes'][i]) for i in range(len(scm_data['nodes']))],
                           style='dashed', edge_color='black', arrows=True,
                           min_source_margin=14.5, min_target_margin=14.5, arrowsize=15)

    # Display the functions next to the graph
    functions = scm_data['functions']
    functions_text = "\n".join([f"{k}: {v}" for k, v in functions.items()])
    # TODO: partially overlaps with the DAG, needs more flexible positioning
    plt.text(1.05, 0.5, functions_text, ha='left', va='center')

    # Save/show the plot
    # TODO: More informative but concise title, better formatted
    plt.title("Structural Causal Model")
    os.makedirs(PATH_SCM, exist_ok=True)
    plot_filename = scm_filename.replace('.json', '.png')
    plot_filename = os.path.join(PATH_PLOTS, plot_filename)
    plt.savefig(plot_filename, bbox_inches='tight')
    print(f"Plot saved to {plot_filename}")
    plt.show()
    plt.close()


def plot_samples(samples, title, bins=30, xlabel="x", ylabel="f(x)"):
    plt.figure(figsize=(10, 6))
    plt.hist(samples, bins=bins, edgecolor='k', alpha=0.7)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.show()


def plot_distributions_from_dict(dict, save=False):
    num_plots = len(dict)
    num_cols = 2  # Select the number of columns in the grid
    num_rows = (num_plots + num_cols - 1) // num_cols

    # Create a grid for the plots
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, num_rows * 5))
    axes = axes.flatten()

    for idx, (key, values) in enumerate(dict.items()):
        axes[idx].hist(values, bins=30, edgecolor='k', alpha=0.7)
        axes[idx].set_title(key)
        axes[idx].set_xlabel('value')
        axes[idx].set_ylabel('Frequency')
        # axes[idx].set_yscale('log')

    # Hide unused plots
    for j in range(idx + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()

    return fig


def plot_rewards(rewards, alg_type, bandit_type, plot_filename, save=False):
    plot_filename = plot_filename + ".png"
    title = "Rewards Over Time"
    subtitle = f"Algorithm used: {alg_type}\n Bandit type: {bandit_type}"
    plt.figure(figsize=(10, 6))
    plt.plot(rewards)
    plt.xlabel("Iterations")
    plt.ylabel("Reward")
    plt.title(title)
    plt.suptitle(subtitle)
    plt.grid(True)
    if save:
        plt.savefig(plot_filename, bbox_inches='tight')
    plt.show()
