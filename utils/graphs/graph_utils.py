import networkx as nx
import matplotlib.pyplot as plt
import random

from PIL import Image
import numpy as np
import glob
import math

from utils.utils import create_folder


def save_nx_graph(G, node_label_map, save_path, n_atom_type, colors, title=None):
    e0 = [(u, v) for (u, v, d) in G.edges(data=True) if d["bond_attr"] == 0]
    e1 = [(u, v) for (u, v, d) in G.edges(data=True) if d["bond_attr"] == 1]
    e2 = [(u, v) for (u, v, d) in G.edges(data=True) if d["bond_attr"] == 2]
    e3 = [(u, v) for (u, v, d) in G.edges(data=True) if d["bond_attr"] == 3]

    nodes = [i for i, n in enumerate(G.nodes) if int(G.nodes[i]['node_attr']) != n_atom_type]
    nodes_c = []
    for i, n in enumerate(G.nodes):
        if int(G.nodes[i]['node_attr']) != n_atom_type:
            # nodes_c.append(int(G.nodes[i]['node_attr']))
            nodes_c.append(colors[int(G.nodes[i]['node_attr'])])
    virtual = [i for i, n in enumerate(G.nodes) if int(G.nodes[i]['node_attr']) == n_atom_type]

    pos = nx.spring_layout(G, k=0.25, iterations=30,
                           seed=7)  # positions for all nodes - seed for reproducibility

    # nodes
    nx.draw_networkx_nodes(G, pos, nodelist=nodes, node_color=nodes_c, node_size=150)
    nx.draw_networkx_nodes(G, pos, nodelist=virtual, alpha=0.1, node_size=50)

    # edges
    nx.draw_networkx_edges(G, pos, edgelist=e0, width=1)
    nx.draw_networkx_edges(G, pos, edgelist=e1, width=2, edge_color='r')
    nx.draw_networkx_edges(G, pos, edgelist=e2, width=3, edge_color='g')
    # nx.draw_networkx_edges(G, pos, edgelist=e3, width=2, alpha=0.01, style="dashed")

    # node labels
    node_labels = nx.get_node_attributes(G, 'node_attr')
    nodes_no_virtual = {}
    for k, v in node_labels.items():
        if not int(v) == n_atom_type:
            label = node_label_map[int(v) + 1]
            nodes_no_virtual[k] = label
    nx.draw_networkx_labels(G, pos, nodes_no_virtual, font_size=10, font_family="sans-serif")
    # edge weight labels
    # edge_labels = nx.get_edge_attributes(G, "bond_attr")
    # edges_no_virtual = {}
    # for k, v in edge_labels.items():
    #     if not v == adj.shape[1] - 1:
    #         edges_no_virtual[k] = v
    # nx.draw_networkx_edge_labels(G, pos, edges_no_virtual)

    ax = plt.gca()
    # ax.margins(0.08)
    if title is not None:
        plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(fname=f'{save_path}.png', format='png')
    plt.close()


def save_nx_graph_attr(G, save_path, title=None):
    e0 = [(u, v) for (u, v, d) in G.edges(data=True) if d["bond_attr"] == 0]
    e1 = [(u, v) for (u, v, d) in G.edges(data=True) if d["bond_attr"] == 1]
    e2 = [(u, v) for (u, v, d) in G.edges(data=True) if d["bond_attr"] == 2]
    e3 = [(u, v) for (u, v, d) in G.edges(data=True) if d["bond_attr"] == 3]

    nodes = [i for i, n in enumerate(G.nodes)]
    nodes_pos = {}
    for i, n in enumerate(G.nodes):
        nodes_pos[i] = G.nodes[i]['node_attr'][:2]

    options_node = {
        "node_color": "skyblue",
    }
    options_edge = {
        "edge_color": [0 for _ in range(len(e0))],
        "width": 4,
        "edge_cmap": plt.cm.Blues_r,
    }
    # nodes
    # nx.draw_networkx_nodes(G, nodes_pos, nodelist=nodes, node_size=10, **options)
    nx.draw_networkx_nodes(G, nodes_pos, nodelist=nodes, **options_node)

    # edges
    # nx.draw_networkx_edges(G, nodes_pos, edgelist=e0, width=10, **options)
    nx.draw_networkx_edges(G, nodes_pos, edgelist=e0, **options_edge)
    # nx.draw_networkx_edges(G, nodes_pos, edgelist=e1, width=2, edge_color='r', **options)
    # nx.draw_networkx_edges(G, nodes_pos, edgelist=e2, width=3, edge_color='g', **options)
    # nx.draw_networkx_edges(G, pos, edgelist=e3, width=2, alpha=0.01, style="dashed")

    if title is not None:
        plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(fname=f'{save_path}.png', format='png', dpi=30)
    plt.close()



