import networkx as nx
import matplotlib.pyplot as plt
import random

from PIL import Image
import numpy as np
import glob
import math

import matplotlib.gridspec as gridspec
from utils.utils import create_folder


def save_nx_graph(G, node_label_map, save_path, title, n_atom_type, colors):
    e0 = [(u, v) for (u, v, d) in G.edges(data=True) if d["bond_type"] == 0]
    e1 = [(u, v) for (u, v, d) in G.edges(data=True) if d["bond_type"] == 1]
    e2 = [(u, v) for (u, v, d) in G.edges(data=True) if d["bond_type"] == 2]
    e3 = [(u, v) for (u, v, d) in G.edges(data=True) if d["bond_type"] == 3]

    nodes = [i for i, n in enumerate(G.nodes) if int(G.nodes[i]['atom_type']) != n_atom_type]
    nodes_c = []
    for i, n in enumerate(G.nodes):
        if int(G.nodes[i]['atom_type']) != n_atom_type:
            # nodes_c.append(int(G.nodes[i]['atom_type']))
            nodes_c.append(colors[int(G.nodes[i]['atom_type'])])
    virtual = [i for i, n in enumerate(G.nodes) if int(G.nodes[i]['atom_type']) == n_atom_type]

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
    node_labels = nx.get_node_attributes(G, 'atom_type')
    nodes_no_virtual = {}
    for k, v in node_labels.items():
        if not int(v) == n_atom_type:
            label = node_label_map[int(v) + 1]
            nodes_no_virtual[k] = label
    nx.draw_networkx_labels(G, pos, nodes_no_virtual, font_size=10, font_family="sans-serif")
    # edge weight labels
    # edge_labels = nx.get_edge_attributes(G, "bond_type")
    # edges_no_virtual = {}
    # for k, v in edge_labels.items():
    #     if not v == adj.shape[1] - 1:
    #         edges_no_virtual[k] = v
    # nx.draw_networkx_edge_labels(G, pos, edges_no_virtual)

    ax = plt.gca()
    # ax.margins(0.08)
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(fname=f'{save_path}.png', format='png')
    plt.close()


def organise_data(path, nrow, format='png', type='horizontal', last_part=None, order=None):
    images = []
    if last_part is None:
        last_part = path.split('/')[-1]
        last_part = '_' + last_part
    files = sorted(glob.glob(f"{path}/*{last_part}.{format}"))
    if order is not None:
        files_ordered = []
        for lab in order:
            for file in files:
                if file.split('/')[-1].split('_')[0] in lab:
                    files_ordered.append(file)
        files = files_ordered
    for file in files:
        image = Image.open(file)
        data = np.asarray(image)
        # if len(data.shape) < 3:
        data = np.expand_dims(data, axis=0)
        images.append(data)
    images = np.concatenate(images, axis=0).astype(np.float32)
    images = np.expand_dims(images, axis=1) if len(images.shape) < 4 else images.transpose(0, 3, 1, 2)

    if type == 'vertical':
        ordered = []
        ncol = nrow
        nrow = math.floor(len(files) / ncol)
        for i in range(ncol):
            for j in range(nrow):
                ind = i + j * ncol
                ordered.append(np.expand_dims(images[ind], axis=0))
        ordered = np.concatenate(ordered, axis=0)
        images = ordered

    return images


def format(images, nrow, save_path, row_legends=None, col_legends=None, show=False, res_name=None):
    # plt.style.use('dark_background')

    ncol = math.ceil(images.shape[0] / nrow)
    ratios = [0.25] + [1] * nrow if col_legends is not None else [1] * nrow

    col_add = 1 if row_legends is not None else 0
    row_add = 1 if col_legends is not None else 0

    im_size1 = images.shape[-2]
    im_size2 = images.shape[-1]
    fig = plt.figure(figsize=(2 * im_size2 * ncol / 100, 2 * im_size1 * nrow / 100))
    gspec = gridspec.GridSpec(
        ncols=ncol + col_add, nrows=nrow + row_add, figure=fig, height_ratios=ratios
    )
    cmap = plt.get_cmap("gist_gray")

    for row in range(0, nrow):
        if row_legends is not None:
            ax = plt.subplot(
                gspec[row + row_add, 0], frameon=False, xlim=[0, 1], xticks=[], ylim=[0, 1], yticks=[]
            )
            ax.text(
                1,
                0.7,
                row_legends[row],
                family="Roboto Condensed",
                horizontalalignment="right",
                verticalalignment="top",
            )

        for col in range(0, ncol):
            ax = plt.subplot(
                gspec[row + row_add, col + col_add],
                aspect=1,
                frameon=False,
                xlim=[0, 1],
                xticks=[],
                ylim=[0, 1],
                yticks=[],
            )
            if row * ncol + col < images.shape[0]:
                ax.imshow(images[row * ncol + col].transpose(1, 2, 0).astype(np.uint8), cmap=cmap, extent=[0, 1, 0, 1],
                          vmin=0, vmax=255)
            if col_legends is not None and row == 0:
                ax.text(
                    0.5,
                    1.1,
                    col_legends[col],
                    ha="center",
                    va="bottom",
                    size="small",
                    weight="bold",
                )
    create_folder(f'{save_path}')
    name = res_name if res_name is not None else 'grid'
    plt.savefig(f"{save_path}/{name}.png", bbox_inches='tight')
    if show:
        plt.show()
    plt.close()
