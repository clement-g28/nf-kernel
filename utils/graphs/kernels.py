import numpy as np
from grakel.kernels import WeisfeilerLehman, VertexHistogram, EdgeHistogram, MultiscaleLaplacian, HadamardCode, \
    ShortestPath, ShortestPathAttr, Propagation, PropagationAttr
from grakel.utils import graph_from_networkx


def compute_hadcode_kernel(datasets, base_grakel_kernel='vertexhistogram', normalize=False, edge_to_node=True,
                           attributed_node=False):
    graphs, tags = get_kernel_graphs(datasets, edge_to_node, attributed_node=attributed_node)

    base_kernel = EdgeHistogram if base_grakel_kernel == 'edgehistogram' else VertexHistogram
    graph_kernel = HadamardCode(normalize=normalize, base_graph_kernel=base_kernel)

    K = compute_gram_matrix(graph_kernel, graphs, tags, use_edge_labels=not edge_to_node)
    return K


def compute_mslap_kernel(datasets, normalize=False, edge_to_node=True, attributed_node=False):
    graphs, tags = get_kernel_graphs(datasets, edge_to_node, attributed_node=attributed_node)

    graph_kernel = MultiscaleLaplacian(normalize=normalize)

    K = compute_gram_matrix(graph_kernel, graphs, tags, use_edge_labels=not edge_to_node)
    return K


def compute_propagation_kernel(datasets, normalize=False, edge_to_node=True, attributed_node=False):
    graphs, tags = get_kernel_graphs(datasets, edge_to_node, attributed_node=attributed_node)

    if attributed_node:
        graph_kernel = PropagationAttr(normalize=normalize)
    else:
        graph_kernel = Propagation(normalize=normalize)

    K = compute_gram_matrix(graph_kernel, graphs, tags, use_edge_labels=not edge_to_node)
    return K


def compute_sp_kernel(datasets, normalize=False, edge_to_node=True, attributed_node=False):
    graphs, tags = get_kernel_graphs(datasets, edge_to_node, attributed_node=attributed_node)

    if attributed_node:
        graph_kernel = ShortestPathAttr(normalize=normalize)
    else:
        graph_kernel = ShortestPath(normalize=normalize, with_labels=True)

    K = compute_gram_matrix(graph_kernel, graphs, tags, use_edge_labels=not edge_to_node)
    return K


def compute_wl_kernel(datasets, base_grakel_kernel='vertexhistogram', wl_height=10, normalize=False, edge_to_node=True,
                      attributed_node=False):
    graphs, tags = get_kernel_graphs(datasets, edge_to_node, attributed_node=attributed_node)

    base_kernel = EdgeHistogram if base_grakel_kernel == 'edgehistogram' else VertexHistogram
    graph_kernel = WeisfeilerLehman(n_iter=wl_height, base_graph_kernel=base_kernel, normalize=normalize)

    # K = compute_gram_matrix(graph_kernel, graphs, tags, use_edge_labels=False)
    K = compute_gram_matrix(graph_kernel, graphs, tags, use_edge_labels=not edge_to_node)
    return K


def get_kernel_graphs(datasets, edge_to_node=True, attributed_node=False):
    if isinstance(datasets, tuple):
        graphs = []
        for dset in datasets:
            graphs.append(dset.get_nx_graphs(edge_to_node=edge_to_node, attributed_node=attributed_node))
        graphs = tuple(graphs)
        dataset = datasets[0]
        node_label_tag = dataset.node_labels[0]
        edge_label_tag = dataset.edge_labels[0]
    else:
        dataset = datasets
        node_label_tag = dataset.node_labels[0]
        edge_label_tag = dataset.edge_labels[0]
        graphs = dataset.get_nx_graphs(edge_to_node=edge_to_node, attributed_node=attributed_node)

    return graphs, (node_label_tag, edge_label_tag)


def compute_gram_matrix(kernel, graphs, tags, use_edge_labels=False):
    if isinstance(graphs, tuple):
        assert len(graphs) == 2, 'if graphs is a tuple, it should have a size of 2'
        # assert not isinstance(graphs[0], list) or not isinstance(graphs[0], list), \
        #     'when a tuple is given the two terms should not be a list, ' \
        #     'or else give one list of all graphs to get the full gram matrix'
        graphs = list(graphs)
        if not isinstance(graphs[0], list):
            graphs[0] = [graphs[0]]
        if not isinstance(graphs[1], list):
            graphs[1] = [graphs[1]]
        indices1 = [i for i, _ in enumerate(graphs[0])]
        indices2 = [len(indices1) + i for i, _ in enumerate(graphs[1])]
        graphs = [g for i in range(len(graphs)) for g in graphs[i]]
    else:
        indices1, indices2 = None, None

    # Test if there are invalid graphs (0 nodes)
    mock_g = None
    bad_indices = []
    for i, g in enumerate(graphs):
        if g.number_of_nodes() == 0:
            bad_indices.append(i)
        elif mock_g is None:
            mock_g = g
    graphs = [g if i not in bad_indices else mock_g for i, g in enumerate(graphs)]

    G = graph_from_networkx(graphs, node_labels_tag=tags[0], edge_labels_tag=tags[1]) if use_edge_labels \
        else graph_from_networkx(graphs, node_labels_tag=tags[0])

    K = kernel.fit_transform(G)

    if K.dtype == np.int:
        K = K.astype(np.float32)
    # Replace by NaN invalid graphs values
    if len(bad_indices) > 0:
        K[bad_indices] = np.NaN
        K[:, bad_indices] = np.NaN

    if indices1 is not None:
        K = K[indices1][:, indices2].squeeze()
        # K = K.tolist()

    return K
