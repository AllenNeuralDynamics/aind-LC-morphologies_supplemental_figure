"""
    Builds summary graphs of axon trees
"""

from collections import Counter

import matplotlib.pyplot as plt
import networkx as nx
import seaborn as sns
from tqdm import tqdm

import lc_reconstruction_analysis.axon as axon
import lc_reconstruction_analysis.utils as utils


def build_all_trees(
    dataDF,
    graphs,
    pos=None,
    DATA_DIR=None,
    RESULTS_DIR=None,
    df=None,
    graph_on="structure",
    include_other=True,
    roi_version=2,
):
    """
    Build and plot all summary trees for each individual cell
    pos = centroids of the ROIs
    df, dataframe of axon lengths. if provided used to size nodes by
        axon length
    """
    if graph_on not in ["structure", "clean_structure", "allen_id"]:
        raise AssertionError(
            '{} is invalid value for "graph_on"'.format(graph_on)
        )

    for name in list(graphs.keys()):
        try:
            g = graphs[name]
            r = build_tree(
                g,
                DATA_DIR,
                graph_on=graph_on,
                include_other=include_other,
                roi_version=roi_version,
            )
            if df is not None:
                size = get_node_size(df, name, r)
            else:
                size = 600
            plot_tree(r, name, pos, RESULTS_DIR=RESULTS_DIR, node_size=size)
        except Exception as e:
            print("{} : {}".format(name, e))


def get_node_size(df, name, tree, base=7200):
    """
    Determine node size by axon length in df
    """
    node_size = [df.loc[name][x] * base for x in list(tree.nodes)]
    return node_size


def build_tree(
    graph,
    DATA_DIR=None,
    graph_on="structure",
    tolerance=2000,
    include_other=True,
    roi_version=2,
):
    """
    Compute summary tree for a single cell
    """
    if graph_on not in ["structure", "clean_structure", "allen_id"]:
        raise AssertionError(
            '{} is invalid value for "graph_on"'.format(graph_on)
        )

    edges = build_summary_tree(
        graph,
        DATA_DIR,
        graph_on=graph_on,
        tolerance=tolerance,
        include_other=include_other,
        roi_version=roi_version,
    )
    r = nx.DiGraph()
    r = nx.from_edgelist(edges, r)
    return r


def build_combined_tree(
    dataDF,
    graphs,
    DATA_DIR=None,
    roi_version=2,
    weight_type="fraction",
    graph_on="structure",
    tolerance=2000,
    include_other=True,
):
    """
    Build combined tree across all neurons
    can weight edges either by "fraction" or "conditional"
    """
    if graph_on not in ["structure", "clean_structure", "allen_id"]:
        raise AssertionError(
            '{} is invalid value for "graph_on"'.format(graph_on)
        )

    edges = []
    ncells = len(dataDF["Graph"])
    nodes = []
    # Iterate through cells and get nodes/edges
    for name in tqdm(dataDF["Graph"]):
        try:
            this_edge = build_summary_tree(
                graphs[name],
                DATA_DIR=DATA_DIR,
                roi_version=roi_version,
                graph_on=graph_on,
                tolerance=tolerance,
                include_other=include_other,
            )
            this_nodes = set([x for y in this_edge for x in y])
            edges += this_edge
            nodes += this_nodes
        except Exception as e:
            print("{} {}".format(name, e))
    edges = Counter(edges)
    nodes = Counter(nodes)

    # Determine edge weights
    weighted_edges = []
    for edge in edges:
        if weight_type == "fraction":
            weighted_edges.append((edge[0], edge[1], edges[edge] / ncells))
        elif weight_type == "conditional":
            weighted_edges.append(
                (edge[0], edge[1], edges[edge] / nodes[edge[0]])
            )

    # Build graph
    r = nx.DiGraph()
    r.add_weighted_edges_from(weighted_edges)
    if "VS" in r.nodes:
        r.remove_node("VS")
    r.info = {
        "roi_version": roi_version,
        "weight_type": weight_type,
        "graph_on": graph_on,
        "tolerance": tolerance,
    }
    return r


def build_summary_tree(
    graph,
    DATA_DIR=None,
    roi_version=2,
    graph_on="structure",
    tolerance=2000,
    include_other=True,
):
    """
    compute the summary tree based on the presence of an edge
    """

    if graph_on not in ["structure", "clean_structure", "allen_id"]:
        raise AssertionError(
            '{} is invalid value for "graph_on"'.format(graph_on)
        )

    if graph_on == "clean_structure":
        graph = axon.add_clean_structure(
            graph,
            tolerance=tolerance,
            DATA_DIR=DATA_DIR,
            include_other=include_other,
            roi_version=roi_version,
        )
    elif graph_on == "structure":
        graph = axon.annotate_current_structure(
            graph,
            annotate_on="structure",
            DATA_DIR=DATA_DIR,
            roi_version=roi_version,
        )

    region_edges = []
    for u, v in graph.edges:
        if (graph.nodes[u]["structure_id"] in [1, 2]) & (
            graph.nodes[v]["structure_id"] in [1, 2]
        ):
            u_location = graph.nodes[u][graph_on]
            v_location = graph.nodes[v][graph_on]
            edge = (u_location, v_location)
            if (
                (u_location != v_location)
                and (edge not in region_edges)
                and ("NaN" not in edge)
                and (None not in edge)
            ):
                region_edges.append(edge)
    return region_edges


def plot_tree(tree, name, pos=None, RESULTS_DIR=None, node_size=600):
    """
    Plot the summary tree for a single cell
    """

    # Set up figure
    plt.figure(figsize=(10, 4))
    ax = plt.gca()

    # Draw tree
    nx.draw_networkx(
        tree,
        pos,
        node_size=node_size,
        font_color="darkred",
        font_weight="bold",
        node_color="cornflowerblue",
    )

    # Clean up
    plt.title(name)
    plt.xlim([3224, 13500])
    plt.ylim([-6382, -2400])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)

    # Save figure
    if RESULTS_DIR is not None:
        plt.savefig(RESULTS_DIR + name + ".png")


def plot_combined_tree(tree_in, pos, min_weight=0):
    """
    Plot the summary combined tree
    """
    # Set up figure
    plt.figure(figsize=(10, 4))
    ax = plt.gca()

    # Plot the tree
    tree = tree_in.copy()
    to_remove = []
    for u, v in tree.edges:
        if tree.edges[u, v]["weight"] < min_weight:
            to_remove.append((u, v))
    for u, v in to_remove:
        tree.remove_edge(u, v)
    weights = [tree[u][v]["weight"] * 4 for u, v in tree.edges()]
    # weights = [x*4 if x > min_weight else 0 for x in weights]
    nx.draw_networkx(
        tree,
        pos,
        width=weights,
        node_size=600,
        font_color="darkred",
        font_weight="bold",
        node_color="cornflowerblue",
    )
    # Add this to split the directional weights
    # connectionstyle='arc3,rad=0.05'

    # Clean up
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    if min_weight > 0:
        plt.title(
            "Combined, {}, {}, {}, min weight {}".format(
                tree_in.info["weight_type"],
                tree_in.info["graph_on"],
                tree_in.info["tolerance"],
                min_weight,
            )
        )
    else:
        plt.title(
            "Combined, {}, {}, {}".format(
                tree_in.info["weight_type"],
                tree_in.info["graph_on"],
                tree_in.info["tolerance"],
            )
        )
    plt.xlim([3224, 13500])
    plt.ylim([-6382, -2400])


def get_adjacency(tree, roi_version):
    """
    Build adjacency matrix from tree
    """
    sorted_columns = utils.get_roi_list(roi_version)

    A = nx.to_pandas_adjacency(tree)
    sorted_columns = [x for x in sorted_columns if x in A.columns]
    A = A[sorted_columns]
    A = (
        A.reset_index()
        .sort_values(
            by="index",
            key=lambda column: column.map(lambda e: sorted_columns.index(e)),
        )
        .set_index("index", drop=True)
    )
    return A


def plot_combined_adjacency(tree, roi_version=2):
    """
    Plot the combined adjacency matrix
    """
    A = get_adjacency(tree, roi_version)
    plt.figure()
    sns.heatmap(A.T, vmin=0, vmax=1)
    plt.ylabel("To")
    plt.xlabel("From")
    plt.tight_layout()
