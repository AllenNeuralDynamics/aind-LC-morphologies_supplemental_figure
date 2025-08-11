"""
    Plotting tools for looking at cell structure
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from mpl_toolkits.mplot3d import axes3d  # noqa: F401

import lc_reconstruction_analysis.utils as utils


def plot_all_soma_location(dataDF, dfLengths, vmax=0.5, ylabel="% of axon"):
    """
    Plot a series of scatter plots of soma location colored by projection
        strength to each brain region
    dataDF - pandas dataframe of cells and soma locations
    dfLengths - pandas dataframe with projection strength to each brain region
    vmax - color bar axis. if None, each brain region is scaled indepdenently
    ylabel - This should reflect the projection strength in dfLengths

    Examples:
    # plot projection % into each brain region, with the same color axis
    visualization.plot_all_soma_location(dataDF, dfLengths)

    # plot projection % into each brain region, with independent color axes
    visualization.plot_all_soma_location(dataDF, dfLengths,vmax=None)

    # plot axon length in each brain region, with the same color axis
    df_unnormalized = clustering.build_length_df(
        dataDF, graphs, DATA_DIR, normalize_df=False
        )
    visualization.plot_all_soma_location(
        dataDF, df_unnormalized, vmax=40000, ylabel='axon length
        )

    # plot axon length in each brain region, with independent color axes
    visualization.plot_all_soma_location(
        dataDF, df_unnormalized, vmax=None, ylabel='axon length
        )
    """
    # Build figure
    fig, ax = plt.subplots(2, 12, figsize=(14, 4))

    # Iterate over brain regions
    for index, c in enumerate(
        [
            "OLF",
            "Isocortex",
            "HPF",
            "CTXsp",
            "CNU",
            "TH",
            "HY",
            "MB",
            "CB",
            "P",
            "MY",
            "Other",
        ]
    ):
        cbar = index == 11
        plot_soma_location(
            dataDF,
            dfLengths,
            c,
            ax=ax[:, index],
            add_cbar=cbar,
            add_rows=index == 0,
            vmax=vmax,
            ylabel=ylabel,
        )


def plot_soma_location(
    dataDF,
    dfLengths,
    color_by,
    vmax=0.5,
    ax=None,
    add_cbar=True,
    add_rows=False,
    ylabel="% of axon",
):
    """
    Plot single brain region scatter plot colored by axon strength
    """

    df = pd.merge(dataDF, dfLengths, left_on="Graph", right_index=True)

    if ax is None:
        fig, ax = plt.subplots(1, 2, figsize=(8, 4))

    if vmax is None:
        cm = ax[0].scatter(
            df["somaML"],
            df["somaDV"],
            c=df[color_by],
            cmap="plasma",
            alpha=0.8,
            vmin=0,
        )
    else:
        cm = ax[0].scatter(
            df["somaML"],
            df["somaDV"],
            c=df[color_by],
            cmap="plasma",
            alpha=0.8,
            vmin=0,
            vmax=vmax,
        )
    if add_cbar:
        cbar = plt.colorbar(cm)
        cbar.set_label(ylabel)

    if vmax is None:
        cm = ax[1].scatter(
            df["somaAP"],
            df["somaDV"],
            c=df[color_by],
            cmap="plasma",
            alpha=0.8,
            vmin=0,
            vmax=vmax,
        )
    else:
        cm = ax[1].scatter(
            df["somaAP"],
            df["somaDV"],
            c=df[color_by],
            cmap="plasma",
            alpha=0.8,
            vmin=0,
            vmax=vmax,
        )
    if add_cbar:
        cbar = plt.colorbar(cm)
        cbar.set_label(ylabel)

    ax[0].set_xticklabels([])
    ax[0].set_yticklabels([])
    ax[0].set_title(color_by)
    ax[1].set_xticklabels([])
    ax[1].set_yticklabels([])
    ax[0].yaxis.set_inverted(True)
    ax[0].set_ylim(bottom=5500)
    ax[1].yaxis.set_inverted(True)
    ax[1].set_ylim(bottom=5500)
    if add_rows:
        ax[0].set_ylabel("Coronal")
        ax[1].set_ylabel("Sagittal")


def plot_cells(dataDF, graphs, ax=None, **kwargs):
    """
    Plot morphology of cells
    dataDF, dataframe of cells
    graphs, graphs of tcells
    """
    if ax is None:
        ax = plt.figure().add_subplot(projection="3d")
    for name in dataDF["Graph"].values:
        ax = plot_cell(graphs[name], ax=ax, **kwargs)
    return ax


def plot_color_map(ax, g, colors):
    """
    Appends a color map legend to the axis
    g is a graph
    colors is a color map dictionary
    """
    xyz = g.nodes[1]["pos"]
    for color in colors:
        ax.plot(*xyz, color=colors[color], label=color)
    ax.legend()


def plot_cell(
    graph,
    ax=None,
    plot_list=["soma", "axon", "dendrites"],
    color=None,
    **kwargs,
):
    """
    Plot morphology of cell
    graph, graph of cells
    plot_list, which structures to plot
    """

    soma = utils.get_subgraph(graph, "structure_id", [1])
    dendrites = utils.get_subgraph(graph, "structure_id", [3])
    axon = utils.get_subgraph(graph, "structure_id", [2])
    center = soma.nodes[1]["pos"]

    if ax is None:
        ax = plt.figure().add_subplot(projection="3d")
    if "dendrites" in plot_list:
        plot_subgraph(
            dendrites, color or "mediumblue", ax, plot_list=["edges"], **kwargs
        )
    if "axon" in plot_list:
        plot_subgraph(
            axon,
            color or "orange",
            ax,
            plot_list=["edges"],
            center=center,
            **kwargs,
        )
    if "soma" in plot_list:
        plot_subgraph(soma, "magenta", ax, plot_list=["nodes"], **kwargs)
        for u, v in graph.edges:
            if u == 1 or v == 1:
                pos1 = graph.nodes[u]["pos"]
                pos2 = graph.nodes[v]["pos"]
                ax.plot(
                    [pos1[0], pos2[0]],
                    [pos1[2], pos2[2]],
                    [pos1[1], pos2[1]],
                    "-",
                    color=color or "magenta",
                    alpha=kwargs.get("alpha", 1),
                )
    ax.set_xlabel("A <--> P", fontsize=12)
    ax.set_zlabel("V <--> D", fontsize=12)
    ax.set_ylabel("L <--> M", fontsize=12)
    ax.invert_zaxis()
    return ax


def plot_subgraph(  # noqa: C901
    graph,
    color,
    ax,
    plot_list=["nodes", "edges"],
    max_radius=None,
    center=None,
    alpha=1,
    max_length=None,
    **kwargs,
):
    """
    Plot just a subgraph
    """
    if (max_radius is not None) and (max_length is not None):
        raise Exception("max_radius and max_length should not both be set")
    if "nodes" in plot_list:
        for node in graph.nodes:
            pos = graph.nodes[node]["pos"]
            ax.plot(pos[0], pos[2], pos[1], "o", color=color, alpha=alpha / 2)
    if "edges" in plot_list:
        for u, v in graph.edges:
            if max_radius is not None:
                pos_u = np.array(graph.nodes[u]["pos"])
                dist_u = np.linalg.norm(pos_u - center)
                pos_v = np.array(graph.nodes[v]["pos"])
                dist_v = np.linalg.norm(pos_v - center)
                if (dist_u > max_radius) or (dist_v > max_radius):
                    continue
            elif max_length is not None:
                if (graph.nodes[u]["wire_length"] > max_length) or (
                    graph.nodes[v]["wire_length"] > max_length
                ):
                    continue
            pos1 = graph.nodes[u]["pos"]
            pos2 = graph.nodes[v]["pos"]
            if "color" in graph.nodes[u]:
                color = graph.nodes[u]["color"]
            ax.plot(
                [pos1[0], pos2[0]],
                [pos1[2], pos2[2]],
                [pos1[1], pos2[1]],
                "-",
                color=color,
                alpha=alpha,
            )


def plot_heatmap(dataDF, dfLengths, cbar_label="",vmax=None):
    """
    Plots a heatmap of cell's projections
    """

    # Copy because we are going to reorder
    sorted_columns = [
        "OLF",
        "Isocortex",
        "HPF",
        "CTXsp",
        "CNU",
        "TH",
        "HY",
        "MB",
        "CB",
        "P",
        "MY",
        "Other",
    ]
    dfPlot = dfLengths[sorted_columns].copy()

    # Determine sort order
    sortProjDict = {}
    for i, col in enumerate(sorted_columns):
        sortProjDict[col] = i

    # sort by projections
    dfPlot["top proj"] = dfPlot.columns[np.argmax(dfPlot, axis=1)]
    dfPlot["top proj"] = dfPlot["top proj"].map(sortProjDict)
    dfPlot = dfPlot.sort_values("top proj")[sorted_columns]

    # Make plot
    plt.figure(figsize=(18, 5))
    sns.heatmap(
        dfPlot.T,
        xticklabels=True,
        yticklabels=True,
        cbar_kws={"label": cbar_label},
        vmax=vmax
    )
    plt.xlabel("Neuron")
    plt.xticks(plt.xticks()[0], labels=[], fontsize=7)
