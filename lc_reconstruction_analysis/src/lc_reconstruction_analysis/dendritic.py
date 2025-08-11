"""
Analysis of dendrites
"""

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from mpl_toolkits.axes_grid1 import Divider, Size

import lc_reconstruction_analysis.utils as utils


def plot_heatmap_with_column(dataDF, sholl_df, normalize=False, col="somaDV"):
    """
    Plots heatmap of sholl analysis for each cell with a column of additional
    information
    """
    height = 4
    width = 8
    pre_horz_offset = 1.5
    post_horz_offset = 2.5
    vertical_offset = 0.75
    fig = plt.figure(figsize=(width, height))
    h = [
        Size.Fixed(pre_horz_offset),
        Size.Fixed(width - pre_horz_offset - post_horz_offset),
    ]
    v = [
        Size.Fixed(vertical_offset),
        Size.Fixed(height - vertical_offset - 0.5),
    ]
    divider = Divider(fig, (0, 0, 1, 1), h, v, aspect=False)
    ax = fig.add_axes(
        divider.get_position(), axes_locator=divider.new_locator(nx=1, ny=1)
    )
    h = [Size.Fixed(width - post_horz_offset + 0.25), Size.Fixed(0.25)]
    divider = Divider(fig, (0, 0, 1, 1), h, v, aspect=False)
    cax = fig.add_axes(
        divider.get_position(), axes_locator=divider.new_locator(nx=1, ny=1)
    )

    names = sholl_df["name"].unique()
    radii = np.sort(sholl_df["radii"].unique())
    pivoted_df = sholl_df.pivot(
        index="name", columns="radii", values="intersections"
    )
    x = pivoted_df.values
    if normalize:
        sums = np.sum(x, 1)
        x = x / sums[:, np.newaxis]

    # Unsorted
    EX = [
        np.sum(x[i, :] / np.sum(x[i, :]) * radii) for i in range(0, len(names))
    ]
    x_ex_sort = x[np.argsort(EX), :]
    ax.imshow(x_ex_sort)
    ax.set_ylabel("cells")
    ax.set_xticks(range(0, 101, 10), radii[0:101:10])
    ax.set_xlabel("radius (um)")
    if col == "cluster":
        cmp = ListedColormap(["tab:blue", "tab:orange"])
        cax.imshow(dataDF[col].values[:, np.newaxis], aspect="auto", cmap=cmp)
    else:
        p05 = np.percentile(dataDF[col].values, 5)
        p95 = np.percentile(dataDF[col].values, 95)
        cax.imshow(
            dataDF[col].values[:, np.newaxis],
            aspect="auto",
            vmin=p05,
            vmax=p95,
        )
    cax.set_yticks([])
    cax.set_ylabel(col)
    cax.set_xticks([])
    df = pd.DataFrame()
    df["name"] = pivoted_df.index.values
    df["sholl_expected_value"] = EX
    df = pd.merge(df, dataDF, left_on="name", right_on="Graph")
    return df


def plot_heatmap_sholl(sholl_df, normalize=True):
    """
    Plots a heatmap of sholl analysis for each cell
    """
    names = sholl_df["name"].unique()
    radii = np.sort(sholl_df["radii"].unique())
    x = sholl_df.pivot(
        index="name", columns="radii", values="intersections"
    ).values
    if normalize:
        sums = np.sum(x, 1)
        x = x / sums[:, np.newaxis]

    # Unsorted
    plt.figure()
    plt.imshow(x)
    plt.title("un sorted")
    plt.ylabel("cells")
    plt.gca().set_xticks(range(0, 101, 10), radii[0:101:10])
    plt.xlabel("radius (um)")

    # Sort by expected value
    EX = [
        np.sum(x[i, :] / np.sum(x[i, :]) * radii) for i in range(0, len(names))
    ]
    x_ex_sort = x[np.argsort(EX), :]
    plt.figure()
    plt.imshow(x_ex_sort)
    plt.title("sort by expected value")
    plt.ylabel("cells")
    plt.gca().set_xticks(range(0, 101, 10), radii[0:101:10])
    plt.xlabel("radius (um)")


def plot_individual_sholl(sholl_df):
    """
    Plots each cell's sholl analysis as a curve
    """
    names = sholl_df["name"].unique()
    fig, ax = plt.subplots()
    for name in names:
        temp = sholl_df.query("name == @name").sort_values(by="radii")
        ax.plot(temp.radii.values, temp.intersections.values, "k-", alpha=0.2)
    # plot style
    ax.set_xlim(0, 1000)
    ax.set_ylim(bottom=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlabel("radius (um)", fontsize=14)
    ax.set_ylabel("avg. intersections", fontsize=14)
    ax.tick_params(axis="both", labelsize=12)
    return


def plot_sholl_groups(
    sholl_df, group_col, error="sem", bootstraps=None, ylim_max=9
):
    """
    Plot summary sholl analysis for cells divided into groups
    given by "group_col"
    """
    fig, ax = plt.subplots()
    groups = np.sort(sholl_df[group_col].unique())
    colors = ["tab:blue", "tab:orange"]
    for i, g in enumerate(groups):
        if bootstraps is not None:
            this_boot = bootstraps[g]
        else:
            this_boot = None
        plot_sholl(
            sholl_df[sholl_df[group_col] == g],
            error=error,
            bootstraps=this_boot,
            ax=ax,
            color=colors[i],
            label=g,
        )
    ax.set_ylim(0, ylim_max)


def plot_sholl(
    sholl_df, error="sem", bootstraps=None, ax=None, color="k", label=None
):
    """
    Plot the sholl analysis using the computed intersections
    in sholl_df
    error should be 'sem','std','bootstrap', or 'hierarchical bootstrap'
    """

    if (
        error == "bootstrap" or error == "hierarchical bootstrap"
    ) and bootstraps is None:
        raise Exception("must pass bootstraps if error=bootstraps")

    # get mean and error
    mean_sholl = sholl_df.groupby("radii")["intersections"].mean()
    if error == "sem":
        sem_sholl = sholl_df.groupby("radii")["intersections"].sem()
        lower = mean_sholl - sem_sholl
        upper = mean_sholl + sem_sholl
    elif error == "std":
        std_sholl = sholl_df.groupby("radii")["intersections"].std()
        lower = mean_sholl - std_sholl
        upper = mean_sholl + std_sholl
    elif (error == "bootstrap") or (error == "hierarchical bootstrap"):
        lower = np.array(mean_sholl) - bootstraps["SEM"]
        upper = np.array(mean_sholl) + bootstraps["SEM"]
    else:
        print("unknown error type")

    # Plot mean and error
    if ax is None:
        fig, ax = plt.subplots()
    if label is None:
        label = error
    ax.plot(mean_sholl.index.values, mean_sholl.values, color)
    ax.fill_between(
        mean_sholl.index.values,
        lower,
        upper,
        color=color,
        alpha=0.25,
        label=label,
    )

    # plot style
    ax.set_xlim(0, mean_sholl.index.values[-1])
    ax.set_ylim(bottom=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlabel("radius (um)", fontsize=14)
    ax.set_ylabel("avg. intersections", fontsize=14)
    ax.tick_params(axis="both", labelsize=12)
    ax.legend()


def build_sholl_df(
    dataDF,
    graphs,
    maxRadius=1000,
    stepSize=10,
    analysis_type="dendrites",
    metric="pos",
):
    """
    Builds a tidy dataframe of sholl intersections
    Each row is the number of intersections of a cell at a specific radius
    """
    if analysis_type not in ["dendrites", "axon"]:
        raise Exception("unknown analysis type")

    cells = []

    # Iterate through cells
    for name in dataDF["Graph"]:

        # Get graph for this cell
        graph = graphs[name]

        # Get dendritic and somatic subgraph
        # structure_id:
        # 1 = soma (single point)
        # 2 = axon
        # 3 = dendrite
        if analysis_type == "dendrites":
            sub_graph = utils.get_subgraph(graph, "structure_id", [1, 3])
        elif analysis_type == "axon":
            sub_graph = utils.get_subgraph(graph, "structure_id", [1, 2])

        # Perform sholl analysis
        intDict = sholl_analysis(
            sub_graph,
            sub_graph.nodes[1]["pos"],
            maxRadius,
            stepSize,
            metric=metric,
        )
        cell_df = pd.DataFrame()
        cell_df["radii"] = list(intDict.keys())
        cell_df["intersections"] = list(intDict.values())
        cell_df["name"] = [name] * len(cell_df)
        cells.append(cell_df)

    sholl_df = pd.concat(cells)
    return sholl_df


def sholl_analysis(graph, center, max_radius, step=1, metric="pos"):
    """
    Perform Sholl analysis on a neuron morphology graph.

    Args:
        graph (nx.DiGraph): Neuron morphology as a directed graph.
        center (tuple): Center of the analysis (x, y, z).
        max_radius (float): Maximum radius for Sholl analysis.
        step (float): Step size for radii.

    Returns:
        dict: radii as keys and intersection counts as values.
    """
    # Extract node positions
    distances = nx.get_node_attributes(graph, metric)
    if metric == "pos":
        for k in distances:
            distances[k] = np.linalg.norm(np.array(distances[k]) - center)

    # Compute intersections
    intersections = {}
    radii = np.arange(0, max_radius + step, step)
    for radius in radii:
        intersections[radius] = 0

    for u, v in graph.edges:
        # Check if the edge intersects the sphere of the given radius
        if (distances[u] > (max_radius + step)) and (
            distances[v] > (max_radius + step)
        ):
            continue
        for radius in radii:
            if (distances[u] <= radius < distances[v]) or (
                distances[v] <= radius < distances[u]
            ):
                intersections[radius] += 1
    return intersections


def bootstrap_sholl(sholl_df, nboots=100):
    """
    Bootstraps
    """
    radii = sholl_df["radii"].unique()

    bootstraps = {}
    summary = {}
    summary["radii"] = radii
    summary["mean"] = []
    summary["SEM"] = []
    for r in radii:
        bootstraps[r] = []
        temp = sholl_df.query("radii==@r")
        for i in range(0, nboots):
            bootstraps[r].append(
                temp["intersections"].sample(frac=1, replace=True).mean()
            )
        summary["mean"].append(np.mean(bootstraps[r]))
        summary["SEM"].append(np.std(bootstraps[r]))

    return bootstraps, summary


def bootstrap_sholl_hierarchically(sholl_df, nboots=1000):
    """
    Perform bootstrapping hierarchically over subjects, then cells
    """
    radii = sholl_df["radii"].unique()
    sholl_df["subject"] = [x.split("-")[1] for x in sholl_df["name"]]
    subjects = sholl_df["subject"].unique()

    bootstraps = {}
    summary = {}
    summary["radii"] = radii
    summary["mean"] = []
    summary["SEM"] = []
    for r in radii:
        bootstraps[r] = []
        temps = []
        for s in subjects:
            temps.append(sholl_df.query("(radii==@r) and (subject==@s)"))
        for i in range(0, nboots):
            subject_samples = np.random.randint(
                len(subjects), size=len(subjects)
            )
            this_values = []
            for s in subject_samples:
                this_values += list(
                    temps[s]["intersections"]
                    .sample(frac=1, replace=True)
                    .values
                )
            bootstraps[r].append(np.mean(this_values))
        summary["mean"].append(np.mean(bootstraps[r]))
        summary["SEM"].append(np.std(bootstraps[r]))
    return bootstraps, summary


def bootstrap_groups(sholl_df, group_col, nboots=1000, hierarchical=True):
    """
    Perform bootstraps for each group defined by "group_col"
    """
    boots = {}
    summary = {}
    groups = sholl_df[group_col].unique()
    for g in groups:
        this_sholl = sholl_df[sholl_df[group_col] == g].copy()
        if hierarchical:
            this_bootstraps, this_summary = bootstrap_sholl_hierarchically(
                this_sholl, nboots
            )
        else:
            this_bootstraps, this_summary = bootstrap_sholl(this_sholl, nboots)
        boots[g] = this_bootstraps
        summary[g] = this_summary
    return boots, summary
