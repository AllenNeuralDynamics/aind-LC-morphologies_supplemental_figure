"""
   Collection of tools for building markov models of LC axon distributions
"""

import os
from collections import Counter, deque
from functools import partial
from multiprocessing import Pool
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import CensoredData, weibull_min
from tqdm import tqdm

import lc_reconstruction_analysis.axon as axon
import lc_reconstruction_analysis.utils as utils


def build_transition_matrix(dataDF, graphs, DATA_DIR, roi_version=2,tolerance=2000):
    """
    Build a markov transition matrix from the data
    """

    # Count edges, terminals, and branches
    edges = Counter()
    terminals = Counter()
    branches = Counter()

    # Iterate over cells
    for name in tqdm(graphs):
        try:
            # Add clean structure annotations
            g = graphs[name]
            g = axon.add_clean_structure(
                g, tolerance=tolerance, DATA_DIR=DATA_DIR, roi_version=roi_version
            )

            # Grab the edges in this cell
            this_edges = [
                (g.nodes[u]["clean_structure"], g.nodes[v]["clean_structure"])
                for u, v in g.edges
            ]
            edges.update(this_edges)

            # Grab the terminals in this cell
            this_terminals = [
                g.nodes[node]["clean_structure"]
                for node in g.nodes
                if g.out_degree(node) == 0
            ]
            terminals.update(this_terminals)

            # Grab the branches in this cell
            this_branches = [
                g.nodes[node]["clean_structure"]
                for node in g.nodes
                if g.out_degree(node) > 1
            ]
            branches.update(this_branches)
        except Exception as e:
            print("{} {}".format(name, e))
    return edges, terminals, branches


def manually_clean_transitions(
    edges, terminals, branches, remove=["SC", "P_to_cortex", "TH_to_cortex"]
):
    """
    Manually remove transition matrix entries based on prior
    anatomical knowledge
    """
    to_delete = []
    for e in edges:
        if (
            ("SC" in remove)
            and (e[0] == "Other")
            and (e[1] not in ["MY", "P", "Other", "CB", "cbf"])
        ):
            count = edges[e]
            edges[("Other", "Other")] += count
            to_delete.append(e)
        elif (
            ("SC" in remove)
            and (e[1] == "Other")
            and (e[0] not in ["MY", "P", "Other", "cbf", "CB"])
        ):
            count = edges[e]
            edges[(e[0], e[0])] += count
            to_delete.append(e)
        elif (
            ("P_to_cortex" in remove)
            and (e[0] in ["P", "MY"])
            and (e[1] in ["HPF", "Isocortex", "OLF"])
        ):
            count = edges[e]
            edges[(e[0], e[0])] += count
            to_delete.append(e)
        elif (
            ("P_to_cortex" in remove)
            and (e[1] in ["P", "MY"])
            and (e[0] in ["HPF", "Isocortex", "OLF"])
        ):
            count = edges[e]
            edges[(e[0], e[0])] += count
            to_delete.append(e)
        elif (
            ("TH_to_cortex" in remove)
            and (e[0] in ["HY", "MB", "TH"])
            and (e[1] in ["Isocortex", "OLF"])
        ):
            count = edges[e]
            edges[(e[0], e[0])] += count
            to_delete.append(e)
        elif (
            ("TH_to_cortex" in remove)
            and (e[1] in ["HY", "MB", "TH"])
            and (e[0] in ["Isocortex", "OLF"])
        ):
            count = edges[e]
            edges[(e[0], e[0])] += count
            to_delete.append(e)

    for e in to_delete:
        del edges[e]

    return edges, terminals, branches


def build_summary(
    edges,
    terminals,
    branches,
    roi_version=3,
    clean_transitions=["SC", "P_to_cortex", "TH_to_cortex"],
):
    """
    Build a summary dataframe of branching and termination rates in each
    brain structure, as well as transition matrices
    """
    if len(clean_transitions) > 0:
        edges, terminals, branches = manually_clean_transitions(
            edges.copy(), terminals, branches, remove=clean_transitions
        )

    branches = pd.DataFrame.from_dict(
        dict(branches), orient="index", columns=["branches"]
    )
    terminals = pd.DataFrame.from_dict(
        dict(terminals), orient="index", columns=["terminals"]
    )
    lengths = {edge[0]: edges[edge] for edge in edges if edge[0] == edge[1]}
    lengths = pd.DataFrame.from_dict(
        dict(lengths), orient="index", columns=["lengths"]
    )

    summary = pd.concat([lengths, terminals, branches], axis=1)
    summary["t/l"] = summary["terminals"] / summary["lengths"]
    summary["b/l"] = summary["branches"] / summary["lengths"]
    summary["t/l_CI"] = (
        1.96
        / np.sqrt(summary["lengths"])
        * np.sqrt(summary["t/l"] * (1 - summary["t/l"]))
    )
    summary["b/l_CI"] = (
        1.96
        / np.sqrt(summary["lengths"])
        * np.sqrt(summary["b/l"] * (1 - summary["b/l"]))
    )
    # We correct lengths here because lengths are in units of wire_lengths
    # and branch/terminal counts are in units of annotation steps
    # which are roughly 10x off from each other
    summary["t/l"] = summary["t/l"] * 0.1
    summary["b/l"] = summary["b/l"] * 0.1
    summary["t/l_CI"] = summary["t/l_CI"] * 0.1
    summary["b/l_CI"] = summary["b/l_CI"] * 0.1

    sorted_columns = utils.get_roi_list(roi_version)
    sorted_columns = [x for x in sorted_columns if x in summary.index.values]
    summary = summary.reindex(sorted_columns)

    A = pd.DataFrame()
    for edge in dict(edges):
        A.at[edge[1], edge[0]] = edges[edge]
    A = A.fillna(0)
    A = A[sorted_columns]
    A = A.reindex(sorted_columns)
    An = A / np.sum(A, axis=0)

    return A, An, summary


def plot_transition_matrix(An, vmax=0.001):
    """
    Plot a heatmap of the transition matrix
    """
    plt.figure()
    sns.heatmap(An, vmax=vmax)
    plt.xlabel("From")
    plt.ylabel("To")
    plt.tight_layout()


def plot_rates(summary):
    """
    plot branching and termination rates in each brain structure

    """
    plt.figure()
    plt.errorbar(
        summary["t/l"],
        summary["b/l"],
        summary["t/l_CI"],
        summary["b/l_CI"],
        alpha=0.25,
        fmt="none",
    )
    plt.plot(summary["t/l"], summary["b/l"], "ko", alpha=0.5)
    plt.plot([0.0007, 0.002], [0.0007, 0.002], "k--", alpha=0.25)
    plt.xlim(0.0007, 0.002)
    plt.ylim(0.0007, 0.002)
    plt.axis("square")
    plt.xlabel("terminals / length")
    plt.ylabel("branches / length")
    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for index, row in summary.iterrows():
        ax.annotate(
            row.name, (row["t/l"] + 0.00002, row["b/l"] - 0.00002), color="b"
        )


def generate_segment_v1(
    A, summary, total_length=0, start="P", max_steps=1000000
):
    """
    Helper method for pure markov model generation
    """
    path = [start]
    branches = []
    for i in range(max_steps):
        path.append(np.random.choice(A.index.values, p=A[path[-1]].values))
        if np.random.rand() < summary["t/l"][path[-1]]:
            break
        if np.random.rand() < summary["b/l"][path[-1]]:
            branches.append(path[-1])
    return dict(Counter(path)), branches


def generate_cell_v1(
    A, summary, random_seed=None, start="P", max_steps=1000000
):
    """
    Generate a cell with markov model and fixed branching/termination rates
    """
    if random_seed is not None:
        np.random.seed(random_seed)
        # np.random.seed((os.getpid() * int(time.time())) % 123456789)
    branches = [start]
    counter = Counter()
    num_branches = 1
    while (len(branches) > 0) and (
        np.sum([counter[areas] for areas in counter.keys()]) < max_steps
    ):
        this_counter, this_branches = generate_segment_v1(
            A,
            summary,
            counter.total(),
            start=branches.pop(),
            max_steps=max_steps,
        )
        counter.update(this_counter)
        branches += this_branches
        num_branches += len(this_branches)
    return counter, num_branches


def generate_cell_v2(
    A, summary, length_summary, random_seed=None, start="P", max_length=1000000
):
    """
    Generate a cell with markov transition matrix, and wire length
    dependent branching and termination rates
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    # np.random.seed((os.getpid() * int(time.time())) % 123456789)

    # Keep track of counts, branches, deaths in each area
    counter = {}
    branches = {}
    terminals = {}
    for area in summary.index.values:
        counter[area] = 0
        branches[area] = 0
        terminals[area] = 0

    # Set up queue to process branches in breadth first order
    queue = deque()
    queue.append((start, 0))
    counter[start] += 1
    total_length = 0

    regions = A.index.values
    p = {}
    for col in A.columns:
        p[col] = A[col].values

    while (len(queue) > 0) and (total_length < max_length):
        (node, this_length) = queue.popleft()
        total_length += 1
        this_length += 1

        # Check for termination
        if np.random.rand() < length_summary["is_leaf"][this_length]:
            terminals[node] += 1
        else:
            next_node = np.random.choice(regions, p=p[node])
            counter[next_node] += 1
            queue.append((next_node, this_length))

        # Check for branching
        if np.random.rand() < length_summary["is_branch_point"][this_length]:
            branches[node] += 1
            queue.append((node, this_length))
    return counter, branches, terminals


def generate_cell_v3(
    A,
    summary,
    length_summary,
    random_seed=None,
    start="P",
    max_length=1000000,
    min_length=0,
):
    """
    Generate a cell with markov transition matrix, and wire length
    dependent branching and termination rates
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    # np.random.seed((os.getpid() * int(time.time())) % 123456789)

    # Keep track of counts, branches, deaths in each area
    counter = {}
    branches = {}
    terminals = {}
    for area in A.index.values:
        counter[area] = 0
        branches[area] = 0
        terminals[area] = 0

    # Set up queue to process branches in breadth first order
    queue = deque()
    queue.append((start, 0))
    counter[start] += 1
    total_length = 0

    regions = A.index.values
    p = {}
    for col in A.columns:
        p[col] = A[col].values

    while (len(queue) > 0) and (total_length < max_length):
        (node, this_length) = queue.popleft()
        total_length += 1
        this_length += 1

        # check for termination if we have multiple active branches
        # or we've reached minimum length
        if (len(queue) > 0) and (
            np.random.rand() < length_summary["is_leaf"][this_length]
        ):
            terminals[node] += 1
        elif (
            (len(queue) == 0)
            and (total_length > min_length)
            and (np.random.rand() < length_summary["is_leaf"][this_length])
        ):
            terminals[node] += 1
        else:
            next_node = np.random.choice(regions, p=p[node])
            counter[next_node] += 1
            queue.append((next_node, this_length))

        # Check for branching
        if np.random.rand() < length_summary["is_branch_point"][this_length]:
            branches[node] += 1
            total_length += 1 
            queue.append((node, this_length))
    return counter, branches, terminals


def generate_cell_v4(
    A, summary, length_summary, random_seed=None, start="P", max_length=1000000
):
    """
    Generate a cell with markov transition matrix, and wire length
    dependent branching and termination rates
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    # np.random.seed((os.getpid() * int(time.time())) % 123456789)

    # Keep track of counts, branches, deaths in each area
    counter = {}
    branches = {}
    terminals = {}
    for area in summary.index.values:
        counter[area] = 0
        branches[area] = 0
        terminals[area] = 0

    # Set up queue to process branches in breadth first order
    queue = deque()
    queue.append((start, 0))
    counter[start] += 1
    total_length = 0

    regions = A.index.values
    p = {}
    for col in A.columns:
        p[col] = A[col].values

    while (len(queue) > 0) and (total_length < max_length):
        (node, this_length) = queue.popleft()
        total_length += 1
        this_length += 1

        # Check for termination
        if np.random.rand() < length_summary["is_leaf"][this_length]:
            terminals[node] += 1
        else:
            next_node = np.random.choice(regions, p=p[node])
            counter[next_node] += 1
            queue.append((next_node, this_length))

        # Check for branching
        if np.random.rand() < length_summary["is_branch_point"][this_length]:
            branches[node] += 1
            queue.append((node, this_length))

        # Check for global size
        if (len(queue) == 0) and (
            total_length < 100000
            or np.random.rand()
            < (1 - weibull_min.cdf(total_length, 2.37, scale=437700, loc=0))
        ):
            s = np.sum(list(counter.values()))
            probs = [counter[x] / s for x in counter]
            next_node = np.random.choice(regions, p=probs)
            branches[next_node] += 1
            this_length = int(
                np.floor(np.sum(list(branches.values())) / total_length)
            )
            queue.append((next_node, this_length))

    return counter, branches, terminals


def get_length_summary(node_df, bins=50):
    """
    Get summary statistics of branching and termination rates as
    a function of wire length from soma
    """
    g = node_df.groupby(pd.qcut(node_df["wire_length"], bins), observed=False)
    df = g[["is_leaf", "is_branch_point"]].mean()
    df["is_leaf"] = df["is_leaf"] * 0.1
    df["is_branch_point"] = df["is_branch_point"] * 0.1

    # Clean up low probability points
    df.loc[34000:, "is_leaf"] = df.loc[34000]["is_leaf"]
    df.loc[34000:, "is_branch_point"] = df.loc[34000]["is_branch_point"]

    # Add a final terminating interval
    final = pd.Interval(df.index.values[-1].right, np.inf)
    final_vals = {"is_leaf": 1, "is_branch_point": 0}
    df.loc[final] = final_vals

    return df


def simulate_graph(
    Anorm,
    summary,
    roi_version=3,
    start="P",
    max_steps=1000000,
    samples=100,
    version=1,
    min_length=None,
    BFS_factor=None,
    node_df=None,
):
    """
    Simulate cells
    """
    if version == 1:
        temp = partial(
            generate_cell_v1,
            Anorm,
            summary,
        )
    elif version == 2:
        length_summary = get_length_summary(node_df)
        temp = partial(
            generate_cell_v2,
            Anorm,
            summary,
            length_summary,
        )
    elif version == 3:
        length_summary = get_length_summary(node_df)
        temp = partial(
            generate_cell_v3,
            Anorm,
            summary,
            length_summary,
            min_length=min_length,
        )
    elif version == 4:
        length_summary = get_length_summary(node_df)
        temp = partial(
            generate_cell_v4,
            Anorm,
            summary,
            length_summary,
        )
    with Pool() as pool:
        outputs = pool.starmap(
            temp, [(random_seed := x,) for x in range(samples)]
        )
    paths = [x[0] for x in outputs]
    x = pd.DataFrame(paths)
    sorted_columns = utils.get_roi_list(roi_version)
    sorted_columns = [y for y in sorted_columns if y in x.columns]
    x = x[sorted_columns].fillna(0)
    x["total"] = x.sum(axis=1)
    if version == 1:
        x["branches"] = [y[1] for y in outputs]
    else:
        x["branches"] = [sum(y[1].values()) for y in outputs]
    return x[sorted_columns], x[["total", "branches"]]


def plot_branches(branch_df, stats):
    """
    Plot branching statistics compared to data
    """
    x = branch_df.groupby("graph")["wire_length"].sum()
    y = branch_df.groupby("graph")["branch"].max()
    xy = pd.merge(x, y, on="graph")

    plt.figure()
    plt.plot(xy["wire_length"], (xy["branch"] - 1) / 2, "bo", label="data")
    plt.plot(stats["total"], stats["branches"], "ko", label="samples")
    plt.xlabel("Total axon length (um)")
    plt.ylabel("# of branches")
    plt.ylim(bottom=0)
    plt.xlim(left=0)
    plt.legend()


def plot_lengths(samples, bins=30, max_length=1000000):
    """
    Plot length distributions
    """
    samples = samples.copy()
    samples["total"] = samples.sum(axis=1)

    plt.figure()
    bins = np.arange(0, max_length + max_length / bins, max_length / bins)
    bins = plt.hist(
        samples["total"],
        bins=bins,
        label="markov samples",
        alpha=0.5,
    )
    plt.xlabel("Total axon length (um)")
    plt.ylabel("count")
    plt.xlim(0, max_length)


def plot_correlation(samples, match_data=False):
    """
    Plot correlation map of axon length in each brain region
    """

    samples = samples.copy()
    sums = samples.sum(axis=1)
    for col in samples.columns:
        samples[col] = samples[col] / sums
    samples = samples.fillna(0)

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
    if match_data:
        corr = samples[sorted_columns].corr(method="spearman")
    else:
        corr = samples.corr(method="spearman")
    plt.figure()
    sns.heatmap(
        corr, square=True, linewidth=0.5, vmin=-1, center=0, cmap="icefire"
    )


def plot_heatmap(dfLengths, cbar_label="", normalize=True, vmax=None):
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
    missing_columns = [x for x in sorted_columns if x not in dfLengths]
    for col in missing_columns:
        dfLengths[col] = 0
    dfPlot = dfLengths[sorted_columns].copy()

    sums = dfPlot.sum(axis=1)
    if normalize:
        for col in dfPlot.columns:
            dfPlot[col] = dfPlot[col] / sums
        dfPlot = dfPlot.fillna(0)

    # Determine sort order
    sortProjDict = {}
    for i, col in enumerate(sorted_columns):
        sortProjDict[col] = i

    # sort by projections
    dfPlot["top proj"] = dfPlot.columns[np.argmax(dfPlot, axis=1)]
    dfPlot["top proj"] = dfPlot["top proj"].map(sortProjDict)
    dfPlot["Total"] = sums / np.max(sums)
    dfPlot = dfPlot.sort_values("top proj")[sorted_columns + ["Total"]]

    # Make plot
    plt.figure(figsize=(18, 5))
    sns.heatmap(
        dfPlot.T,
        xticklabels=True,
        yticklabels=True,
        cbar_kws={"label": cbar_label},
        vmax=vmax,
    )
    plt.hlines([12], *plt.xlim(), colors="white", linewidth=2)
    plt.xlabel("Neuron")
    plt.xticks(plt.xticks()[0], labels=[], fontsize=7)


def fit_length_distribution(df, bins=25):
    """
    describe length distribution in data
    """
    df = df.copy()
    df["total"] = df.sum(axis=1)
    SC = utils.get_spinal_cord_labels()
    df["graph"] = df.index
    df["SC"] = [x in SC for x in df["graph"]]

    uncensored = df.query("not SC")["total"].values
    right_censored = df.query("SC")["total"].values
    data = CensoredData(uncensored=uncensored, right=right_censored)
    c, loc, scale = weibull_min.fit(data, floc=0)

    x = np.linspace(0, weibull_min.ppf(0.999, c, scale=scale), 100)

    plt.figure()

    plt.plot(
        x,
        weibull_min.pdf(x, c, loc=loc, scale=scale),
        "r-",
        lw=2,
        alpha=0.5,
        label="Weibull pdf (c={:.2f}, scale={:.0f})".format(c, scale),
    )
    bins = plt.hist(
        df.query("not SC")["total"],
        bins=bins,
        label="no spinal cord (uncensored)",
        alpha=0.5,
        density=True,
    )
    bins = plt.hist(
        df.query("SC")["total"],
        bins=bins[1],
        label="spinal cord (censored)",
        alpha=0.5,
        density=True,
    )
    plt.legend()
    plt.xlabel("Total length (um)")
    plt.ylabel("prob")
    return c, scale


def plot_leaf_length(node_df):
    """
    Plot termination points as function of wire length from soma
    """
    plt.figure()
    plt.hist(node_df.query("is_leaf")["wire_length"], bins=120)
    plt.xlabel("wire length")
    plt.ylabel("leaf nodes")


def plot_rates_by_length(node_df, summary, bins=50):
    """
    Plot branching and termination rates as function of wire length
    """
    plt.figure()
    g = node_df.groupby(pd.qcut(node_df["wire_length"], bins), observed=False)
    df = g[["is_leaf", "is_branch_point", "wire_length"]].mean()
    df["total_counts"] = g["is_leaf"].count()
    df.loc[34000:, "is_leaf"] = df.loc[34000]["is_leaf"]
    df.loc[34000:, "is_branch_point"] = df.loc[34000]["is_branch_point"]
    df["leaf_CI"] = (
        1.96
        / np.sqrt(df["total_counts"])
        * np.sqrt(df["is_leaf"] * (1 - df["is_leaf"]))
    )
    df["branch_CI"] = (
        1.96
        / np.sqrt(df["total_counts"])
        * np.sqrt(df["is_branch_point"] * (1 - df["is_branch_point"]))
    )
    df = df.set_index("wire_length")

    df["is_leaf"] = df["is_leaf"] * 0.1
    df["is_branch_point"] = df["is_branch_point"] * 0.1
    df["leaf_CI"] = df["leaf_CI"] * 0.1
    df["branch_CI"] = df["branch_CI"] * 0.1

    plt.plot(df["is_leaf"], "k-", label="termination rate")
    plt.plot(df["is_branch_point"], "b-", label="branch rate")
    plt.fill_between(
        df.index.values,
        (df["is_leaf"] - df["leaf_CI"]).values,
        (df["is_leaf"] + df["leaf_CI"]).values,
        color="black",
        alpha=0.1,
    )
    plt.fill_between(
        df.index.values,
        (df["is_branch_point"] - df["branch_CI"]).values,
        (df["is_branch_point"] + df["branch_CI"]).values,
        color="blue",
        alpha=0.1,
    )
    plt.axhline(
        summary["t/l"].mean(),
        color="k",
        linestyle="--",
        alpha=0.5,
        label="fixed termination rate",
    )
    plt.axhline(
        summary["b/l"].mean(),
        color="b",
        linestyle="--",
        alpha=0.5,
        label="fixed branch rate",
    )

    plt.legend()
    plt.xlabel("wire length (um)")
    plt.ylabel("rate (1/um)")
    plt.xlim(left=0)
    plt.ylim(0, 0.002)
    plt.tight_layout()


def plot_length_by_depth(branch_df, aggregate="mean"):
    """
    Plot average length between branch points as function of branch depth
    """
    plt.figure()
    if aggregate == "mean":
        x = branch_df.groupby("depth")["wire_length"].mean()
        y = branch_df.groupby("depth")["wire_length"].sem()
        plt.fill_between(x.index, x - y, x + y, color="gray")
    elif aggregate == "median":
        x = branch_df.groupby("depth")["wire_length"].median()
    plt.plot(x.index, x, "k-")
    plt.ylabel("{} distance between branch points".format(aggregate))
    plt.xlabel("branch depth")
    plt.xlim(-5, 125)


def plot_length_at_depth(branch_df, depth, bins=20):
    """
    plot distribution of wire lengths at a specific depth range
    """
    if isinstance(depth, int):
        depth = [depth]
    plt.figure()
    plt.hist(branch_df.query("depth in @depth")["wire_length"], bins=bins)
    plt.yscale("log")
    plt.ylabel("count (log scale)")
    plt.xlabel("distance to between branch points (um)")
    plt.xlim(0, 16000)
