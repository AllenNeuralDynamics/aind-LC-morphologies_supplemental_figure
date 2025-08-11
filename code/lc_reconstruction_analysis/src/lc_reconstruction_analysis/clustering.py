"""
Clustering cells
"""

from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans

import lc_reconstruction_analysis.utils as utils


def get_roi_map(DATA_DIR, roi_version=1):
    """
    Returns maps from ID to ROI, acronym, and parent
    """
    ccf_structures = pd.read_csv(
        Path(
            DATA_DIR,
            "allen_mouse_ccf",
            "annotation",
            "adult_mouse_ccf_structures.csv",
        )
    )
    id_to_acronym = defaultdict(lambda: "NaN")
    temp = ccf_structures.set_index("id")["acronym"].to_dict()
    for key in temp:
        id_to_acronym[key] = temp[key]
    acronym_to_id = {
        acronym: ccf_id for ccf_id, acronym in id_to_acronym.items()
    }
    id_to_parent = ccf_structures.set_index("id")[
        "parent_structure_id"
    ].to_dict()
    id_to_parent[None] = None  # to account for out of bands points
    # Create pathing dictionaries to group children CCF levels
    rois = utils.get_roi_list(roi_version)
    roiIDs = [acronym_to_id[roi] for roi in rois]
    # Get the pathing for each leaf node, find which roi each belongs to
    id_to_path = ccf_structures.set_index("id")["structure_id_path"].to_dict()
    # Create a new dictionary corresponding each CCF copartment to matching ROI
    # id_to_roi = {}
    id_to_roi = defaultdict(lambda: np.nan)
    # For each CCF compartment, break down path and find matching ROI
    for key, val in id_to_path.items():
        # Break down path
        pathList = [int(struct) for struct in val.split("/") if struct]
        # Find matching ROI (should be empty, or a single element)
        roiList = [id for id in roiIDs if id in pathList]
        if not roiList:
            id_to_roi[key] = np.nan
        else:
            id_to_roi[key] = roiList[0]
    return id_to_roi, id_to_acronym, id_to_parent


def build_length_df(
    dataDF, graphs, DATA_DIR, trim_df=False, normalize_df=True
):
    """
    Builds length dataframe
    """
    id_maps = get_roi_map(DATA_DIR)

    # Iterate over cells
    output = []
    for i, graph in graphs.items():
        output.append(
            df_length_inner(
                i,
                graph,
                id_maps,
            )
        )

    # Convert list of tuples to dictionary
    lengthDict = {x[0]: x[1] for x in output}

    # Convert to dataframe
    dfLengths = pd.DataFrame.from_dict(lengthDict, orient="index").fillna(0)
    dfLengths.rename(
        columns={"NaN": "Other"}, inplace=True
    )  # Rename "NaN" to "Other"

    if trim_df:
        total_counts = dfLengths.sum(axis=0, numeric_only=True)
        cumsum = total_counts / sum(total_counts)
        # threshold of a structure contributing at least 0.05% of total length
        keep_cols = pd.concat(
            [cumsum > 0.0005]
        )  # add back in non-numeric columns
        keep_cols["NaN"] = False  # enforce removal of "NaN"
        dfLengths = dfLengths.loc[:, keep_cols]

    if normalize_df:
        # Try normalizing to axonal length within a cell
        dfLengths = dfLengths.divide(dfLengths.sum(axis=1), axis=0)

    return dfLengths


def df_length_inner(input_i, graph, id_maps, split_borders=True):
    """
    Compute dictionary of lengths in each ROI
    """
    id_to_roi = id_maps[0]
    id_to_acronym = id_maps[1]

    # Assemble all nodes and edges
    nodes = graph.nodes(data=True)
    edges = graph.edges()

    regions = {node["allen_id"] for _, node in nodes}
    region_lengths = {}

    # Precompute which edges are in which region
    # Include edges where at least one node is in the region
    if split_borders:
        full_region_edge_mapper = {region: set() for region in regions}
        half_region_edge_mapper = {region: set() for region in regions}
        for u, v in edges:
            if nodes[u]["allen_id"] == nodes[v]["allen_id"]:
                full_region_edge_mapper[nodes[u]["allen_id"]].add((u, v))
            else:
                half_region_edge_mapper[nodes[u]["allen_id"]].add((u, v))
                half_region_edge_mapper[nodes[v]["allen_id"]].add((u, v))
    else:
        region_edge_mapper = {region: set() for region in regions}
        for u, v in edges:
            region_edge_mapper[nodes[u]["allen_id"]].add((u, v))
            region_edge_mapper[nodes[v]["allen_id"]].add((u, v))

    # For each region, sum edge weights
    for region in regions:

        if split_borders:
            # Filter edges in the given region
            full_edges_in_region = full_region_edge_mapper[region]
            half_edges_in_region = half_region_edge_mapper[region]

            # Calculate the total length of edges within the region
            full_length = sum(
                graph[u][v]["weight"] for u, v in full_edges_in_region
            )
            half_length = sum(
                graph[u][v]["weight"] for u, v in half_edges_in_region
            )
            total_length = full_length + 0.5 * half_length
        else:
            edges_in_region = region_edge_mapper[region]
            total_length = sum(
                graph[u][v]["weight"] for u, v in edges_in_region
            )

        # Store the result in the dictionary
        region_lengths[region] = total_length

    # Sum length within regions of interest
    roi_lengths = defaultdict(np.float64)
    for key, val in region_lengths.items():
        roi_lengths[id_to_roi[key]] += val

    lengthDict = {id_to_acronym[key]: val for key, val in roi_lengths.items()}
    return input_i, lengthDict


def cluster_kmeans(dataDF, dfLengths, norm=False, NORM_THRESHOLD=0.01):
    """
    Performs k=2 kmeans clustering
    """

    sorted_columns = [
        "OLF",
        "Isocortex",
        "HPF",
        "CTXsp",
        "CNU",
        "TH",
        "HY",
        "MB",
        "P",
        "MY",
        "CB",
        "Other",
    ]  # manually sort loosely anterior to posterior
    dfLengths = dfLengths[
        sorted_columns
    ]  # Reorder the DataFrame based on sorted column names
    a = dataDF.copy()
    b = dfLengths.copy()
    if norm:
        b = b > NORM_THRESHOLD
        b = b.astype(int)

    plotDF = a.merge(b, left_on="Graph", right_index=True)
    plotDF = plotDF.sort_values("somaDV")
    dfPlotClustered = plotDF.set_index("Graph")[sorted_columns]

    # Initialize the KMeans model
    n_clusters = 2
    kmeans = KMeans(n_clusters=n_clusters, n_init="auto", random_state=0)
    kmeans.fit(dfPlotClustered)

    # Get the cluster labels
    labels = kmeans.labels_

    # We'll createa copy of our dataframe and add the cluster labels
    plotDF["cluster"] = labels
    sortedClusters = (
        plotDF.groupby("cluster")["somaDV"].mean().sort_values().index
    )
    sortClusterDict = {cluster: i for i, cluster in enumerate(sortedClusters)}
    plotDF["cluster"] = plotDF["cluster"].map(sortClusterDict)

    return plotDF, sorted_columns


def plot_clustering(plotDF, sorted_columns):
    """
    Plot clustering in the space of the feature matrix
    """
    dfPlotClustered = plotDF.sort_values("somaDV").set_index("Graph")[
        sorted_columns
    ]
    plt.figure(figsize=(18, 5))
    sns.heatmap(dfPlotClustered, xticklabels=True, yticklabels=True)
    plt.yticks(fontsize=6)

    dfPlotClustered = plotDF.sort_values(
        "cluster"
    )  # sort by cluster label to organize plot

    # Plot the heatmap
    plt.figure(figsize=(18, 5))
    sns.heatmap(
        dfPlotClustered[sorted_columns], xticklabels=True, yticklabels=True
    )

    # Outline clusters by adding lines
    unique_clusters = dfPlotClustered["cluster"].unique()
    for cluster in unique_clusters:
        idx = np.where(dfPlotClustered["cluster"] == cluster)[0]
    plt.hlines(
        [min(idx), max(idx) + 1], *plt.xlim(), colors="white", linewidth=1
    )
