"""
Development scratch pad
"""

import pandas as pd

import lc_reconstruction_analysis.clustering as clustering
import lc_reconstruction_analysis.dendritic as dendritic

# DATA_DIR = "/Users/alex.piet/data/LC/"
# dataDF, graphs = utils.load_cells(DATA_DIR)


def run_sholl(dataDF, graphs, stepSize=10, nboots=1000):
    """
    Compute and plot sholl analysis for all cells (not clustered)
    """
    sholl_df = dendritic.build_sholl_df(dataDF, graphs, stepSize=stepSize)

    # Run bootstrapping over cells, not subjects
    # bootstraps, summary = dendritic.bootstrap_sholl(sholl_df, nboots=1000)
    # dendritic.plot_sholl(sholl_df,error='bootstrap',bootstraps=summary)

    # Run bootstrapping over subjects and cells hierarchically
    bootstraps, summary = dendritic.bootstrap_sholl_hierarchically(
        sholl_df, nboots=nboots
    )
    dendritic.plot_sholl(
        sholl_df, error="hierarchical bootstrap", bootstraps=summary
    )
    return sholl_df, bootstraps, summary


def run_sholl_with_clusters(
    dataDF,
    graphs,
    plotDF,
    group_col="cluster",
    stepSize=10,
    nboots=1000,
    error="hierarchical bootstrap",
):
    """
    Run Sholl analysis with clusters defined by "group_col"
    if error is "bootstrap" or "hierarchical bootstrap" compute the
    bootstraps, otherwise plot SEM
    """
    sholl_df = dendritic.build_sholl_df(dataDF, graphs, stepSize=stepSize)

    # Add cluster labels
    sholl_df = pd.merge(
        sholl_df,
        plotDF[["Graph", "cluster"]],
        left_on="name",
        right_on="Graph",
        how="left",
    )

    if error == "hierarchical bootstrap":
        # Do bootstraps for each group
        bootstraps, summary = dendritic.bootstrap_groups(
            sholl_df, group_col, nboots
        )
        dendritic.plot_sholl_groups(
            sholl_df,
            group_col,
            error="hierarchical bootstrap",
            bootstraps=summary,
        )
    elif error == "bootstrap":
        # Do bootstraps for each group
        bootstraps, summary = dendritic.bootstrap_groups(
            sholl_df, group_col, nboots
        )
        dendritic.plot_sholl_groups(
            sholl_df,
            group_col,
            error="bootstrap",
            bootstraps=summary,
        )
    else:
        dendritic.plot_sholl_groups(sholl_df, group_col, error=error)

    dendritic.plot_heatmap_with_column(plotDF, sholl_df, col="cluster")


def run_clustering(dataDF, graphs, DATA_DIR):
    """
    Run k=2 kmeans clustering on cells based on projection matrix
    in dfLengths
    """
    dfLengths = clustering.build_length_df(dataDF, graphs, DATA_DIR)
    plotDF, sorted_columns = clustering.cluster_kmeans(dataDF, dfLengths)
    clustering.plot_clustering(plotDF, sorted_columns)
    return dfLengths, plotDF, sorted_columns
