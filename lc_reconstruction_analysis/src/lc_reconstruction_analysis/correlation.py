"""
   Tools for shuffling the correlation matrix
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import chi2

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
group1 = ["OLF", "Isocortex", "HPF", "CTXsp", "CNU", "TH", "HY"]
group2 = ["P", "MY", "Other"]
group3 = ["MB", "CB"]


def plot_correlation(df):
    """
    Plot the correlation of projection percentages to each brain region
    """
    df = df[sorted_columns]
    corr = df.corr(method="spearman")
    plt.figure()
    sns.heatmap(
        corr, square=True, linewidth=0.5, vmin=-1, center=0, cmap="icefire"
    )
    plt.title("Original data")


# Shuffle Rows
def plot_shuffle_row(df):
    """
    Shuffle each cell's projections, then plot the correlation matrix
    """
    corr_shuffle = shuffle_row(df)
    plt.figure()
    sns.heatmap(
        corr_shuffle,
        square=True,
        linewidth=0.5,
        vmin=-1,
        center=0,
        cmap="icefire",
    )
    plt.title("Shuffle each cell's projections")


def shuffle_row(df):
    """
    Shuffle each cell's projections
    """
    x_shuffle = np.apply_along_axis(np.random.permutation, 1, df.to_numpy())
    df_shuffle = pd.DataFrame(x_shuffle, columns=sorted_columns)
    corr_shuffle = df_shuffle.corr(method="spearman")
    return corr_shuffle


def test_shuffle_row(df, n=10000, alpha=0.05):
    """
    Compare real data correlation matrix with shuffled row correlation matrix
    perform n bootstraps
    use alpha as significance level
    """
    corr = df.corr(method="spearman")
    n1 = len(df)
    statistics = [
        jennrich_test(corr.to_numpy(), shuffle_row(df).to_numpy(), n1, n1)[1]
        for x in range(n)
    ]
    p = np.shape(corr)[0]
    dof = p * (p - 1) / 2
    chisquared_val = chi2.isf(alpha, dof)
    p = 1 - np.sum(np.array(statistics) > chisquared_val) / len(statistics)
    print(
        "Probability of observing data in shuffle distribution: {}".format(p)
    )


# Shuffle columns
def plot_shuffle_columns(df_unnormalized):
    """
    Note!!! This method uses the unnormalized df_lengths data
    df_unnormalized = clustering.build_length_df(
        dataDF, graphs, DATA_DIR, normalize_df=False
        )
    """
    corr = shuffle_columns(df_unnormalized)
    plt.figure()
    sns.heatmap(
        corr, square=True, linewidth=0.5, vmin=-1, center=0, cmap="icefire"
    )
    plt.title("shuffle columns")


def shuffle_columns(df_unnormalized):
    """
    Shuffle the columns
    """
    temp = df_unnormalized[sorted_columns].copy()
    for col in temp.columns.values:
        temp[col] = temp[col].sample(frac=1).values
    temp = temp.divide(temp.sum(axis=1), axis=0)
    corr = temp.corr(method="spearman")
    return corr


def test_shuffle_columns(df, df_unnormalized, n=10000, alpha=0.05):
    """
    Compare real data correlation matrix with shuffled row correlation matrix
    perform n bootstraps
    use alpha as significance level
    """
    corr = df.corr(method="spearman")
    n1 = len(df)
    statistics = [
        jennrich_test(
            corr.to_numpy(),
            shuffle_columns(df_unnormalized).to_numpy(),
            n1,
            n1,
        )[1]
        for x in range(n)
    ]
    p = np.shape(corr)[0]
    dof = p * (p - 1) / 2
    chisquared_val = chi2.isf(alpha, dof)
    p = 1 - np.sum(np.array(statistics) > chisquared_val) / len(statistics)
    print(
        "Probability of observing data in shuffle distribution: {}".format(p)
    )


# Random Projections
def plot_random_projections(df):
    """
    Plot correlation matrix from random R^12 vectors
    """
    # Plot correlation in randomly generated projection patterns
    corr_random = random_projections(df)
    plt.figure()
    sns.heatmap(
        corr_random,
        square=True,
        linewidth=0.5,
        vmin=-1,
        center=0,
        cmap="icefire",
    )
    plt.title("Random projections")


def random_projections(df):
    """
    Generate random r^12 vectors
    """
    df_random = df.copy()
    for i in range(len(df_random)):
        x = np.random.rand(len(sorted_columns))
        df_random.iloc[i, :] = x / np.sum(x)
    corr_random = df_random.corr(method="spearman")
    return corr_random


def test_random_projections(df, n=10000, alpha=0.05):
    """
    Compare real data correlation matrix with shuffled row correlation matrix
    perform n bootstraps
    use alpha as significance level
    """
    corr = df.corr(method="spearman")
    n1 = len(df)
    statistics = [
        jennrich_test(
            corr.to_numpy(), random_projections(df).to_numpy(), n1, n1
        )[1]
        for x in range(n)
    ]
    p = np.shape(corr)[0]
    dof = p * (p - 1) / 2
    chisquared_val = chi2.isf(alpha, dof)
    p = 1 - np.sum(np.array(statistics) > chisquared_val) / len(statistics)
    print(
        "Probability of observing data in shuffle distribution: {}".format(p)
    )


# Shuffle a subset (group) of brain regions
def plot_shuffle_row_group(df, test_group):
    """
    Shuffle projections on a subset (group) of brain regions)
    """
    df = df.copy()
    x_shuffle = np.apply_along_axis(
        np.random.permutation, 1, df[test_group].to_numpy()
    )
    df_group = pd.DataFrame(x_shuffle, columns=test_group)
    for g in test_group:
        df[g] = df_group[g].values
    df = df[sorted_columns]
    corr = df.corr(method="spearman")
    plt.figure()
    sns.heatmap(
        corr, square=True, linewidth=0.5, vmin=-1, center=0, cmap="icefire"
    )
    plt.title("Shuffled subset: {}".format(test_group))


def shuffle_row_group(df, test_group, return_full=False):
    """
    Shuffle each cell's projections within the subset of regions
    defined in test_group. Then compute the norm of the correlation matrix
    """
    x_shuffle = np.apply_along_axis(
        np.random.permutation, 1, df[test_group].to_numpy()
    )
    df_group = pd.DataFrame(x_shuffle, columns=test_group)
    if return_full:
        df_full = df.copy()
        for g in test_group:
            df_full[g] = df_group[g].values
        df_group = df_full
    corr_group = df_group.corr(method="spearman")
    return corr_group


def test_shuffle_row_group(
    df, test_group, n=10000, alpha=0.05, use_full=False
):
    """
    Compute statistics for shuffling a subset of regions
    """
    if use_full:
        corr = df.corr(method="spearman")
    else:
        corr = df[test_group].corr(method="spearman")
    n1 = len(df)
    statistics = [
        jennrich_test(
            corr.to_numpy(),
            shuffle_row_group(df, test_group, use_full).to_numpy(),
            n1,
            n1,
        )[1]
        for x in range(n)
    ]
    p = np.shape(corr)[0]
    if use_full:
        dof = len(test_group) * (p - 1) / 2
        # Some entries of the correlation matrix cannot change
    else:
        dof = p * (p - 1) / 2
    chisquared_val = chi2.isf(alpha, dof)
    p = 1 - np.sum(np.array(statistics) > chisquared_val) / len(statistics)
    print(
        "Probability of observing data in shuffle distribution: {}".format(p)
    )


# Shuffle columns
def plot_shuffle_columns_group(df_unnormalized, test_group):
    """
    Note!!! This method uses the unnormalized df_lengths data
    df_unnormalized = clustering.build_length_df(
        dataDF, graphs, DATA_DIR, normalize_df=False
        )
    """
    corr = shuffle_columns_group(df_unnormalized, test_group)
    plt.figure()
    sns.heatmap(
        corr, square=True, linewidth=0.5, vmin=-1, center=0, cmap="icefire"
    )
    plt.title("shuffle columns")


def shuffle_columns_group(df_unnormalized, test_group):
    """
    Shuffle columns just for a subset of regions
    """
    temp = df_unnormalized[sorted_columns].copy()
    for col in test_group:
        temp[col] = temp[col].sample(frac=1).values
    temp = temp.divide(temp.sum(axis=1), axis=0)
    corr = temp.corr(method="spearman")
    return corr


def test_shuffle_columns_group(
    df, df_unnormalized, test_group, n=10000, alpha=0.05
):
    """
    Compare real data correlation matrix with shuffled row correlation matrix
    perform n bootstraps
    use alpha as significance level
    """
    corr = df.corr(method="spearman")
    n1 = len(df)
    statistics = [
        jennrich_test(
            corr.to_numpy(),
            shuffle_columns_group(df_unnormalized, test_group).to_numpy(),
            n1,
            n1,
        )[1]
        for x in range(n)
    ]
    p = np.shape(corr)[0]
    dof = len(test_group) * (p - 1) / 2
    chisquared_val = chi2.isf(alpha, dof)
    p = 1 - np.sum(np.array(statistics) > chisquared_val) / len(statistics)
    print(
        "Probability of observing data in shuffle distribution: {}".format(p)
    )


# Statistical method functions
def test_implementation(df):
    """
    p value should be >> 0.05, since the shuffle data
    are all similar
    """
    p = []
    s = []
    for i in range(10000):
        pval, statistic = test_implementation_inner(df)
        p.append(pval)
        s.append(statistic)
    # alpha = 0.05
    # Degrees of Freedom based on 12 brain regions
    chisquared_val = chi2.isf(0.05, 12 * (12 - 1) / 2)
    print(1 - np.sum(np.array(s) > chisquared_val) / len(s))


def test_implementation2(df):
    """
    p value should be << 0.05, since the data is very different
    from the shuffle
    """
    p = []
    s = []
    for i in range(10000):
        pval, statistic = test_implementation2_inner(df)
        p.append(pval)
        s.append(statistic)
    # alpha = 0.05
    # Degrees of Freedom based on 12 brain regions
    chisquared_val = chi2.isf(0.05, 12 * (12 - 1) / 2)
    print(1 - np.sum(np.array(s) > chisquared_val) / len(s))


def test_implementation_inner(df):
    """
    Test statistical method by comparing two shuffled matrices
    """
    # Shuffle once
    x_shuffle = np.apply_along_axis(np.random.permutation, 1, df.to_numpy())
    df_shuffle = pd.DataFrame(x_shuffle, columns=sorted_columns)
    corr = df_shuffle.corr(method="spearman")
    R1 = corr.to_numpy()

    # Shuffle twice
    x_shuffle2 = np.apply_along_axis(np.random.permutation, 1, df.to_numpy())
    df_shuffle2 = pd.DataFrame(x_shuffle2, columns=sorted_columns)
    corr2 = df_shuffle2.corr(method="spearman")
    R2 = corr2.to_numpy()

    # Compute statistic
    n1 = len(df_shuffle)
    n2 = len(df_shuffle2)
    pval, statistic = jennrich_test(R1, R2, n1, n2)
    return pval, statistic


def test_implementation2_inner(df):
    """
    Testing the implementation by checking unshuffled data
    against shuffled data
    """
    # Compute correlation method for data
    corr = df.corr(method="spearman")
    R1 = corr.to_numpy()

    # Shuffle once
    x_shuffle2 = np.apply_along_axis(np.random.permutation, 1, df.to_numpy())
    df_shuffle2 = pd.DataFrame(x_shuffle2, columns=sorted_columns)
    corr2 = df_shuffle2.corr(method="spearman")
    R2 = corr2.to_numpy()

    # Compute statistic
    n1 = len(df)
    n2 = len(df_shuffle2)
    pval, statistic = jennrich_test(R1, R2, n1, n2)
    return pval, statistic


def jennrich_test(R1, R2, n1, n2):
    """
    Compute Jennrich's test for the equality of two
    correlation matrices.
    """
    p = R1.shape[0]

    # 1. Calculate the pooled correlation matrix (R_bar)
    R_bar = (n1 * R1 + n2 * R2) / (n1 + n2)
    R_inv = np.linalg.inv(R_bar)

    # 2. Calculate Z
    Z = R_inv @ (R1 - R2) * np.sqrt((n1 * n2) / (n1 + n2))

    # 3. Calculate W
    S = np.eye(p) + R_bar * R_inv
    S_inv = np.linalg.inv(S)

    # Calculate the test statistic (J)
    Zdiag = np.diag(Z)
    j_statistic = 0.5 * np.trace(Z @ Z) - (Zdiag.T @ S_inv @ Zdiag)

    dof = p * (p - 1) / 2
    pval = 1 - chi2.cdf(j_statistic, dof)
    return pval, j_statistic
