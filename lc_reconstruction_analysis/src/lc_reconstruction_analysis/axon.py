"""
    Adds wire length computation and other analyses of axons
"""

import queue

import numpy as np
import pandas as pd
from tqdm import tqdm

import lc_reconstruction_analysis.clustering as clustering
import lc_reconstruction_analysis.utils as utils


def add_all_wire_lengths(dataDF, graphs):
    """
    compute wire length for all cells
    """
    for name in tqdm(dataDF["Graph"]):
        graphs[name] = add_wire_length(graphs[name])
    return graphs


def annotate_all_branch_targets(dataDF, graphs, **kwargs):
    """
    Annotate nodes based on where they terminate
    """
    for name in tqdm(dataDF["Graph"]):
        graphs[name] = annotate_branch_targets(graphs[name], **kwargs)
    return graphs


def add_all_target_colors(dataDF, graphs, **kwargs):
    """
    Annotate nodes with a color based on where they terminate
    """
    target_colors = {}
    for name in tqdm(dataDF["Graph"]):
        graphs[name], target_colors = add_target_colors(
            graphs[name], target_colors=target_colors, **kwargs
        )
    return graphs, target_colors


def add_wire_length(graph):
    """
    Add wire length calculation to each node
    graph = axon.add_wire_length(graph)
    """
    graph.nodes[1]["wire_length"] = 0
    node_queue = queue.Queue()
    node_queue.put(1)
    while not node_queue.empty():
        node = node_queue.get()
        edges = dict(graph[node])
        for k in edges.keys():
            graph.nodes[k]["wire_length"] = (
                graph.nodes[node]["wire_length"] + edges[k]["weight"]
            )
            node_queue.put(k)
    return graph


def add_branch_points(graph):
    """
    Annotate nodes based on whether they are a branch point
    """
    for node in graph.nodes():
        if graph.out_degree(node) > 1:
            graph.nodes[node]["branch_point"] = True
        else:
            graph.nodes[node]["branch_point"] = False
    return graph


def annotate_current_structure(
    graph, annotate_on="structure", DATA_DIR=None, roi_version=2
):
    """
    Annotate nodes based on the current structure
    """
    id_to_roi, id_to_acronym, id_to_parent = clustering.get_roi_map(
        DATA_DIR, roi_version=roi_version
    )
    for node in graph.nodes():
        if annotate_on == "structure":
            graph.nodes[node]["structure"] = id_to_acronym[
                id_to_roi[graph.nodes[node]["allen_id"]]
            ]
        elif annotate_on == "allen_id":
            graph.nodes[node]["structure"] = id_to_acronym[
                graph.nodes[node]["allen_id"]
            ]
        else:
            raise AssertionError(
                'unknown type for "annotate_on": {}'.format(annotate_on)
            )
    return graph


def annotate_color_on_current_structure(
    graph, structure="structure", colors=None
):
    """
    annotate nodes with a color based on the current structure
    """
    targets = set()
    for node in graph.nodes:
        targets.add(graph.nodes[node][structure])
    targets = list(targets)

    # For each target-set, assign a random color
    if colors is None:
        colors = {}
        for target in targets:
            colors[target] = np.random.rand(3)
    else:
        for target in targets:
            if target not in colors:
                colors[target] = np.random.rand(3)

    # Assign colors to nodes based on target-sets
    for node in graph.nodes:
        graph.nodes[node]["color"] = colors[graph.nodes[node][structure]]
    return graph, colors


def annotate_branch_targets(graph, use_course_names=True, DATA_DIR=None):
    """
    Annotate axons/dendrites based on what structures they target

    use_course_names (bool), collapses allen_ids into course names
        from clustering.get_roi_map
    DATA_DIR (str), only used with use_course_names=True

    """
    target_type = "allen_id"
    if use_course_names:
        target_type = "structure"
        id_to_roi, id_to_acronym, id_to_parent = clustering.get_roi_map(
            DATA_DIR
        )
        for node in graph.nodes():
            graph.nodes[node]["structure"] = id_to_acronym[
                id_to_roi[graph.nodes[node]["allen_id"]]
            ]

    # Make a queue of nodes, and initialize with the terminal axon points
    node_queue = queue.Queue()
    for x in graph.nodes():
        if graph.out_degree(x) == 0:
            node_queue.put(x)
            if graph.nodes[x][target_type] is None:
                graph.nodes[x]["targets"] = set([-1])
            else:
                graph.nodes[x]["targets"] = set([graph.nodes[x][target_type]])
        else:
            graph.nodes[x]["targets"] = set()

    # For each node, propagate any terminal points
    # Note, we could ADD the current nodes location if we wanted to track
    # where structures propagate through
    while not node_queue.empty():
        node = node_queue.get()
        for in_node in graph.predecessors(node):
            node_queue.put(in_node)
            graph.nodes[in_node]["targets"].update(
                graph.nodes[node]["targets"]
            )
    return graph


def add_target_colors(  # noqa: C901
    graph,
    target_colors=None,
    condense_multi_targets=True,
    highlight_structure=None,
    color_type="ascending",
):
    """
    Assigns each node a color based on the targets innervated by
    the axon/dendrites of which this node is a part of
    condense_multi_targets (bool), show all nodes that innervate multiple
        targets as black

    returns the dictionary that maps target-sets to colors
    """

    # Build list of target sets
    targets = set()
    for node in graph.nodes:
        sort_targets = list(graph.nodes[node]["targets"])
        sort_targets.sort()
        targets.add(tuple(sort_targets))
    targets = list(targets)

    # For each target-set, assign a random color
    if target_colors is None:
        target_colors = {}
    if color_type == "random":
        for target in targets:
            if target not in target_colors:
                if condense_multi_targets and (len(target) > 1):
                    target_colors[target] = np.array([0, 0, 0])
                else:
                    target_colors[target] = np.random.rand(
                        3,
                    )
    elif color_type == "highlight_structure":
        for target in targets:
            if target not in target_colors:
                if highlight_structure in target:
                    target_colors[target] = np.array([1, 0, 0])
                else:
                    target_colors[target] = np.array([0, 0, 0])
    elif color_type == "ascending":
        descending = set(["P", "MY", "Other"])
        ascending = set(
            [
                "OLF",
                "Isocortex",
                "HPF",
                "fiber tracts",
                "CTXsp",
                "TH",
                "HY",
                "CNU",
                "VS",
            ]
        )
        midbrain = set(["CB", "MB"])
        for target in targets:
            in_ascending = len(set(target).intersection(ascending)) > 0
            in_descending = len(set(target).intersection(descending)) > 0
            in_midbrain = len(set(target).intersection(midbrain)) > 0
            if (
                np.sum(np.array([in_ascending, in_descending, in_midbrain]))
                > 1
            ):
                target_colors[target] = np.array([0, 0, 0])
            elif in_ascending:
                target_colors[target] = np.array([1, 0, 0])
            elif in_descending:
                target_colors[target] = np.array([0, 0, 1])
            elif in_midbrain:
                target_colors[target] = np.array([0, 1, 0])
            else:
                target_colors[target] = np.array([0.5, 0.5, 0.5])

    # Assign colors to nodes based on target-sets
    for node in graph.nodes:
        this_target = list(graph.nodes[node]["targets"])
        this_target.sort()
        graph.nodes[node]["color"] = target_colors[tuple(this_target)]
    return graph, target_colors


def build_combined_branch_table(dataDF, graphs):
    """
    Builds a branch table aggregating over all cells
    """
    branch_dfs = []
    node_dfs = []
    for name in tqdm(graphs):
        try:
            n, b = build_branch_table(graphs[name])
            b["graph"] = name
            n["graph"] = name
            branch_dfs.append(b)
            node_dfs.append(n)
        except Exception as e:
            print("problem {}, {}".format(name, e))
    return (
        pd.concat(branch_dfs).reset_index(),
        pd.concat(node_dfs).reset_index(),
    )


def build_branch_table(graph, ccf_labels="clean_structure"):
    """
    Builds a dataframe of nodes with columns:
        node #
        branch segment
        parent for that branch (node with branch point)
        last for that branch (node with branch point or terminal)
        parent CCF structure
        last CCF structure
        wire length
    ccf_labels: use either "structure" or "clean_structure"

    Builds a dataframe of branch segments with columns:
        parent
        min_length
        max_length
    """
    if "wire_length" not in graph.nodes[1]:
        raise Exception("Need to compute wire length first")
    if ccf_labels not in graph.nodes[1]:
        raise Exception("Need to compute {} first".format(ccf_labels))
    if ccf_labels not in ["structure", "clean_structure"]:
        raise Exception("Unknown CCF label type")

    # annotate first branch of the axon from the root
    branch_num = 1
    branch_depth = 1
    axon_tree = utils.get_subgraph(graph, "structure_id", [1, 2])
    results, successors = build_branch_table_inner(
        axon_tree, 1, branch_num, branch_depth, 1
    )

    # Iterate through successor branches until none remain
    branch_num = 2
    while len(successors) != 0:
        this_result, this_successors = build_branch_table_inner(
            axon_tree,
            successors[0][1],
            branch_num,
            successors[0][2],
            successors[0][0],
        )
        results = results + this_result
        successors = successors[1:] + this_successors
        branch_num += 1

    # Aggregate results into a dataframe of nodes
    node_df = pd.DataFrame(
        results,
        columns=[
            "node",
            "branch",
            "depth",
            "parent",
            "wire_length",
            "is_leaf",
            "is_branch_point",
            "structure",
        ],
    )

    # Group nodes into branches
    branch_df = pd.DataFrame()
    branch_df["depth"] = node_df.groupby("branch")["depth"].first()
    branch_df["parent"] = node_df.groupby("branch")["parent"].first()
    branch_df["last"] = node_df.groupby("branch")["node"].last()
    branch_df["is_leaf"] = node_df.groupby("branch")["is_leaf"].any()
    branch_df["min_length"] = [
        graph.nodes[x["parent"]]["wire_length"]
        for _, x in branch_df.iterrows()
    ]
    branch_df["max_length"] = node_df.groupby("branch")["wire_length"].max()
    branch_df["wire_length"] = (
        branch_df["max_length"] - branch_df["min_length"]
    )

    # Get CCF labels for each branch
    branch_df["start_structure"] = [
        graph.nodes[x][ccf_labels] for x in branch_df["parent"]
    ]
    branch_df["end_structure"] = [
        graph.nodes[x][ccf_labels] for x in branch_df["last"]
    ]

    return node_df, branch_df


def build_branch_table_inner(graph, node, branch_number, branch_depth, parent):
    """
    Recursive function to build table of branch segments
    node, the current node
    branch_number, the current branch segment
    branch_depth, the number of branches (including this one) to the root
    parent, the connection point for this branch segment

    returns
        results, a list of tuples
            Each tuple is (
                node,
                branch number,
                depth,
                parent,
                wire length,
                is_leaf,
                is_branch_point,
                structure
                )
        successors, a list of tuples
            Each tuple is (parent, node, depth)
    """
    if graph.out_degree(node) == 0:
        # At a leaf node
        return [
            (
                node,
                branch_number,
                branch_depth,
                parent,
                graph.nodes[node]["wire_length"],
                True,
                False,
                graph.nodes[node]["clean_structure"],
            )
        ], []
    elif graph.out_degree(node) > 1:
        # We are at a branch point
        successors = [
            (node, x, branch_depth + 1) for x in list(graph.successors(node))
        ]
        return [
            (
                node,
                branch_number,
                branch_depth,
                parent,
                graph.nodes[node]["wire_length"],
                False,
                True,
                graph.nodes[node]["clean_structure"],
            )
        ], successors
    else:
        # In the middle of a branch, continue until a branch point or leaf
        next_node = list(graph.successors(node))[0]
        results, successors = build_branch_table_inner(
            graph, next_node, branch_number, branch_depth, parent
        )
        return [
            (
                node,
                branch_number,
                branch_depth,
                parent,
                graph.nodes[node]["wire_length"],
                False,
                False,
                graph.nodes[node]["clean_structure"],
            )
        ] + results, successors


def add_clean_structure(
    graph,
    tolerance=1000,
    DATA_DIR=None,
    include_other=True,
    roi_version=2,
):
    """
    Adds a new attribute to each node "clean_structure"
    tolerance - the minimum wire length into each structure required
        for the label to update
    """
    graph = annotate_current_structure(
        graph,
        annotate_on="structure",
        DATA_DIR=DATA_DIR,
        roi_version=roi_version,
    )
    if include_other:
        for node in graph.nodes:
            if graph.nodes[node]["structure"] == "NaN":
                graph.nodes[node]["structure"] = "Other"

    current_structure = graph.nodes[1]["structure"]
    next_info = {
        "next_node": 1,
        "current_structure": current_structure,
        "switch_count": None,
        "switch_structure": current_structure,
        "switch_nodes": [],
    }
    while next_info is not None:
        graph, orphans, next_info = add_clean_structure_inner(
            graph,
            next_info["next_node"],
            next_info["current_structure"],
            next_info["switch_count"],
            next_info["switch_structure"],
            tolerance,
            next_info["switch_nodes"],
        )
    return graph


def add_clean_structure_inner(  # noqa: C901
    graph,
    node,
    current_structure,
    switch_count,
    switch_structure,
    tolerance,
    switch_nodes=[],
):
    """
    helper function for add_clean_structure()
    """
    graph.nodes[node]["clean_structure"] = current_structure
    if graph.out_degree(node) == 0:
        if switch_structure != current_structure:
            orphans = switch_nodes
        else:
            orphans = []
        return graph, orphans, None

    successors = list(graph.successors(node))
    if (
        (len(successors) == 1)
        and (current_structure == switch_structure)
        and (graph.nodes[successors[0]]["structure"] == graph.nodes[node])
    ):
        return (
            graph,
            [],
            {
                "next_node": successors[0],
                "current_structure": current_structure,
                "switch_structure": switch_structure,
                "switch_count": None,
                "switch_nodes": [],
            },
        )

    orphans = []
    switch_start = False
    for next_node in successors:
        if (current_structure == switch_structure) and (
            graph.nodes[next_node]["structure"] == graph.nodes[node]
        ):
            # We are not in a switch,
            # and the next node is still in this structure
            graph, this_orphans, _ = add_clean_structure_inner(
                graph,
                next_node,
                current_structure,
                None,
                current_structure,
                tolerance,
                [],
            )
        elif (current_structure == switch_structure) and (
            graph.nodes[next_node]["structure"] != graph.nodes[node]
        ):
            # Start of a switch evaluation
            switch_start = True
            this_switch_count = graph.edges[node, next_node]["weight"]
            if this_switch_count >= tolerance:
                # End of a switch
                graph, this_orphans, _ = add_clean_structure_inner(
                    graph,
                    next_node,
                    graph.nodes[next_node]["structure"],
                    None,
                    graph.nodes[next_node]["structure"],
                    tolerance,
                    [],
                )
            else:
                # Start of a switch
                graph, this_orphans, _ = add_clean_structure_inner(
                    graph,
                    next_node,
                    current_structure,
                    this_switch_count,
                    graph.nodes[next_node]["structure"],
                    tolerance,
                    [next_node],
                )
                orphans += this_orphans
        elif (current_structure != switch_structure) and (
            graph.nodes[next_node]["structure"] == switch_structure
        ):
            # Switch evaluation is still active
            this_switch_count = (
                switch_count + graph.edges[node, next_node]["weight"]
            )
            if this_switch_count >= tolerance:
                # End of a switch because we hit the tolerance
                for n in switch_nodes:
                    graph.nodes[n]["clean_structure"] = switch_structure
                graph, this_orphans, _ = add_clean_structure_inner(
                    graph,
                    next_node,
                    graph.nodes[next_node]["structure"],
                    None,
                    graph.nodes[next_node]["structure"],
                    tolerance,
                    [],
                )
                orphans += this_orphans
            else:
                # Still evaluating switch
                graph, this_orphans, _ = add_clean_structure_inner(
                    graph,
                    next_node,
                    current_structure,
                    this_switch_count,
                    switch_structure,
                    tolerance,
                    switch_nodes + [next_node],
                )
                orphans += this_orphans
        elif (current_structure != switch_structure) and (
            graph.nodes[next_node]["structure"] != switch_structure
        ):
            # Reset switch timer, because we have a new structure
            this_switch_count = graph.edges[node, next_node]["weight"]
            graph, this_orphans, _ = add_clean_structure_inner(
                graph,
                next_node,
                current_structure,
                this_switch_count,
                graph.nodes[next_node]["structure"],
                tolerance,
                switch_nodes + [next_node],
            )
            orphans += this_orphans

    # Check for orphaned nodes that can be assigned
    if (len(successors) > 1) and (
        (current_structure != switch_structure) or switch_start
    ):
        children = list(
            set([graph.nodes[s]["clean_structure"] for s in successors])
        )
        if len(children) > 1:
            for n in orphans:
                graph.nodes[n]["clean_structure"] = graph.nodes[node][
                    "clean_structure"
                ]
            orphans = []
    return graph, orphans, None
