"""
General data access tools
"""

import json
import os
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd


def load_cells(DATA_DIR, reflect=True, remove_bad=True):
    """
    Load dataframe of cells with soma position, and graphs
    reflect_about_midline (bool) Reflects cells about midline
    """
    filePaths, genotypeDict = get_filepaths(DATA_DIR)
    graphs = load_graphs(filePaths)
    dataDF = load_data_to_dataframe(filePaths, genotypeDict, graphs)
    if reflect:
        dataDF, graph = reflect_about_midline(dataDF, graphs)

    if remove_bad:
        to_remove = set(graphs.keys()) - set(dataDF["Graph"].values)
        for name in to_remove:
            del graphs[name]
    return dataDF, graphs


def get_filepaths(DATA_DIR):
    """
    Loads a list of filepaths of cells, and a dictionary of genotypes
    """
    genotypeDict = {}
    folderPath = Path(DATA_DIR)
    filePaths = []
    for folder in os.listdir(folderPath):
        dataPath = folderPath / folder / "Complete_annotated"
        if os.path.exists(dataPath):
            filePaths.append(
                [
                    os.path.join(dataPath, fn)
                    for fn in os.listdir(dataPath)
                    if fn.endswith(".json")
                ]
            )
            subjectPath = dataPath.parent / "subject.json"
            with open(subjectPath) as f:
                subjectJSON = json.load(f)
            genotypeDict[subjectJSON["subject_id"]] = subjectJSON["genotype"]

    filePaths = [file for files in filePaths for file in files]
    return filePaths, genotypeDict


def load_data_to_dataframe(filePaths, genotypeDict, graphs):
    """
    Build a dataframe of cells
    filePaths - list of file locations
    genotypeDict - dictionary of genotype information
    graphs - networkx graphs
    """
    i = 0
    datasetDicts = {}
    for key, val in graphs.items():
        neuronID, sample, annotator = key.split("-")
        try:
            soma = [
                node
                for node in val.nodes()
                if val.nodes[node]["structure_id"] == 1
            ]  # Get soma nodes
            assert len(soma) == 1
            genotype = genotypeDict[sample]
            x, y, z = val.nodes[soma[0]]["pos"]
        except Exception:
            print(
                f"Error finding structures for: {key}, dropping from dataframe"
            )
            continue
        neuronDict = {
            "Graph": key,
            "ID": neuronID,
            "Sample": sample,
            "Annotator": annotator,
            "Genotype": genotype,
            "somaAP": x,
            "somaDV": y,
            "somaML": z,
        }
        datasetDicts[i] = neuronDict
        i = i + 1

    dataDF = pd.DataFrame.from_dict(datasetDicts, orient="index")

    # Sort by the name of the graph, because mutlipool loading is stochastic
    dataDF = dataDF.sort_values(by="Graph")
    return dataDF


def euclidean_distance(node1, node2):
    """
    Calculate the Euclidean distance between two nodes.

    Parameters:
    node1, node2 (dict): Nodes with 'pos' key containing x, y, z coordinates.

    Returns:
    float: Euclidean distance between node1 and node2.
    """
    pos1 = np.array(node1["pos"])
    pos2 = np.array(node2["pos"])
    return np.linalg.norm(pos1 - pos2)


def add_node_to_graph(graph, node):
    """
    Add a node with attributes to the graph.

    Parameters:
    graph (nx.DiGraph): The graph to which the node will be added.
    node (dict): Node data.
    """
    graph.add_node(
        node["sampleNumber"],
        pos=(node["x"], node["y"], node["z"]),
        radius=node["radius"],
        structure_id=node["structureIdentifier"],
        allen_id=node["allenId"],
    )


def add_edge_to_graph(graph, parent, child):
    """
    Add an edge between parent and child nodes in the graph,
    with weight as Euclidean distance.

    Parameters:
    graph (nx.DiGraph): The graph to which the edge will be added.
    parent, child (int): The sampleNumbers of the parent and child nodes.
    """
    graph.add_edge(
        parent,
        child,
        weight=euclidean_distance(graph.nodes()[parent], graph.nodes()[child]),
    )


def json_to_digraph(file_path):
    """
    Load a neuronal reconstruction from a JSON file into a NetworkX graph.

    The JSON file contains SWC data with additional brain region
    information for each node. The graph will be a directed tree.

    Parameters:
    file_path (str): Path to the JSON file containing reconstruction data.

    Returns:
    nx.DiGraph: A directed graph representing the neuronal tree.
    """
    try:
        with open(file_path, "r") as file:
            data = json.load(file)
    except IOError as e:
        print(f"Error opening file: {e}")
        return None

    # Certain JSON files may have a single 'neuron' object
    # instead of a 'neurons' array
    neuron_data = data["neuron"] if "neuron" in data else data["neurons"][0]

    axon_graph, dendrite_graph = nx.DiGraph(), nx.DiGraph()

    for structure, graph in [
        ("dendrite", dendrite_graph),
        ("axon", axon_graph),
    ]:
        if structure not in neuron_data:
            # Some reconstructions may be missing an axon or dendrite tracing
            print(f"Missing structure {structure} for {file_path}")
            continue
        for node in sorted(
            neuron_data[structure], key=lambda x: x["sampleNumber"]
        ):
            add_node_to_graph(graph, node)
            if node["parentNumber"] != -1:
                add_edge_to_graph(
                    graph, node["parentNumber"], node["sampleNumber"]
                )

    if dendrite_graph.nodes() and axon_graph.nodes():
        # Remove duplicate soma node from axon graph
        axon_graph.remove_node(1)

    # The sampleNumber starts at 1 for both axon and dendrite, so
    # relabel axon nodes to avoid key collisions when merging the graphs,.
    first_axon_label = (
        max(dendrite_graph.nodes()) + 1 if dendrite_graph.nodes() else 1
    )
    joined_graph = nx.union(
        dendrite_graph,
        nx.convert_node_labels_to_integers(
            axon_graph, first_label=first_axon_label
        ),
    )
    roots = [n for n in joined_graph if joined_graph.in_degree(n) == 0]
    # Link the dendrite to the axon
    if len(roots) == 2:
        add_edge_to_graph(joined_graph, roots[0], roots[1])

    return file_path, joined_graph


# Define a function for filtering the graph based on attribute values
def get_subgraph(G, attribute, values):
    """
    Extract a subgraph from the given graph based on
    specified attribute values.

    Parameters:
    G (nx.Graph): The original graph from which to extract the subgraph.
    attribute (str): The node attribute used for filtering.
    values (tuple): A tuple of attribute values to include in the subgraph.

    Returns:
    nx.Graph: A subgraph of G containing only nodes with the
    specified attribute values.
    """
    filtered_nodes = [
        node
        for node, attr in G.nodes(data=True)
        if attr.get(attribute) in values
    ]
    return G.subgraph(filtered_nodes)


def get_subgraph_targets(G, attribute, value):
    """
    Extract a subgraph from the given graph based on
    specified attribute values overlaps with the value

    Parameters:
    G (nx.Graph): The original graph from which to extract the subgraph.
    attribute (str): The node attribute used for filtering.
    values (tuple): A tuple of attribute values to include in the subgraph.

    Returns:
    nx.Graph: A subgraph of G containing only nodes with the
    specified attribute values.
    """
    filtered_nodes = [
        node
        for node, attr in G.nodes(data=True)
        if attr.get(attribute).intersection(value)
    ]
    return G.subgraph(filtered_nodes)


def load_graphs(filepaths):
    """
    Load all JSON files in the given directory as graphs using multiprocessing.

    Parameters:
    directory_path (str): Path to the directory containing JSON files.

    Returns:
    list of nx.Graph: A list of graphs loaded from the JSON files.
    """
    # Use multiprocessing pool to load graphs in parallel
    with Pool() as pool:
        graphs = pool.map(json_to_digraph, filepaths)

    # Organize into dictionary
    return {
        os.path.splitext(os.path.split(fn)[1])[0]: graph
        for fn, graph in graphs
    }


def get_cells_in_regions(manifest_path, acronyms):
    """
    Get cells located in a region
    """
    # Load the CSV file
    df = pd.read_csv(manifest_path)

    # If a single acronym is provided, convert it to a list
    if isinstance(acronyms, str):
        acronyms = [acronyms]

    # Filter the dataframe for the specified acronyms and get the filenames
    filtered_df = df[df["soma_acronym"].isin(acronyms)]
    filenames = filtered_df["filename"].tolist()
    filtered_acronyms = filtered_df["soma_acronym"].tolist()

    return filenames, filtered_acronyms


def reflect_about_midline(dataDF, graphs):
    """
    reflect cells about brain midline
    """
    mlMidline = 5700
    mlReflection = mlMidline * 2

    # Add column for original soma side
    dataDF["somaOnRight"] = dataDF["somaML"] > mlMidline

    # Reflect rightside graphs across midline
    for index, row in dataDF.iterrows():
        # Grab neurons with somas on right
        if row["somaOnRight"]:
            graph = graphs[row["Graph"]]
            # Update soma position in dataframe
            dataDF.loc[index, "somaML"] = (
                mlReflection - dataDF.loc[index, "somaML"]
            )
            # Reflect every node's position along ML axis
            for node in graph.nodes:
                graph.nodes[node]["pos"] = (
                    graph.nodes[node]["pos"][0],
                    graph.nodes[node]["pos"][1],
                    mlReflection - graph.nodes[node]["pos"][2],
                )
    return dataDF, graph


def load_obj(filename):
    """
    Load the vertices, vertex normals, and indices from a .obj file.

    Parameters:
    filename (str): Path to the .obj file

    Returns:
    tuple: A tuple containing three elements:
        - vertices (list of tuples):
            List of vertices, each vertex is a tuple (x, y, z)
        - normals (list of tuples):
            List of vertex normals, each normal is a tuple
            (nx, ny, nz)
        - indices (list of tuples):
            List of indices, each index is a tuple of vertex indices
            defining a face
    """
    vertices = []
    normals = []
    indices = []

    with open(filename, "r") as file:
        for line in file:
            if line.startswith("v "):  # Vertex definition
                parts = line.split()
                vertices.append(
                    (float(parts[1]), float(parts[2]), float(parts[3]))
                )
            elif line.startswith("vn "):  # Vertex normal definition
                parts = line.split()
                normals.append(
                    (float(parts[1]), float(parts[2]), float(parts[3]))
                )
            elif line.startswith("f "):  # Face definition
                parts = line.split()
                # Extracting only the vertex indices
                # (ignoring texture and normal indices)
                face_indices = [int(p.split("/")[0]) - 1 for p in parts[1:]]
                indices.append(tuple(face_indices))

    return vertices, normals, indices


def get_mesh_from_id(allen_id, DATA_DIR):
    """
    Load the CCF mesh for the id
    DATA_DIR - path to data
    """
    obj_path = os.path.join(DATA_DIR, "ccf_2017_obj", f"{allen_id}.obj")
    return load_obj(obj_path)


def compute_mesh_centroid(vertices, indices):
    """
    Compute the centroid of the mesh defined by vertices and indices
    """
    # Convert inputs to numpy arrays if they aren't already
    vertices = np.array(vertices, dtype=float)
    indices = np.array(indices, dtype=int)

    total_area = 0.0
    weighted_centroid = np.zeros(3)

    for tri in indices:
        v0, v1, v2 = vertices[tri[0]], vertices[tri[1]], vertices[tri[2]]

        tri_centroid = (v0 + v1 + v2) / 3.0

        edge1 = v1 - v0
        edge2 = v2 - v0
        tri_area = np.linalg.norm(np.cross(edge1, edge2)) / 2.0

        weighted_centroid += tri_centroid * tri_area
        total_area += tri_area

    if total_area == 0:
        # Fallback to average of vertices if total area is zero
        return np.mean(vertices, axis=0)
    else:
        return weighted_centroid / total_area


def get_roi_list(roi_version=2):
    """
    Returns a collection of ROIs
    """
    if roi_version == 1:
        rois = [
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
            "fiber tracts",
            "VS",
        ]
    elif roi_version == 2:
        rois = [
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
            "cbf",
            "lfbs",
            "mfbs",
            "VS",
        ]
    elif roi_version == 3:
        rois = [
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
            "cbf",
            "lfbs",
            "mfbs",
            "VS",
        ]
    return rois


def get_roi_centroids(DATA_DIR, roi_version=2):
    """
    get a dictionary of the AP/DV position
    of the centroids of the ROIs.
    roi_verion determines which set of ROIs to use
    """
    rois = get_roi_list(roi_version)

    # Get map of structure acronyms to ccf IDs
    acronym_to_id = get_acronym_to_id(DATA_DIR)

    # Iterate through ROIs and get centroid for each
    centroidDict = {}
    for name in rois:
        if name == "Other":
            centroidDict[name] = np.array([13000, -6018])
            continue
        x = get_mesh_from_id(acronym_to_id[name], DATA_DIR)
        centroidDict[name] = compute_mesh_centroid(x[0], x[2])[0:2]
        centroidDict[name][1] = -centroidDict[name][1]

    # manual adjustments, these are for plotting purposes
    centroidDict["HY"][0] = centroidDict["HY"][0] + 400
    centroidDict["TH"][1] = centroidDict["TH"][1] + 400
    centroidDict["TH"][0] = centroidDict["TH"][0] - 400
    centroidDict["HPF"][1] = centroidDict["HPF"][1] + 400
    centroidDict["MB"][1] = centroidDict["MB"][1] + 400
    centroidDict["VS"][0] = centroidDict["VS"][0] - 200
    centroidDict["P"][1] = centroidDict["P"][1] + 400
    if "fiber tracts" in centroidDict:
        centroidDict["fiber tracts"][1] = centroidDict["fiber tracts"][1] - 600
    else:
        centroidDict["cbf"][1] = centroidDict["cbf"][1] - 600
        centroidDict["cbf"][0] = centroidDict["cbf"][0] - 200
        centroidDict["lfbs"][1] = centroidDict["lfbs"][1] - 600
        centroidDict["mfbs"][1] = centroidDict["mfbs"][1] + 600

    return centroidDict


def get_acronym_to_id(DATA_DIR):
    """
    Returns a map of structure acronyms to allen ccf IDs
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
    return acronym_to_id


def get_spinal_cord_labels():
    """
    Get labels for spinal cord projection cells
    """
    ctxAndSc = [
        "N030-685222-NM",
        "N048-685221-VM",
        "N044-685221-DS",
        "N014-685221-YV",
        "N022-685221-YP",
        "N054-685221-HS",
        "N052-685221-BP",
        "N041-685221-JN",
        "N038-685221-HS",
        "N005-685222-YP",
        "N021-685222-NM",
        "N016-685221-PG",
        "N035-685221-DS",
        "N016-685222-YP",
        "N004-685222-BP",
        "N010-685222-VM",
    ]

    ibAndSc = [
        "N001-685221-PG",
        "N043-685221-HD",
        "N006-685222-DS",
        "N007-685222-BP",
        "N026-685222-SA",
        "N051-685221-YP",
        "N066-685221-JN",
        "N039-685222-AK",
        "N056-685221-AK",
        "N042-685221-HD",
    ]

    noCtxAndSc = [
        "N009-648434-KV",
        "N045-685221-VM",
        "N034-685222-HD",
        "N034-685221-VM",
        "N057-685221-AP",
        "N036-685221-YP",
        "N020-685221-BP",
        "N015-685222-VM",
        "N055-685221-JN",
        "N058-685221-JN",
        "N046-685221-SA",
        "N060-685221-YV",
        "N025-648434-PG",
        "N013-685221-DS",
        "N012-685222-BP",
        "N013-685222-DS",
        "N050-685221-HD",
        "N024-685221-VM",
        "N008-685222-HD",
        "N009-685222-SP",
        "N024-685222-AP",
        "N059-685221-SA",
        "N012-648434-JN",
        "N006-648434-JN",
        "N025-685222-SA",
        "N061-685221-YP",
        "N030-685221-VM",
        "N026-648434-PG",
    ]
    SC = list(set(noCtxAndSc + ibAndSc + ctxAndSc))
    return SC
