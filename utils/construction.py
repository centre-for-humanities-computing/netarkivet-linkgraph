from typing import Dict, Tuple, Optional

import numpy as np
import pandas as pd
import scipy.sparse as spr


def create_mappings(edges: pd.DataFrame) -> Tuple[Dict[str, int], Dict[int, str]]:
    """
    Creates mapping from node index to key and vice versa, based on the edges of the graph.

    Parameters
    ----------
    edges: DataFrame
        Data frame containing edges of the graph

    Returns
    -------
    key_to_index: dict of str to int
        Mapping from node keys to indices
    index_to_key: dict of int to str
        Mapping from node indices to keys
    """
    domains = pd.concat((edges.source, edges.target), ignore_index=True).unique()
    key_to_index = {domain: index for index, domain in enumerate(domains)}
    index_to_key = {index: domain for index, domain in enumerate(domains)}
    return key_to_index, index_to_key


def create_distance_matrix(
    edges: pd.DataFrame, key_to_index: Dict[str, int], directional: bool = False,
    affinity_column: str = "connections"
) -> spr.csr_matrix:
    """
    Creates distance matrix of graph based on the edges.

    Parameters
    ----------
    edges: DataFrame
        Data frame containing edges of the graph
    key_to_index: dict of str to int
        Mapping from node keys to indices
    directional: bool, default False
        Flag indicating whether the distance matrix should be directional
    affinity_column: str, default "connections"
        Column in the edges data frame indicating affinity of elements
        
    Returns
    -------
    delta: sparse matrix of shape (n_nodes, n_nodes)
        Sparse distance matrix of the graph
    """
    # Mapping node names to node indices
    edges = edges.assign(
        target=edges.target.map(key_to_index), source=edges.source.map(key_to_index)
    )
    if directional:
        connections = edges[affinity_column]
        source = edges.source
        target = edges.target
    else:
        connections = edges[affinity_column].tolist() * 2
        source = pd.concat([edges.source, edges.target], ignore_index=True)
        target = pd.concat([edges.target, edges.source], ignore_index=True)
    # Creating a coo style sparse matrix for storing weights
    delta = spr.csr_matrix(
        (connections, (source, target)), shape=(len(key_to_index), len(key_to_index))
    )
    return delta


def undirected_edges(
    edges: pd.DataFrame, key_to_index: Optional[Dict[str, int]] = None, index_to_key: Optional[Dict[int, str]]=None
) -> pd.DataFrame:
    """
    Converts edges to undirected edges.

    Parameters
    ----------
    edges: DataFrame
        Data frame containing edges of the graph
    key_to_index: dict of str to int, or None, default None
        Mapping from node keys to indices
        If not supplied, mappings are created.
    index_to_key: dict of int to str, or None, default None
        Mapping from node indices to keys

    Returns
    -------
    edges: DataFrame
        Data frame containing undirected edges
    """
    if key_to_index is None:
        key_to_index, index_to_key = create_mappings(edges)
    # Convert columns to indices instead of node names
    edges = edges.assign(
        target=edges.target.map(key_to_index), source=edges.source.map(key_to_index)
    )
    # Undirect graph by assigning the lower index to be the source
    edges = edges.assign(
        source=np.minimum(edges.target, edges.source),
        target=np.maximum(edges.target, edges.source),
    )
    # Adding together the weights
    edges = edges.groupby(["source", "target"]).sum().reset_index()
    # Remapping nodes to names
    return edges.assign(
        target=edges.target.map(index_to_key), source=edges.source.map(index_to_key)
    )
