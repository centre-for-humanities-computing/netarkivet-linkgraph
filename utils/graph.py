"""
Module containing graph related classes and functions
"""
from __future__ import annotations

import os
import pickle
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import networkx as nx
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import scipy.sparse as spr
from sklearn.preprocessing import normalize

from utils.construction import (create_distance_matrix, create_mappings,
                                undirected_edges)
from utils.display import plotly_network


@dataclass
class Graph:
    """
    Class to ease graph manipulation and loading.

    Attributes
    ----------
    affinity: sparse or dense matrix of shape (n_nodes, n_nodes)
        Affinity matrix of the graph
    key_to_index: dict of str to int
        Mapping of node labels to node indices
    index_to_key: dict of int to str
        Mapping from node indices to node names
    directed: bool, default False
        Indicating whether the graph is directional or not
    """

    affinity: Union[spr.base.spmatrix, np.ndarray]
    key_to_index: Dict[str, int]
    index_to_key: Dict[int, str]

    #def __post_init__(self):
        #self.affinity = normalize(self.affinity, norm="max")
        
    def remove_loops(self) -> Graph:
        """
        Removes all loops from the graph, returns new graph.
        
        Notes
        -----
        This operation requires the affinity matrix to be turned
        into a dense matrix.
        If the affinity matrix would be too large in a dense format,
        this operation might fail.
        """
        affinity = self.affinity.todense()
        np.fill_diagonal(affinity, val=0)
        return type(self)(affinity, self.key_to_index, self.index_to_key)
        
    @classmethod
    def from_edges(
        cls,
        edges: pd.DataFrame,
        affinity_column: str = "connections"
    ) -> Graph:
        """
        Constructs a graph from a DataFrame of edges.

        Parameters
        ----------
        edges: DataFrame
            Data frame containing all edges.
        affinity_column: str, default "connections"
            Column in the edges data frame indicating affinity of elements
    
        Returns
        -------
        graph: Graph

        Notes
        -----
        The affinity column has to be called 'connections' in the supplied
        DataFrame.
        If the edges are directional they will be summed up and converted
        to non-directional.
        """
        key_to_index, index_to_key = create_mappings(edges)
        edges = undirected_edges(edges, key_to_index, index_to_key)
        affinity = create_distance_matrix(edges, key_to_index, directional=False, affinity_column=affinity_column)
        return cls(affinity, key_to_index, index_to_key)

    def invert(self) -> Graph:
        """
        Turns affinities to distances and vice-versa.

        Returns
        -------
        g: Graph
            New Graph object with affinities/distances inverted.

        Notes
        -----
        This operation requires the affinity matrix to be turned
        into a dense matrix.
        If the affinity matrix would be too large in a dense format,
        this operation might fail.
        """
        if spr.issparse(self.affinity):
            affinity = self.affinity.toarray()
        else:
            affinity = self.affinity
        return type(self)(1 - affinity, self.key_to_index, self.index_to_key)

    def __getitem__(self, index: Union[Tuple[str, str], Tuple[int, int]]):
        row, column = index
        if isinstance(row, str) and isinstance(column, str):
            row, column = self.key_to_index[row], self.key_to_index[column]
        return self.affinity[row, column]

    @property
    def _n_connections(self) -> np.ndarray:
        """
        Calculates the number of connections/sum of weights of each node.

        Returns
        ----------
        connections: ndarray of shape (n_nodes,)
        """
        return np.array(self.affinity.sum(axis=1)).flatten()

    @property
    def n_connections(self) -> Dict[str, int]:
        """
        Calculates the number of connections/sum of weights of each node.

        Returns
        ----------
        connections: dict of str to int
            A mapping of each node to its number of connections/sum of weights.
        """
        connections = self._n_connections()
        return {self.index_to_key[i]: n for i, n in enumerate(connections)}

    @property
    def node_names(self) -> List[str]:
        return pd.Series(self.index_to_key).sort_index().tolist()

    def display(
        self,
        edges: Optional[List[Tuple[str, str]]] = None,
        node_size: Union[np.ndarray, float] = 10.0,
        node_color: Union[np.ndarray, str] = "red",
        edge_weight: Union[np.ndarray, float] = 0.5,
        edge_color: Union[np.ndarray, str] = "#888",
    ) -> go.Figure:
        """
        Displays network with plotly.

        Parameters
        ----------
        edges: list of tuple of str, str or None, default None
            A list of tuples describing which nodes should be connected.
            Node names should be supplied.
            If not specified, the edges are inferred from the distance matrix.
        node_size: ndarray of shape (n_nodes,) or float, default 10
            Sizes of the nodes, if an array, different sizes will
            be used for each annotation.
        node_color: ndarray of shape (n_nodes,) or str, default "#ffb8b3"
            Specifies what color the nodes should be, if an array,
            different colors will be assigned to nodes based on a color scheme.
        edge_weight: ndarray of shape (n_edges,) or float, default 0.5
            Specifies the thickness of the edges connecting the nodes in the graph.
            If an array, different thicknesses will be used for each edge.
        edge_color: ndarray of shape (n-edges,) or str, default "#888"
            Specifies what color the edges should be, if an array,
            different colors will be assigned to edges based on a color scheme.

        Returns
        ----------
        figure: plotly figure
            Network graph drawn with plotly
        """
        if edges is not None:
            source, target = zip(*edges)
            source, target = pd.Series(source), pd.Series(target)
            source = source.map(self.key_to_index).to_numpy()
            target = target.map(self.key_to_index).to_numpy()
            edges = np.stack([source, target], axis=1)
        return plotly_network(
            self.affinity,
            edges=edges,
            node_labels=self.node_names,
            node_size=node_size,
            node_color=node_color,
            edge_weight=edge_weight,
            edge_color=edge_color,
        )
