"""Utility functions for building network graphs in Plotly"""
from typing import Any, Dict, List, Literal, Optional, Tuple, TypeVar, Union

import networkx as nx
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from networkx.drawing.layout import spring_layout


def _edge_pos(edges: np.ndarray, x_y: np.ndarray) -> np.ndarray:
    """
    Through a series of nasty numpy tricks, that I® wrote
    this function transforms edges and either the x or the y positions of nodes to
    the x or y positions for the lines in the plotly figure.
    In order for the line not to be connected, the algorithm
    has to insert a nan value after each pair of points that have to be connected.
    """
    edges = np.array(edges)
    x_y = np.array(x_y)
    a = x_y[edges]
    b = np.zeros((a.shape[0], a.shape[1] + 1))
    b[:, :-1] = a
    b[:, -1] = np.nan
    return b  # .flatten()


def get_node_positions(nx_graph) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculates node positions based on a networkx graph.

    Parameters
    ----------
    nx_graph: NetworkX graph
        Graph to position

    Returns
    ----------
    x: ndarray of shape (n_nodes,)
        x coordinates of nodes
    y: ndarray of shape (n_nodes,)
        y coordinates of nodes
    """
    pos = spring_layout(nx_graph)
    x, y = zip(*pos.values())
    x, y = np.array(x), np.array(y)
    return x, y


def create_annotations(
    labels: List[str],
    x: np.ndarray,
    y: np.ndarray,
    size: Union[np.ndarray, float] = 5,
) -> List[Dict[str, Any]]:
    """
    Creates annotation objects for a plotly graph object.

    Parameters
    ----------
    labels: list of str
        List of annotation strings
    x: ndarray of shape (n_nodes,)
        x coordinates of annotations
    y: ndarray of shape (n_nodes,)
        y coordinates of annotations
    size: ndarray of shape (n_nodes,) or float, default 5
        Sizes of the annotations, if an array, different sizes will
        be used for each annotation.

    Returns
    ----------
    annotations: list of dict
        Plotly annotation objects
    """
    annotations = []
    if size is float:
        size = np.full(len(x), size)
    for i, label in enumerate(labels):
        annotations.append(
            dict(
                text=label,
                x=x[i],
                y=y[i],
                showarrow=False,
                xanchor="center",
                bgcolor="rgba(255,255,255,0.5)",
                bordercolor="rgba(0,0,0,0.5)",
                font={
                    "family": "Helvetica",
                    "size": size[i],  # type: ignore
                    "color": "black",
                },
            )
        )
    return annotations


MAX_SIZE = 20


def create_node_trace(
    x: np.ndarray,
    y: np.ndarray,
    labels: Optional[List[str]] = None,
    color: Union[np.ndarray, str] = "#ffb8b3",
    size: Union[np.ndarray, float] = 10,
    display_mode: str = "markers",
    textposition: Optional[str] = None,
    colorbar_title: str = "",
) -> go.Figure:
    """
    Draws a trace of nodes for a plotly network.

    Parameters
    ----------
    x: ndarray of shape (n_nodes,)
        x coordinates of nodes
    y: ndarray of shape (n_nodes,)
        y coordinates of nodes
    labels: list of str or None, default None
        Labels to assign to each node, if not specified,
        node indices will be displayed.
    size: ndarray of shape (n_nodes,) or float, default 10
        Sizes of the nodes, if an array, different sizes will
        be used for each annotation.
    color: ndarray of shape (n_nodes,) or str, default "#ffb8b3"
        Specifies what color the nodes should be, if an array,
        different colors will be assigned to nodes based on a color scheme.
    display_mode: str, default "markers"
        Specifies how the nodes should be displayed,
        consult Plotly documentation for further details
    textposition: str or None, default None
        Specifies how text should be positioned on the graph,
        consult Plotly documentation for further details

    Returns
    ----------
    node_trace: go.Figure
        Nodes for the graph drawn by plotly.
    """
    if not isinstance(size, float) and not isinstance(size, int):
        # Normalize size
        size_norm = np.linalg.norm(size, np.inf)
        size = (size / size_norm) * MAX_SIZE + 5
    else:
        size = np.full(x.shape, size)
    indices = np.arange(len(x))
    node_trace = go.Scatter(
        x=x,
        y=y,
        mode=display_mode,
        hoverinfo="text",
        text=labels or indices,
        textposition=textposition,
        marker={
            "color": color,
            "size": size,
            "line_width": 2,
            "colorbar": dict(title=colorbar_title),
            "colorscale": "Viridis_r",
        },
        customdata=indices,
    )
    return node_trace  # type: ignore


MAX_WEIGHT = 5


def create_edge_trace(
    x: np.ndarray,
    y: np.ndarray,
    edges: np.ndarray,
    width: Union[np.ndarray, float] = 0.5,
    color: Union[np.ndarray, str] = "#888",
) -> go.Figure:
    """
    Draws a trace of edges for a plotly network.

    Parameters
    ----------
    x: ndarray of shape (n_nodes,)
        x coordinates of nodes
    y: ndarray of shape (n_nodes,)
        y coordinates of nodes
    edges: ndarray of shape (n_edges, 2) or None
        A matrix describing which nodes in the graph should be connected.
        Each row describes one connection with the indices of the two nodes.
        If not specified, a fully connected graph will be used.
    width: ndarray of shape (n_edges,) or float, default 0.5
        Specifies the thickness of the edges connecting the nodes in the graph.
        If an array, different thicknesses will be used for each edge.
    color: ndarray of shape (n-edges,) or str, default "#888"
        Specifies what color the edges should be, if an array,
        different colors will be assigned to edges based on a color scheme.

    Returns
    ----------
    edge_trace: list of graph objects
        Edges for the graph drawn by plotly.
    """
    x_edges = _edge_pos(edges, x)
    y_edges = _edge_pos(edges, y)
    n_edges = edges.shape[0]
    indices = np.arange(n_edges)
    if isinstance(width, float) or isinstance(width, int):
        width = np.full(n_edges, width)
    else:
        size_norm = np.linalg.norm(width, 4)
        width = (width / size_norm) * MAX_WEIGHT + 1
    if isinstance(color, str):
        color = np.full(n_edges, color)
    edge_trace = [
        go.Scatter(
            x=x_edges[i],
            y=y_edges[i],
            line={"width": width[i], "color": color[i]},
            hoverinfo="none",
            mode="lines",
            showlegend=False,
        )
        for i in indices
    ]
    return edge_trace  # type: ignore


def _network(
    node_x: np.ndarray,
    node_y: np.ndarray,
    edges: np.ndarray,
    node_labels: Optional[List[str]] = None,
    node_size: Union[np.ndarray, float] = 10.0,
    node_color: Union[np.ndarray, str] = "red",
    edge_weight: Union[np.ndarray, float] = 0.5,
    edge_color: Union[np.ndarray, str] = "#888",
    colorbar_title: str = "",
):
    x, y = node_x, node_y
    # Creating node trace for the network
    node_trace = create_node_trace(
        x,
        y,
        labels=node_labels,
        color=node_color,
        size=node_size,
        colorbar_title=colorbar_title,
    )
    # Creating edge trace lines
    edge_trace = create_edge_trace(x, y, edges, width=edge_weight, color=edge_color)
    # Making figure
    fig = go.Figure(
        data=[*edge_trace, node_trace],
        layout=go.Layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            titlefont_size=16,
            height=1000,
            showlegend=False,
            margin=dict(b=20, l=5, r=5, t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        ),
    )
    return fig


def plotly_network(
    affinity_matrix: np.ndarray,
    edges: Optional[np.ndarray] = None,
    node_labels: Optional[List[str]] = None,
    node_size: Union[np.ndarray, float] = 10.0,
    node_color: Union[np.ndarray, str] = "red",
    edge_weight: Union[np.ndarray, float] = 0.5,
    edge_color: Union[np.ndarray, str] = "#888",
) -> go.Figure:
    """
    Draws plotly network graph based on a networkx graph.

    Parameters
    ----------
    affinity_matrix: ndarray of shape (n_nodes, n_nodes)
        Matrix containing the affinities between two nodes.
        Note: though ndarray is in the type hint, sparse matrices may also be used.
    edges: ndarray of shape (n_edges, 2) or None, default None
        A matrix describing which nodes in the graph should be connected.
        Each row describes one connection with the indices of the two nodes.
        If not specified, a fully connected graph will be used.
    node_labels: list of str or None, default None
        Labels to assign to each node, if not specified,
        node indices will be displayed.
    node_size: ndarray of shape (n_nodes,) or float, default 10
        Sizes of the nodes, if an array, different sizes will
        be used for each annotation.
    node_color: ndarray of shape (n_nodes,) or str, default "red"
        Specifies what color the nodes should be, if an array,
        different colors will be assigned to nodes based on a color scheme.
    edge_weight: ndarray of shape (n_edges,) or float, default 0.5
        Specifies the thickness of the edges connecting the nodes in the graph.
        If an array, different thicknesses will be used for each edge.
    edge_color: ndarray of shape (n-edges,) or str, default "#888"
        Specifies what color the edges should be, if an array,
        different colors will be assigned to edges based on a color scheme.

    Returns
    -------
    figure: plotly figure
        Network graph drawn with plotly

    Note
    ----
    I will probably deprecate this function as soon as possible
    in favor of px_network.
    This is literally the worst of all worlds.
    """
    # Creating NetworkX graph and obtaining node positions
    nx_graph = nx.Graph(affinity_matrix)
    node_x, node_y = get_node_positions(nx_graph)
    if edges is None:
        # If no edges are supplied we take the edges of the
        # NetworkX graph, which are based on the affinity matrix
        edges = np.array(nx_graph.edges())
    # Deleting references to variables, that way it won't be in locals()
    del nx_graph
    del affinity_matrix
    # God forgive me for writing these few lines,
    # I swear they also do this in Plotly's code 😩
    return _network(
        node_x,
        node_y,
        edges,
        node_labels,
        node_size,
        node_color,
        edge_weight,
        edge_color,
        colorbar_title="Total antal forbindelser",
    )


# Again a few lines so that the type annotations don't become incredibly long
T = TypeVar("T")
Real = Union[float, int]
Weight = Dict[Literal["weight"], Real]
Edge = Tuple[T, T]
WeightedEdge = Tuple[T, T, Weight]
Edges = List[Edge]
WeightedEdges = List[WeightedEdge]


def _networkx_edges(
    edges_df: pd.DataFrame,
    source_column: str,
    target_column: str,
    weight_column: Optional[str] = None,
) -> Union[Edges, WeightedEdges]:
    source = edges_df[source_column]
    target = edges_df[target_column]
    if weight_column is None:
        edges = zip(target, source)
    else:
        weights: List[Weight] = [
            {"weight": weight} for weight in edges_df[weight_column]
        ]
        edges = zip(target, source, weights)
    return list(edges)  # type: ignore


def px_network(
    edges: pd.DataFrame,
    nodes: Optional[pd.DataFrame] = None,
    *,
    source: Optional[str] = None,
    target: Optional[str] = None,
    weight: Optional[str] = None,
    x: Optional[str] = None,
    y: Optional[str] = None,
    names: Optional[str] = None,
    color: Optional[str] = None,
    size: Optional[str] = None,
):
    """
    plotly.express-style network graph.

    Parameters
    ----------
    edges: DataFrame
        Pandas data frame containing information about the edges.
        If the data frame has enough columns and
        source, target, weight aren't specified, these will be initialised
        from the first three columns of the data frame.
    nodes: DataFrame or None, default None
        Pandas data frame containing additional information about the nodes.

    Keyword Only
    ------------
    source: str or None, default None
        Column name in edges, describing which column shopuld be taken as
        the source of connections.
    target: str or None, default None
        Column name in edges, describing which column shopuld be taken as
        the target of connections.
    weight: str or None, default None
        Column name in edges, describing which column shopuld be taken as
        the weight of connections.

    x: str or None, default None
        Column name in nodes describing the x position of the nodes.
        If not given, positions are inferred from the weights.
    y: str or None, default None
        Column name in nodes describing the y position of the nodes.
        If not given, positions are inferred from the weights.
    names: str or None, default None
        Column name in nodes containing node labels.
        If not specified, names are inferred from the edges.
    color: str or None, default None
        Column name in nodes determining the colors of the nodes.
    size: str or None, default None
        Column name in nodes determining the size of the nodes.

    Returns
    -------
    figure: plotly figure
        Network graph drawn with plotly

    Notes
    -----
    If both weights are given and position is specified,
    weights are displayed as the opacity of the edges.
    """
    if (source is None) or (target is None):
        source, target, *rest = edges.columns
        if rest and (weight is None):
            weight = rest[0]
    edgelist = _networkx_edges(edges, source, target, weight)  # type: ignore
    nx_graph = nx.Graph(edgelist)
    node_labels = list(nx_graph)
    node_color = "red"
    node_size = 10.0
    edge_color = "#888"
    if (x is None) or (y is None):
        node_x, node_y = get_node_positions(nx_graph)
    else:
        if nodes is None:
            raise ValueError(
                "Please supply a 'nodes' data frame to get x and y values from."
            )
        node_x, node_y = nodes[x], nodes[y]
        if weight is not None:
            affinity = edges[weight]
            affinity = affinity / np.max(affinity)
            edge_color = [f"rgba(0,0,0,{opacity})" for opacity in affinity]
    mapping = {label: index for index, label in enumerate(node_labels)}
    _edges = np.array(nx.relabel_nodes(nx_graph, mapping).edges())
    if nodes is not None:
        if names is not None:
            node_labels = nodes[names]
        if color is not None:
            node_color = nodes[color]
        if size is not None:
            node_size = nodes[size]
    colorbar_title = names or ""
    # I'm solely doing this madness to satisfy the type checker.
    if isinstance(node_x, pd.Series):
        node_x = node_x.to_numpy()
    if isinstance(node_y, pd.Series):
        node_y = node_y.to_numpy()
    return _network(
        node_x=node_x,
        node_y=node_y,
        edges=_edges,
        node_labels=node_labels,  # type: ignore
        node_size=node_size,  # type: ignore
        node_color=node_color,  # type: ignore
        edge_weight=2.5,
        edge_color=edge_color,  # type: ignore
        colorbar_title=colorbar_title,
    )
