"""
Script to create Plotly network plots for each of the topics per year
based on precalculated link graphs and semantic summaries.
"""

import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from preprocess_topics import EMNER_PATH, SAVE_PATH
from utils.graph import Graph


def linkgraph_plot(linkgraph: pd.DataFrame, domain_summary: pd.DataFrame) -> go.Figure:
    """
    Produces plot of a link graph with semantic information added.

    Parameters
    ----------
    linkgraph: DataFrame
        Data frame containing edges of a link graph.
        Edge weights should be represented by 'connections'
        and 'semantic_affinity' columns.
    domain_summary: DataFrame
        Semantic summary of domains in a data frame.
        Should contain a 'spread' column.

    Returns
    -------
    Figure
        Plotly network plot.

    Notes
    -----
    Node sizes and node distances represent semantic spread
    and semantic affinity respectively.
    Node colors represent the total number of links of a domain.
    Opacity of edges represent the number of links between two domains.
    Edges are only added between nodes that are connected by the link graph.
    """
    # Creating a semantic and a link graph for extracting different properties
    semantic_graph = Graph.from_edges(linkgraph, "semantic_affinity").remove_loops()
    _linkgraph = Graph.from_edges(linkgraph, "connections").remove_loops()
    # Normalizing number of connections
    norm_connections = linkgraph.connections.to_numpy()
    norm_connections = norm_connections / np.max(norm_connections)
    # Setting edge opacity based on number of connections
    # aka the more connections there are in the link graph,
    # the stronger the color.
    edge_color = [f"rgba(0,0,0,{opacity})" for opacity in norm_connections]
    # Setting node size based on semantic spread.
    node_size = (
        domain_summary.set_index("domain_key")
        .loc[semantic_graph.node_names]
        .spread.to_numpy()
    )
    # Setting node color based on how many connections each
    # domain has in total
    node_color = _linkgraph.n_connections
    return semantic_graph.display(
        edge_weight=3,
        node_size=node_size,
        node_color=node_color,
        edge_color=np.array(edge_color),
    )


def main() -> None:
    """Main function of the script, responsible for going through each year and topic"""
    topics = pd.read_csv(EMNER_PATH)
    for year, keyword in zip(topics.year, topics.emne):
        print(f"Creating plot for: {year} - {keyword}")
        data_path = os.path.join(SAVE_PATH, str(year), keyword)
        linkgraph = pd.read_feather(os.path.join(data_path, "linkgraph_edges.feather"))
        semantic_summary = pd.read_pickle(
            os.path.join(data_path, "semantic_summary.pickle")
        )
        figure = linkgraph_plot(linkgraph, semantic_summary)
        figure.write_html(os.path.join(data_path, "linkgraph_plot.html"))
    print("🥳 DONE 🥳")


if __name__ == "__main__":
    main()
