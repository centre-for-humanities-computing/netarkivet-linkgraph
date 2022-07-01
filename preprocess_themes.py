"""
Script for collecting records and preprocessing them for further analysis.
Additionally the file functions as a module containing utility functions
for dealing with the data produced.
"""

import json
import os
from functools import partial
from typing import List, Literal, TypeVar, Union

import numpy as np
import pandas as pd
from gensim.models import Doc2Vec
from sklearn.metrics import pairwise_distances

from utils.construction import undirected_edges
from utils.graph import Graph
from utils.stream import filter_contains, stream_year
from utils.text import get_urls, normalize_url, normalized_document
from utils.linkgraph import create_linkgraph


# Path to Netarkivet
DATA_PATH = "/work/netarkivet-cleaned/"
# Path to the Doc2Vec model used for semantic distance
DOC2VEC_PATH = "/work/doc2vec/third-porn_filtered/doc2vec.model"


def process_records(records_path: str, doc2vec: Doc2Vec) -> pd.DataFrame:
    """
    Function to process all save records from a given path,
    collect them to a DataFrame, add normalized, tokenized documents,
    and document embeddings. It normalizes domain keys
    and collects hyperlinks from all records.

    Parameters
    ----------
    records_path: str
        Path to jsonl records file.
    doc2vec: Doc2Vec
        Gensim document embedding model.

    Returns
    -------
    DataFrame
        Pandas data frame containing all records + the following added columns.
    doc: list of str
        Normalized, tokenized documents
    embedding: np.ndarray
        Doc2Vec embedding for each record's textual content
    links: list of str
        All hyperlinks contained in the documents stripped down to
        domains.
    """
    # Reading in these records
    df = pd.read_json(records_path, orient="records", lines=True)
    # Normalizing and preprocessing text
    df = df.assign(doc=df["text"].map(normalized_document))
    # Inferring embeddings for the records in question
    df = df.assign(embedding=df.doc.map(doc2vec.infer_vector))
    # Getting out urls from the collected records
    df = df.assign(links=df.text.map(get_urls))
    # Normalizing the domain_key field so that it would match with the collected urls
    df = df.assign(domain_key=df.domain_key.map(normalize_url))
    return df


def spread(embeddings: np.ndarray) -> float:
    """
    Calculates semantic spread from a bunch of document embeddings.

    Parameters
    ----------
    embeddings: ndarray of shape (n_documents, n_features)
        Matrix of document embeddings

    Returns
    -------
    float
        Semantic spread of embeddings
    """
    # Computing standard deviation along all dimensions
    std = np.std(embeddings, axis=0)
    # Returning the sum of standard deviations along all dimensions
    return np.sum(std)


def create_semantic_summary(records: pd.DataFrame) -> pd.DataFrame:
    """
    Summarises document embeddings for each domain.

    Parameters
    ----------
    records: DataFrame
        Pandas data frame containing processed records with embeddings added.

    Returns
    -------
    DataFrame
        Pandas DataFrame, where the indices are domains, and have the
        following columns:
    centroid: np.ndarray
        Centroids of all embeddings related to one domain.
    spread: float
        Semantic spread of all embedings related to one domain.
        aka. sum of standard deviations across all dimensions.
    """
    # Stacking up all embeddings related to one domain
    summary = (
        records[["domain_key", "embedding"]].groupby("domain_key").aggregate(np.stack)
    )
    # Taking the mean and standard deviation of all vectors in
    summary = summary.assign(
        spread=summary["embedding"].map(spread),
        # Taking the centroid of all embeddings belonging to the same domain as
        # position on the graph
        centroid=summary["embedding"].map(partial(np.mean, axis=0)),
    )
    #Adding count of occurances
    summary = (
        records
        .value_counts("domain_key")
        .reset_index(name="counts")
        .merge(
            summary,
            on="domain_key",
            how="right"
        )
    )
    return summary

T = TypeVar("T")

def not_empty(values: List[T]) -> Union[List[T], Literal[np.nan]]:
    """
    Checks if the given list of values is empty,
    if it contains elements, it returns the list.
    otherwise it returns nan.
    """
    return values if values else np.nan


def create_semantic_graph(semantic_summary: pd.DataFrame) -> Graph:
    """
    Creates semantic graph based on semantic summaries of domains.

    Parameters
    ----------
    semantic_summary: DataFrame
        Pandas data frame containing semantic summaries of domains.

    Returns
    -------
    Graph
        Graph based on semantic affinity.
    """
    domains = semantic_summary.domain_key.tolist()
    key_to_index = {domain: index for index, domain in enumerate(domains)}
    index_to_key = {index: domain for index, domain in enumerate(domains)}
    # Stacking centroids for all domains in a matrix
    X = np.stack(semantic_summary.centroid)
    # Calculating distance matrix for the semantic graph
    affinity = 1 - pairwise_distances(X, metric="cosine")
    # Creating Graph object
    return Graph(affinity, key_to_index, index_to_key)


def add_semantic_affinity(
    linkgraph: pd.DataFrame, semantic_graph: Graph
) -> pd.DataFrame:
    """
    Adds semantic distance for each edge of the linkgraph.

    Parameters
    ----------
    linkgraph: DataFrame
        Pandas data frame containing the link graph
    semantic_graph: Graph
        Semantic graph where indexing results in the semantic affinity of
        the two nodes.

    Returns
    -------
    DataFrame
        Pandas data frame containing the link graph,
        and the following columns:
    semantic_affinity: float
        Semantic affinity between the centroids of the
        two domains.

    Notes
    -----
    Beware, that the supplied linkgraph is turned into an undirectional one
    during this operation.
    The resulting data frame will not contain information about the directionality
    of the links.
    """
    distances_df = undirected_edges(linkgraph)
    distances_df = distances_df.assign(
        semantic_affinity=distances_df.apply(
            lambda s: semantic_graph[s.source, s.target], axis=1
        )
    )
    return distances_df


def collect_records(year: str, keyword: str) -> None:
    """
    Collects all record from year, containing keyword
    into RECORDS_SAVE_PATH jsonl file.
    """
    records = stream_year(DATA_PATH, year)
    records = filter_contains(records, keyword)
    json_stream = map(json.dumps, records)
    with open(RECORDS_SAVE_PATH, "w") as records_file:
        for json_string in json_stream:
            records_file.write(json_string + "\n")


def preprocess_theme(year: str, keyword: str) -> None:
    """
    Collects records, creates semantic summaries and linkgraph edges.
    """
    save_dir = f"/work/linkgraph_cleaned/{year}/{keyword}/"
    #Making all missing directories
    os.makedirs(save_dir, exist_ok=True)
    # If the records are already saved it would be pointless to collect them again
    # if you want to repeat the collection process, delete the file.
    print("  a) Record collection")
    records_path = os.path.join(save_dir, "records.jsonl")
    if not os.path.isfile(records_path):
        print(f"   Collecting records to {records_path}")
        print("   Note: this might take quite a bit of time, not to worry")
        collect_records(year, keyword)
    else:
        print(
            f"    Records file at {records_path} already exists, skipping collection."
        )
    # Since the rest doesn't take forever, we can knock ourselves out with repeating them
    
    print("  b) Record processing")
    print(f"    - Loading document embedding model from {DOC2VEC_PATH}")
    doc2vec = Doc2Vec.load(DOC2VEC_PATH)
    print("    - Processing records..")
    processed_df = process_records(records_path, doc2vec)
    data_frame_path = os.path.join(save_dir, "records.pickle")
    print(f"    - Saving DataFrame to {data_frame_path}")
    processed_df.to_pickle(data_frame_path)
    print("  c) Semantic summaries")
    summary = create_semantic_summary(processed_df)
    summary_path = os.path.join(save_dir, "semantic_summary.pickle")
    print(f"    - Saving to {summary_path}")
    summary.to_pickle(summary_path)
    print("  d) Linkgraph")
    print("    - Summarizing links")
    linkgraph = create_linkgraph(processed_df)
    print("    - Filtering for domains that are actually in the collected dataset")
    domains = summary.reset_index().domain_key
    linkgraph = linkgraph[
        linkgraph.source.isin(domains) & linkgraph.target.isin(domains)
    ]
    print("    - Adding semantic affinities")
    semantic_graph = create_semantic_graph(summary)
    linkgraph = add_semantic_affinity(linkgraph, semantic_graph)
    linkgraph_path = os.path.join(save_dir, "linkgraph_edges.feather")
    print(f"    - Saving to {linkgraph_path}")
    linkgraph.to_feather(linkgraph_path)
    print("  👌DONE👌")


EMNER_PATH = "/work/linkgraph_cleaned/emner.csv"
    
def main() -> None:
    """
    Main function, collects data for all themes and year.
    """
    emner = pd.read_csv(EMNER_PATH)
    for year, keyword in zip(emner.year, emner.emne):
        print(f"Collection and processing for: {keyword} in {year}")
        preprocess_theme(year, keyword)

if __name__ == "__main__":
    main()