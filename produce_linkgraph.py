"""
Module for creating linkgraph from a stream of records.
"""

from typing import Iterable, Optional
import sys

import pandas as pd
from tabulate import tabulate

import utils.linkgraph as lg
import utils.stream as st

DEFAULT_CHUNK_SIZE = 10_000


def memory_info(data: pd.DataFrame) -> str:
    """
    Displays some tabulated information about memory usage of a DataFrame.
    """
    mem_use = (
        data
        .memory_usage()
        .reset_index()
        .rename(columns={"index": "column", 0: "memory_usage"})
    )
    return tabulate(mem_use, headers="keys", tablefmt="psql")


def records_to_linkgraph(
    record_stream: Iterable[dict],
    save_path: Optional[str] = None,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Turns any stream of records into a linkgraph by chunking.

    Parameters
    ----------
    records_stream: iterable of dict
        Stream of records from netarkivet,
        containing a text and domain_key field.
    save_path: str or None, default None
        Path to save the result after each iteration.
        If not specified, saving won't happen.
    chunk_size: int, default 15_000
        Number of records to process at once.
    verbose: bool, default False
        If set to True, the function will print out

    Returns
    -------
    linkgraph: DataFrame

    Notes
    -----
    The linkgraph always gets saved as a .feather file.
    Don't forget this when specifying file name.
    Remember to install pyarrow as an optional dependency.
    """
    record_chunks = st.chunk(record_stream, chunk_size)
    linkgraph = pd.DataFrame(columns=["source", "target", "connections"])
    # NOTES:
    # Since this can be an embarassingly parallel problem
    # it might make sense to parallelize or at least asynchronize it
    # if performance is not satisfactory
    for i, chunk in enumerate(record_chunks):
        chunk_df = pd.DataFrame.from_records(chunk)
        # NOTES:
        # This is nice and concise, but I'll have to test
        # whether it would make sense to extract, expand,
        # and then add it to the previous result.
        # There is possibly a performance benefit to only collapsing once,
        # But I will have to do some profiling.
        current = lg.create_linkgraph(chunk_df)
        linkgraph = lg.add(linkgraph, current)
        if save_path is not None:
            linkgraph.to_feather(save_path)
        if verbose:
            pass
            # print("\r", end="")
            # summary = f"Chunk {i} done:\n" + memory_info(linkgraph)
            # print(summary, end="")
            # for _ in range(summary.count("\n")):
            #     sys.stdout.write("\x1b[1A\x1b[2K")
    return linkgraph


DATA_PATH = "/work/netarkivet-cleaned/"


def main() -> None:
    """
    Produces a linkgraph for 2006 by default.
    Saves results to '/work/linkgraph_cleaned/2006/new_edges.feather'

    NOTE: I might write a CLI at one point, that would probs be more convenient.
    """
    print("Starting linkgraph construction")
    records = st.stream_year(DATA_PATH, 2006)
    records_to_linkgraph(
        records,
        save_path="/work/linkgraph_cleaned/2006/new_edges.feather",
        verbose=True,
    )
    print("👌DONE👌")


if __name__ == "__main__":
    main()
