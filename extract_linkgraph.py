"""
Script to extract linkgraphs from the textual contents of Netarkivet.
"""

from typing import Iterable, Optional

import pandas as pd

import utils.linkgraph as lg
import utils.stream as st

DEFAULT_CHUNK_SIZE = 10_000


def dfs_to_linkgraph(
    df_stream: Iterable[pd.DataFrame],
    save_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    Turns a stream of record data frames into a linkgraph.

    Parameters
    ----------
    records_stream: iterable of DataFrame
        Stream of record data frames from netarkivet,
        containing a text and domain_key field.
    save_path: str or None, default None
        Path to save the result after each iteration.
        If not specified, saving won't happen.

    Returns
    -------
    linkgraph: DataFrame

    Notes
    -----
    The linkgraph always gets saved as a .feather file.
    Don't forget this when specifying file name.
    Remember to install pyarrow as an optional dependency.
    """
    linkgraph = pd.DataFrame(columns=["source", "target", "connections"])
    # NOTES:
    # Since this can be an embarassingly parallel problem
    # it might make sense to parallelize or at least asynchronize it
    # if performance is not satisfactory
    for records in df_stream:
        current = lg.create_linkgraph(records)
        linkgraph = lg.add(linkgraph, current)
        if save_path is not None:
            linkgraph.to_feather(save_path)
    return linkgraph


def records_to_linkgraph(
    record_stream: Iterable[dict],
    save_path: Optional[str] = None,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
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
    # Chunking records
    record_chunks = st.chunk(record_stream, chunk_size)
    # Turning record chunks into DataFrames
    record_dfs = map(pd.DataFrame.from_records, record_chunks)
    # Running the other function :))
    return dfs_to_linkgraph(record_dfs, save_path)


DATA_PATH = "/work/netarchive/"


def main() -> None:
    """
    Produces a linkgraph for each year.
    Results are saved at /work/linkgraph_cleaned/<year>/extracted_edges.feather

    NOTE: I might write a CLI at one point, that would probs be more convenient.
    """
    print("Starting linkgraph construction")
    for year in range(2006, 2016 + 1):
        records: Iterable[dict] = st.stream_year(DATA_PATH, str(year), verbose=True)
        records_to_linkgraph(
            records,
            save_path=f"/work/linkgraph_cleaned/{year}/extracted_edges.feather",
        )
    print("👌DONE👌")


if __name__ == "__main__":
    main()
