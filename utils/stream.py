"""
Module containing utility functions for streaming data
"""

import json
import os
import random
import re
from itertools import islice, zip_longest
from typing import Callable, Iterable, List, Optional, Tuple, TypeVar

import pandas as pd
from gensim.models.doc2vec import TaggedDocument
from tqdm import tqdm


def stream_edges(
    data_path: str = "/work/linkgraph_cleaned/",
) -> Iterable[Tuple[int, pd.DataFrame]]:
    """
    Streams edge data frames for all years

    Parameters
    ----------
    data_path: str, default "/work/linkgraph_cleaned/"
        Path to stream the data from

    Yields
    ---------
    year: int
        Year of the supplied edges dataframe
    edges: DataFrame
        Dataframe containing information for the current year
    """
    for year in range(2006, 2016 + 1):
        year_path = os.path.join(data_path, str(year))
        path = os.path.join(year_path, "edges.feather")
        try:
            edges = pd.read_feather(path)
        except FileNotFoundError:
            print(f"{year}/edges.feather not Found, continuing")
            continue
        yield year, edges


T = TypeVar("T")


def flatten(nested: Iterable[Iterable[T]]) -> Iterable[T]:
    """
    Function that turns a nested stream into a flat stream.
    Parameters
    ----------
    nested: iterable of iterable of T
        Nested iterable that you want to flatten
    Yields
    ----------
    element: T
        Individual elements of the nested iterable
    """
    for sub in nested:
        for elem in sub:
            yield elem


def reusable(gen_func: Callable) -> Callable:
    """
    Function decorator that turns your generator function into an
    iterator, thereby making it reusable.
    Parameters
    ----------
    gen_func: Callable
        Generator function, that you want to be reusable
    Returns
    ----------
    _multigen: Callable
        Sneakily created iterator class wrapping the generator function
    """

    class _multigen:
        def __init__(self, *args, limit=None, **kwargs):
            self.__args = args
            self.__kwargs = kwargs
            self.limit = limit

        def __iter__(self):
            if self.limit is not None:
                return islice(gen_func(*self.__args, **self.__kwargs), self.limit)
            return gen_func(*self.__args, **self.__kwargs)

    return _multigen


U = TypeVar("U")


@reusable
def chunk(
    iterable: Iterable[U], chunk_size: int, sample_size: Optional[int] = None
) -> Iterable[List[U]]:
    """
    Generator function that chunks an iterable for you.
    Parameters
    ----------
    iterable: Iterable of T
        The iterable you'd like to chunk.
    chunk_size: int
        The size of chunks you would like to get back
    sample_size: int or None, default None
        If specified the yielded lists will be randomly sampled with the buffer
        with replacement. Sample size determines how big you want those lists to be.
    Yields
    ----------
    buffer: list of T
        sample_size or chunk_size sized lists chunked from the original iterable
    """
    buffer = []
    for index, elem in enumerate(iterable):
        buffer.append(elem)
        if (index % chunk_size == (chunk_size - 1)) and (index != 0):
            if sample_size is None:
                yield buffer
            else:
                yield random.choices(buffer, k=sample_size)
            buffer = []


def chunked(chunk_size: int, sample_size: Optional[int] = None) -> Callable:
    """
    Decorator that chunks a generator function.
    Parameters
    ----------
    chunk_size: int
        The size of chunks you would like to get back
    sample_size: int or None, default None
        If specified the yielded lists will be randomly sampled with the buffer
        with replacement. Sample size determines how big you want those lists to be.
    Returns
    ----------
    _chunked: Callable
        Wrapper for the generator function.
    """

    def _chunked(gen_func: Callable):
        def _iterable(*args, **kwargs):
            return chunk(gen_func(*args, **kwargs), chunk_size, sample_size=sample_size)

        return _iterable

    return _chunked


def get_years(data_path: str = ".") -> List[str]:
    """
    Gets the names of the year folders.
    Parameters
    ----------
    data_path: str, default '.'
        Specifies where our data lives
    Returns
    ----------
    years: list of str
        List of all years processed
    """
    years = []
    for root, dirs, files in os.walk(data_path):
        for directory in dirs:
            if re.match(r"\d\d\d\d", directory):
                years.append(directory)
    return years


def stream_records_from_file(file_path: str) -> Iterable[dict]:
    """
    Streams all records from the file at the given path that have passed
    the quality filters, are in Danish and aren't duplicates
    Parameters
    ----------
    file_path: str
        Path to the file you'd like to stream
    Yields
    ----------
    record: dict
        Each record from the file
    """
    with open(file_path) as input_file:
        # Since id is not one of the fields, I have to enumerate all records
        for line in input_file:
            # parsing the record
            record = json.loads(line)
            # If passes quality filters, it yields the content of the record
            record_okay = (
                record["passed_quality_filter"]
                and record["language"] == "da"
                and not record["is_duplicate"]
            )
            if record_okay:
                yield record


I = TypeVar("I")

BAR_LENGTH = 100
N_DECIMALS = 1
FILL_CHARACTER = "█"


@reusable
def progress_bar_stream(items: List[I]) -> Iterable[I]:
    """
    Wraps list in an iterable that shows a progress bar and the current element.
    Parameters
    ----------
    items: list of U
        Items to iterate over (of type U)
    Yields
    ----------
    item: U
        Current item under processing
    """
    from IPython.display import clear_output

    total = len(items)
    for iteration, item in enumerate(items):
        percent = ("{0:." + str(N_DECIMALS) + "f}").format(
            100 * (iteration / float(total))
        )
        filled_length = int(BAR_LENGTH * iteration // total)
        progress_bar = FILL_CHARACTER * filled_length + "-" * (
            BAR_LENGTH - filled_length
        )
        clear_output(wait=True)
        print(
            f"Progress: |{progress_bar}| {percent}% \n Current item processed: {item}\n"
        )
        yield item


@reusable
def stream_year(
    data_path: str,
    year: str,
    verbose: bool = True
) -> Iterable[dict]:
    """
    Streams all records from a given year.
    Parameters
    ----------
    data_path: str
        Path to the dataset
    year: str
        The year from which records should be streamed.
    Yields
    ----------
    record: dict
        Records from the given year.
    """
    for root, _, files in os.walk(os.path.join(data_path, f"{year}")):
        # Go through all files in the year directory
        if verbose:
            files = tqdm(files, desc=f"Processing year: {year}")
        for file in files: #progress_bar_stream(files):
            # If it's a jsonl file, stream all records from it
            if file.endswith(".jsonl"):
                records = stream_records_from_file(os.path.join(root, file))
                for record in records:
                    yield record


@reusable
def to_text_stream(records: Iterable[dict]) -> Iterable[str]:
    """
    Turns a stream of records to a stream of texts
    Parameters
    ----------
    records: iterable of dict
        Stream of records you want to turn into texts
    Yields
    ----------
    text: str
        Texts extracted from the records
    """
    for record in records:
        yield record["text"]


@reusable
def stream_all_records(data_path: str) -> Iterable[dict]:
    """
    Generator yielding all records from the dataset.
    Parameters
    ----------
    data_path: str
        Specifies where our data lives, where to get file contents from.
    Yields
    ----------
    record: dict
        All records
    """
    # List of all years
    years = get_years(data_path=data_path)
    # Collects streams of all years into a list
    year_streams = [stream_year(data_path, year) for year in years]
    # Streams records from all years at the same time, so that the data is more shuffled
    # We use the zip_longest function from itertools, so that we iterate as
    # long as the longest iterable is not exhausted
    # Once a shorter iterable is exhausted, we will get None values.
    for records in zip_longest(*year_streams, fillvalue=None):
        for record in records:
            # If the record is not from an exhausted stream, we yield it
            if record is not None:
                yield record

def filter_contains(records: Iterable[dict], keyword: str) -> Iterable[dict]:
    """
    Filters a stream of records based on whether they contain the given keyword.
    
    Parameters
    ----------
    records: iterable of dict
        Stream of records to filter
    keyword: str
        Keyword, the records have to contain
    
    Yields
    ------
    record: dict
        Records containing keyword
    """
    for record in records:
        if keyword in record["text"].lower():
            yield record