"""
Module with useful functions for creating an manipulating link graphs.
"""

import pandas as pd

from utils.url import URL_REGEX, normalize_domains, parse_domain, strip_braces

# Creating a link graph has three ditinct steps,
# that I decided to separate for performance considerations.
# This way we can chunk the records and do the calculations in smaller steps
# instead of having to deal with very large DataFrames.
# Since I have no idea how much memory usage I should expect, it is preferable to
# use streams and chunking.

# Before doing any analises you would expect the records to look sth like this:
# note: you may have additional fields, but these are necessary
# +----+-------------------+-------------+
# |    | domain_key        | text        |
# |----+-------------------+-------------|
# |  0 | www.kum.dk        | Kulturmi... |
# |  1 | www.gladsaxe.dk   | "    Kal... |
# |  2 | www.brondbybib.dk | Om hje...   |
# |  3 | www.ekkofilm.dk   | "Forside... |
# |  4 | www.kum.dk        | Kulturmi... |
# +----+-------------------+-------------+


# First step: EXTRACTION
# All links get extracted from the text fields of the records
# Both the domain keys and the links are left unnormalized
#
# Example output:
# +-----+--------------------------+----------------------------------+
# |     | domain_key               | links                            |
# |-----+--------------------------+----------------------------------|
# |  25 | www.musikerbasen.dk      | ['RQNNE.dk...']                  |
# | 109 | www.gladsaxe.dk          | ['kimsbille...']                 |
# | 137 | www.barbue.dk            | ['zentropa....', 'derne.tv2...'] |
# | 147 | www.presseskolen.dk      | ['turbulens...', 'turbulens...'] |
# | 150 | www.konservativungdom.dk | ['zentropa....']                 |
# +-----+--------------------------+----------------------------------+


def extract(records: pd.DataFrame) -> pd.DataFrame:
    """
    Extracts all links from the text fields of records and selects the necessary columns.
    """
    # Extracting links
    links = records.text.str.findall(URL_REGEX)
    records = records.assign(links=links)
    return records[["domain_key", "links"]]


# Second step: EXPANSION
# Link lists get expanded to their own fields, as such each field represents
# one connection from one domain to the other.
# Links are normalized, a connections column is added
# containing all ones, so that connections can be easily added up.
#
# Example output:
# +-----+---------------------+----------------------+---------------+
# |     | source              | target               |   connections |
# |-----+---------------------+----------------------+---------------|
# |  25 | www.musikerbasen.dk | www.rqnne.dk         |             1 |
# | 109 | www.gladsaxe.dk     | www.kimsbilleder.dk  |             1 |
# | 137 | www.barbue.dk       | www.zentropa.dk      |             1 |
# | 137 | www.barbue.dk       | www.nyhederne.tv2.dk |             1 |
# | 147 | www.presseskolen.dk | www.turbulens.net    |             1 |
# +-----+---------------------+----------------------+---------------+


def expand(links: pd.DataFrame) -> pd.DataFrame:
    """
    Expands link lists and normalizes collected and given urls.
    """
    links = links.dropna()
    # Normalizing all domain_keys just to be sure that there is no discrepency
    links = links.assign(domain_key=normalize_domains(links.domain_key))
    # Exploding each link to its own row
    links = links.explode("links").dropna()
    # Removing braces
    links = links.assign(links=strip_braces(links.links))
    # Parsing domains
    links = links.assign(
        links=links.links.map(parse_domain, na_action="ignore"),
        # NOTE: This is rather inefficient
        # Consider writing a vectorized version of normalize_url
        connections=1,
    )
    # Normalizing links columnn
    links = links.assign(links=normalize_domains(links.links))
    links = links.rename(columns={"domain_key": "source", "links": "target"})
    return links.dropna()


# Third step: COLLAPSE
# All source, target pairs get summed up.
# Note that this function can also be used for adding two separate linkgraphs
# thereby allowing us to add results of different chunks
#
# Example output:
# +----+-------------------+--------------------------+---------------+
# |    | source            | target                   |   connections |
# |----+-------------------+--------------------------+---------------|
# |  0 | www.alfarvej.dk   | www.alfarvej.dk          |             1 |
# |  1 | www.allanolsen.dk | www.annettebjergfeldt.dk |             1 |
# |  2 | www.allanolsen.dk | www.apple.com            |             1 |
# |  3 | www.allanolsen.dk | www.cnn.com              |             1 |
# |  4 | www.allanolsen.dk | www.dimeadozen.org       |             3 |
# +----+-------------------+--------------------------+---------------+


def collapse(connections: pd.DataFrame) -> pd.DataFrame:
    """
    Sums up all connections for source, target pairs.
    """
    return (
        connections.groupby(
            ["source", "target"]
        )  # Grouping values based on source-target pairs
        .sum()  # Summing up all connections
        .reset_index()  # Resetting index, so that source and target become columns
    )


def add(*linkgraphs: pd.DataFrame) -> pd.DataFrame:
    """
    Adds multiple linkgraphs together, allowing you to add up results
    of chunks.
    """
    linkgraph = pd.concat(linkgraphs, ignore_index=True)
    return collapse(linkgraph)


def create_linkgraph(records: pd.DataFrame) -> pd.DataFrame:
    """
    Runs the entire linkgraph pipeline on a set of records.
    """
    links = extract(records)
    connections = expand(links)
    linkgraph = collapse(connections)
    return linkgraph
