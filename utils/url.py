""" Utility functions related to manipulation of urls """
from typing import Optional
from urllib.parse import urlparse

import pandas as pd


def parse_domain(url: str) -> Optional[str]:
    """
    Normalizes urls to all have the same form.

    Parameters
    ----------
    url: str
        Raw url

    Returns
    -------
    domain: str or None
        Domain name.
        Returns np.nan if the url is invalid
        or does not contain a domain name.
    """
    try:
        domain = urlparse(url).netloc
    except ValueError:
        # if the url is invalid return None
        return None
    # If the domain part is empty (parsing failed)
    # return None
    return domain or None


def normalize_domains(domains: pd.Series) -> pd.Series:
    """
    Adds "www." to domains that are missing it.

    Parameters
    ----------
    domains: Series of str
        Pandas series of domain names

    Returns
    -------
    Series of str

    Note
    ----
    domain_key field in Netarkivet is quite often missing this for some reason,
    while my extraction procedure contains it.
    """
    return domains.where(domains.str.startswith("www."), "www." + domains)


def strip_braces(urls: pd.Series) -> pd.Series:
    """
    Removes braces and values between braces from urls.

    Parameters
    ----------
    urls: Series of str
        Pandas series of urls.

    Returns
    -------
    Series of str

    Note
    ----
    Surprisingly many urls had braces in them,
    and were invalid as a result of that.
    The function is vectorized, make sure to use
    this whenever possible instead of .map
    """
    return (
        urls.str.lower()
        .str.replace(r"\[.*\]", "", regex=True)
        .str.replace(r"\].*", "", regex=True)
    )


URL_REGEX = (
    r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+"
)
