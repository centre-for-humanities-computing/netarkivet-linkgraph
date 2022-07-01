""" Utility functions related to text manipulation such as normalization, tokenization sentencization """
import re
import string
from typing import List, Union, Literal

import numpy as np

from url_normalize import url_normalize


def only_dots(text: str) -> str:
    """
    Exchanges all question marks and exclamation marks in the text for dots.

    Parameters
    ----------
    text: str
        Text to alter

    Returns
    ----------
    text: str
        New string containing only dots
    """
    return text.translate(str.maketrans({"?": ".", "!": ".", ";": "."}))


def remove_digits(text: str) -> str:
    """
    Removes all digits from the text.

    Parameters
    ----------
    text: str
        Text to alter

    Returns
    ----------
    text: str
        New string without digits
    """
    return text.translate(str.maketrans("", "", string.digits))


PUNCT = "\"#$%&'()*+,-/:<=>@[\\]^_`{|}~"


def remove_punctuation(text: str, keep_sentences: bool) -> str:
    """
    Replaces all punctuation from the text with spaces
    except for dots, exclamation marks and question marks.

    Parameters
    ----------
    text: str
        Text to alter
    keep_sentences: bool
        Specifies whether the normalization should keep sentence borders or not
        (exclamation marks, dots, question marks)

    Returns
    ----------
    text: str
        New string without punctuation
    """
    if keep_sentences:
        punctuation = PUNCT
    else:
        punctuation = string.punctuation
    return text.translate(str.maketrans(punctuation, " " * len(punctuation)))


DANISH_CHARACTERS = string.printable + "åæøÅÆØéáÁÈ"


def normalize(text: str, keep_sentences: bool = False) -> str:
    """
    Removes digits and punctuation from the text supplied.

    Parameters
    ----------
    text: str
        Text to alter
    keep_sentences: bool
        Specifies whether the normalization should keep sentence borders or not
        (exclamation marks, dots, question marks)

    Returns
    ----------
    text: str
        New normalized string
    """
    text = remove_digits(text)
    text = remove_punctuation(text, keep_sentences=keep_sentences)
    text = text.lower()
    # Removing non Danish characters
    non_danish = re.compile(f"[^{DANISH_CHARACTERS}]")
    text = re.sub(non_danish, " ", text)
    # Strips accents
    # table = str.maketrans({"é": "e'", "á": "a'"})
    # text = text.translate(table)
    return text


def tokenize(text: str) -> List[str]:
    """
    Tokenizes cleaned text.

    Parameters
    ----------
    text: str
        Text to tokenize

    Returns
    ----------
    tokens: list of str
        List of tokens
    """
    return text.split()


def sentencize(text: str) -> List[List[str]]:
    """
    Cleans up the text, sentencizes and tokenizes it.

    Parameters
    ----------
    text: str
        Text to sentencize

    Returns
    ----------
    sentences: list of list of str
        List of sentences in the form of list of tokens.
    """
    norm_text = normalize(text, keep_sentences=True)
    sentences = only_dots(norm_text).split(".")
    return [tokenize(sentence) for sentence in sentences]


def normalized_document(text: str) -> List[str]:
    """
    Normalizes text, then tokenizes it.

    Parameters
    ----------
    text: str
        Text to sentencize

    Returns
    ----------
    tokens: list of str
        List of tokens in the text.
    """
    norm_text = normalize(text, keep_sentences=False)
    return tokenize(norm_text)

NA = Literal[np.nan]

def normalize_url(url: str) -> Union[str, NA]:
    """
    Normalizes urls to all have the same form.

    Parameters
    ----------
    url: str
        Raw url

    Returns
    -------
    domain: str or NA
        Normalized url of form: 'www.<domain_name>'
        Returns np.nan if the url is invalid.
    """
    try:
        url = url_normalize(url)
    except ValueError:
        return np.nan
    domain = url.split("/")[2]
    if not domain.startswith("www."):
        domain = "www." + domain
    return domain


URL_REGEX = (
    r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+"
)


def get_urls(text: str) -> List[str]:
    """
    Finds all urls in the text based on a regex.

    Parameters
    ----------
    text: str
        Textual content of a website, to get urls out of.

    Returns
    -------
    urls: list of str
        List of urls found in the text

    Notes
    -----
    Could probably be more refined.
    """
    urls = re.findall(URL_REGEX, text)
    urls = list(map(normalize_url, urls))
    return urls
