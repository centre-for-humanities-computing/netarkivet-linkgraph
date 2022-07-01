from typing import Union, List, TypeVar
import warnings

import numpy as np
import pandas as pd
from scipy.stats import zscore

T = TypeVar("T", pd.Series, pd.DataFrame, np.ndarray)

def filter_outliers(data: T, columns: Union[str, List[str], None] = None) -> T:
    """
    Filters outliers from a sequence of data.
    
    Parameters
    ----------
    data: Series, DataFrame or ndarray
        Collection of data to remobve outliers from.
    columns: str, list of str, or None, default None
        Column name or list of column names to filter outliers from.
        If not specified, all columns will be filtered.
        If data is not a data frame, this parameter
        will be ignored.
    
    Returns
    -------
    Series, DataFrame or ndarray
        Same type of object as passed in, without outliers.
    
    Note
    ----
    Data points are considered outliers if their absolute
    z-score is larger than or equal to 3.
    """
    if isinstance(data, (pd.Series, np.ndarray)):
        if columns is not None:
            warnings.warn(
                "'columns' is only a valid parameter if the 'data' is a DataFrame, ignoring."
            )
        is_outlier = np.abs(zscore(data)) >= 3
    elif isinstance(data, pd.DataFrame):
        if columns is None:
            columns = data.columns
        subset = data[columns]
        is_outlier_each = subset.apply(zscore) >= 3
        is_outlier = is_outlier_each.agg(np.any, axis=1)
    else:
        arg_type = type(data)
        raise TypeError(
            f"Please supply a Series, ndarray or DataFrame, {arg_type} is not supported."
        )
    return data[~is_outlier]