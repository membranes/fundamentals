"""Module vault.py"""
import typing

import pandas as pd


class Vault(typing.NamedTuple):
    """

    Attributes
    ----------

    training : pandas.DataFrame
        A training data set

    validating : pandas.DataFrame
        A validation data set

    testing : pandas.DataFrame
        A testing data set
    """

    training: pd.DataFrame
    validating: pd.DataFrame
    testing: pd.DataFrame = None
