"""
Module loader.py
"""

import torch.utils.data as tu


class Loader:
    """
    Description
    ===========

    For creating DataLoaders.  The Steps class of a model calls this
    class for creating DataLoaders.
    """

    def __init__(self):
        pass


    @staticmethod
    def exc(dataset: tu.Dataset, parameters: dict) -> tu.DataLoader:
        """

        :param dataset:
        :param parameters:
        :return:
        """

        return tu.DataLoader(dataset, **parameters)
