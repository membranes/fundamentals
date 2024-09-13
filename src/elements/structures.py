"""Module structures.py"""
import typing

import torch.utils.data as tu

class Structures(typing.NamedTuple):
    """

    Attributes
    ----------

    dataset : torch.utils.data.Dataset
        A data object for modelling

    dataloader : torch.utils.data.DataLoader
        A data object that consists of a Dataset, and a
        set of modelling parameters
    """

    dataset: tu.Dataset
    dataloader: tu.DataLoader
