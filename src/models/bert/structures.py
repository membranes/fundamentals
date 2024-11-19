"""Module structures.py"""
import datasets
import pandas as pd
import transformers

import src.elements.arguments as ag
import src.elements.vault as vu
import src.models.bert.dataset


class Structures:
    """
    Class Structures<br>
    ----------------<br>

    Builds and delivers the data structures per modelling stage
    """

    def __init__(self, enumerator: dict, arguments: ag.Arguments, vault: vu.Vault,
                 tokenizer: transformers.tokenization_utils_base):
        """

        :param enumerator:
        :param arguments:
        :param vault:
        :param tokenizer:
        """

        # A set of values, and data, for machine learning model development
        self.__enumerator = enumerator
        self.__arguments = arguments
        self.__vault = vault

        self.__tokenizer = tokenizer

    def __structure(self, frame: pd.DataFrame) -> datasets.Dataset:
        """
        dataloader: tu.DataLoader = self.__loader.exc(
                    dataset=dataset, parameters=parameters)

        :param frame: A data frame
        :return:
            NamedTuple consisting of a torch.util.data.Dataset, and a
            torch.util.data.DataLoader
        """

        dataset = src.models.bert.dataset.Dataset(
            frame=frame, arguments=self.__arguments, enumerator=self.__enumerator,
            tokenizer=self.__tokenizer)

        return dataset

    def training(self) -> datasets.Dataset:
        """
        parameters = {'batch_size': self.__arguments.TRAIN_BATCH_SIZE,
                      'shuffle': True, 'num_workers': 0}

        :return:
        """

        return self.__structure(frame=self.__vault.training)

    def validating(self) -> datasets.Dataset:
        """
        parameters = {'batch_size': self.__arguments.VALID_BATCH_SIZE,
                      'shuffle': True, 'num_workers': 0}

        :return:
        """

        return self.__structure(frame=self.__vault.validating)

    def testing(self) -> datasets.Dataset:
        """
        parameters = {'batch_size': self.__arguments.TEST_BATCH_SIZE,
                      'shuffle': True, 'num_workers': 0}

        :return:
        """

        return self.__structure(frame=self.__vault.testing)
