"""Module structures.py"""
import logging

import pandas as pd
import torch.utils.data as tu
import transformers

import src.elements.frames as fr
import src.elements.structures as sr
import src.elements.variable as vr
import src.models.dataset
import src.models.loadings


class Structures:
    """
    Collecting<br>
    ----------<br>

    Builds and delivers the data structures per modelling stage
    """

    def __init__(self, enumerator: dict, variable: vr.Variable, frames: fr.Frames,
                 tokenizer: transformers.tokenization_utils_base):
        """

        :param enumerator:
        :param variable:
        :param frames:
        """

        # A set of values, and data, for machine learning model development
        self.__enumerator = enumerator
        self.__variable = variable
        self.__frames = frames

        self.__tokenizer = tokenizer

        # For DataLoader creation
        self.__loadings = src.models.loadings.Loadings()

        # Logging
        logging.basicConfig(level=logging.INFO,
                            format='\n\n%(message)s\n%(asctime)s.%(msecs)03d',
                            datefmt='%Y-%m-%d %H:%M:%S')
        self.__logger = logging.getLogger(__name__)

    def __structure(self, frame: pd.DataFrame, parameters: dict) -> sr.Structures:
        """

        :param frame: A data frame
        :param parameters: The data frame's corresponding modelling stage parameters
        :return:
            NamedTuple consisting of a torch.util.data.Dataset, and a
            torch.util.data.DataLoader
        """

        dataset = src.models.dataset.Dataset(
            frame=frame, variable=self.__variable, enumerator=self.__enumerator,
            tokenizer=self.__tokenizer)

        dataloader: tu.DataLoader = self.__loadings.exc(
            dataset=dataset, parameters=parameters)

        return sr.Structures(dataset=dataset, dataloader=dataloader)

    def training(self) -> sr.Structures:
        """
        Delivers the training data's Dataset & DataLoader

        :return:
        """

        # Modelling parameters
        parameters = {'batch_size': self.__variable.TRAIN_BATCH_SIZE,
                      'shuffle': True, 'num_workers': 0}

        return self.__structure(frame=self.__frames.training, parameters=parameters)

    def validating(self) -> sr.Structures:
        """
        Delivers the validation data's Dataset & DataLoader

        :return:
        """

        # Modelling parameters
        parameters = {'batch_size': self.__variable.VALID_BATCH_SIZE,
                      'shuffle': True, 'num_workers': 0}

        return self.__structure(frame=self.__frames.validating, parameters=parameters)

    def testing(self) -> sr.Structures:
        """
        Delivers the testing data's Dataset & DataLoader

        :return:
        """

        # Modelling parameters
        parameters = {'batch_size': self.__variable.TEST_BATCH_SIZE,
                      'shuffle': True, 'num_workers': 0}

        return self.__structure(frame=self.__frames.testing, parameters=parameters)
