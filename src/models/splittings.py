"""Module splittings.py"""
import logging
import typing

import numpy as np
import pandas as pd

import config


class Splittings:
    """
    Class Splittings
    """

    def __init__(self) -> None:
        """
        Constructor
        """

        # Configurations
        self.__configurations = config.Config()

        # Logging
        logging.basicConfig(level=logging.INFO,
                            format='\n\n%(message)s\n%(asctime)s.%(msecs)03d',
                            datefmt='%Y-%m-%d %H:%M:%S')
        self.__logger = logging.getLogger(__name__)

    def __split(self, data: pd.DataFrame, frac: float) -> typing.Tuple[pd.DataFrame, pd.DataFrame]:
        """
        This method splits  data set into parent & child data sets.

        :return:
        parent : pandas.DataFrame<br>
            The data set for parent<br>
        child : pandas.DataFrame<br>
            The data set for the child stage
        """

        blob = data.copy()

        parent = blob.sample(frac=frac, random_state=self.__configurations.seed)
        child = blob.drop(parent.index)

        parent.reset_index(drop=True, inplace=True)
        child.reset_index(drop=True, inplace=True)

        return parent, child

    def exc(self, data: pd.DataFrame) -> typing.Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """

        :param data: The data set for the modelling stages
        :return:
            training: pandas.DataFrame
                The training stage data
            validating: pandas.DataFrame
                The validating stage data
            testing: pandas.DataFrame
                The testing stage data
        """

        training, validating = self.__split(data=data, frac=self.__configurations.fraction)

        if self.__configurations.aside > 0:
            frac = 1 - self.__configurations.aside
            validating, testing = self.__split(data=validating, frac=frac)
        else:
            testing = pd.DataFrame()

        self.__logger.info('training: %s', training.shape)
        self.__logger.info('validating: %s', validating.shape)
        self.__logger.info('testing: %s', testing.shape if not testing.empty else np.nan)

        return training, validating, testing
