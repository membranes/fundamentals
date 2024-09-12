import logging
import typing

import pandas as pd

import config


class Splittings:

    def __init__(self, frame: pd.DataFrame) -> None:
        """

        :param frame: The data set for the modelling stages
        """

        self.__frame = frame

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

        self.__logger.info('parent: %s', parent.shape)
        self.__logger.info('child: %s', child.shape)

        return parent, child

    def exc(self) -> typing.Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """

        :return:
        training: pandas.DataFrame
            The training stage data
        validating: pandas.DataFrame
            The validating stage data
        testing: pandas.DataFrame
            The testing stage data
        """

        training, validating = self.__split(data=self.__frame, frac=self.__configurations.fraction)

        if self.__configurations.aside > 0:
            frac = 1 - self.__configurations.aside
            validating, testing = self.__split(data=validating, frac=frac)
        else:
            testing = None

        return training, validating, testing
