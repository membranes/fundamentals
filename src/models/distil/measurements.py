"""Module metrics.py"""
import logging

import seqeval.metrics as sme
import sklearn.metrics as sm


class Measurements:
    """
    For classification metrics calculations
    """

    def __init__(self):
        """
        Constructor
        """

        # Logging
        logging.basicConfig(level=logging.INFO,
                            format='\n\n%(message)s\n%(asctime)s.%(msecs)03d',
                            datefmt='%Y-%m-%d %H:%M:%S')
        self.__logger = logging.getLogger(__name__)

    def __sci(self, originals: list, predictions: list):
        """

        :param originals:
        :param predictions:
        :return:
        """

        self.__logger.info('SCIKIT LEARN')

        self.__logger.info(
            sm.classification_report(y_true=originals, y_pred=predictions, zero_division=0.0))

    def __seq(self, originals: list, predictions: list):
        """

        :param originals:
        :param predictions:
        :return:
        """

        y_true = [originals]
        y_pred = [predictions]

        self.__logger.info('SEQ EVAL')

        self.__logger.info(
            sme.classification_report(y_true=y_true, y_pred=y_pred, zero_division=0.0))

        self.__logger.info(
            sme.accuracy_score(y_true=y_true, y_pred=y_pred))

    def exc(self, originals: list, predictions: list):
        """

        :param originals: The true values
        :param predictions: The predictions
        :return:
        """

        self.__sci(originals=originals, predictions=predictions)
        self.__seq(originals=originals, predictions=predictions)
