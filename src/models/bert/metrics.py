"""Module metrics.py"""
import logging

import seqeval.metrics as sme


class Metrics:
    """
    For classification metrics calculations
    """

    def __init__(self):
        """
        Constructor
        """

        # Logging
        logging.basicConfig(level=logging.INFO, format='\n\n%(message)s\n%(asctime)s.%(msecs)03d',
                            datefmt='%Y-%m-%d %H:%M:%S')
        self.__logger = logging.getLogger(__name__)

    def exc(self, originals: list, predictions: list):
        """

        :param originals: The true values
        :param predictions: The predictions
        :return:
        """

        self.__logger.info(originals)
        self.__logger.info(predictions)

        y_true = [originals]
        y_pred = [predictions]

        self.__logger.info(
            sme.classification_report(y_true=y_true, y_pred=y_pred, zero_division=0.0))

        self.__logger.info(
            sme.accuracy_score(y_true=y_true, y_pred=y_pred))
