import logging
import typing

import transformers
import numpy as np

import src.elements.structures as sr


class Validation:
    """Validation Class

    <br>
    Summarises a variety of validation steps

    """

    def __init__(self, validating: sr.Structures, archetype: dict):
        """

        :param validating:
        """

        self.__validating = validating
        self.__archetype = archetype

        # Logging
        logging.basicConfig(level=logging.INFO, format='\n\n%(message)s\n%(asctime)s.%(msecs)03d',
                            datefmt='%Y-%m-%d %H-%M-%S')
        self.__logger = logging.getLogger(__name__)

    def exc(self, model: transformers.Trainer) -> typing.Tuple[list, list]:
        """

        :param model:
        :return:
            labels: The codes of the original labels<br>
            predictions: The predicted codes
        """

        # The outputs bucket
        bucket = model.predict(self.__validating.dataset)
        __labels: np.ndarray = bucket.label_ids
        __predictions: np.ndarray = bucket.predictions
        self.__logger.info('Labels: %s', __labels.shape)
        self.__logger.info('Predictions: %s', __predictions.shape)

        # Reshaping
        ref = __labels.reshape(-1)
        matrix = __predictions.reshape(-1, model.model.config.num_labels)
        est = np.argmax(matrix, axis=1)

        # Active
        self.__logger.info('Determining active labels & predictions')
        active = np.not_equal(ref, -100)
        labels = ref[active]
        predictions = est[active]

        # Code -> tag
        labels_ = [self.__archetype[code.item()] for code in labels]
        predictions_ = [self.__archetype[code.item()] for code in predictions]

        return labels_, predictions_
