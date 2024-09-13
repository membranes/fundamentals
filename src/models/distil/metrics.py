"""Module metrics.py"""
import logging
import typing

import numpy as np
import evaluate
import transformers.trainer_utils



class Metrics:
    """

    https://huggingface.co/spaces/evaluate-metric/seqeval/blob/main/seqeval.py
    """

    def __init__(self, archetype: dict):
        """

        :param archetype:
        """

        self.__archetype = archetype
        self.__seqeval = evaluate.load('seqeval')

    def __matrices(self, predictions: np.ndarray, labels: np.ndarray) -> typing.Tuple[list[list], list[list]]:
        """

        :param predictions:
        :param labels:
        :return:
        """

        # Reshaping
        prs = predictions.reshape(-1)
        lls = labels.reshape(-1)

        # Active
        logging.info('Determining active labels & predictions')

        # noinspection DuplicatedCode
        active = np.not_equal(lls, -100)
        _labels = lls[active]
        _predictions = prs[active]

        # Decoding
        labels_ = [self.__archetype[code.item()] for code in _labels]
        predictions_ = [self.__archetype[code.item()] for code in _predictions]

        return [labels_], [predictions_]

    def __lists(self, predictions: np.ndarray, labels: np.ndarray) -> typing.Tuple[list[list], list[list]]:
        """

        :param predictions:
        :param labels:
        :return:
        """

        predictions_: list[list] = [
            [self.__archetype[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        labels_: list[list] = [
            [self.__archetype[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]

        return labels_, predictions_

    def exc(self, bucket: transformers.trainer_utils.PredictionOutput, via_matrices: bool = False):
        """

        :param bucket:
        :param via_matrices
        :return:
        """

        predictions = bucket.predictions
        predictions = np.argmax(predictions, axis=2)
        labels = bucket.label_ids

        if via_matrices:
            labels_, predictions_  = self.__matrices(predictions=predictions, labels=labels)
        else:
            labels_, predictions_ = self.__lists(predictions=predictions, labels=labels)

        # Hence
        logging.info(predictions_[:5])
        logging.info(labels_[:5])
        results = self.__seqeval.compute(predictions=predictions_, references=labels_, zero_division=0.0)

        return {
            "precision": results["overall_precision"],
            "recall": results["overall_recall"],
            "f1": results["overall_f1"],
        }
