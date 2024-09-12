"""Module interface.py"""
import pandas as pd

import src.models.splittings
import src.models.bert.steps
import src.models.distil.steps
import src.elements.frames


class Interface:
    """
    An interface to each model.
    """

    def __init__(self, frame: pd.DataFrame, enumerator: dict, archetype: dict):
        """

        :param frame: The data set for the training, validating, etc., stages
        :param enumerator:
        :param archetype:
        """

        self.__training, self.__validating, _ = src.models.splittings.Splittings(frame=frame).exc()
        self.__frames = src.elements.frames.Frames(training=self.__training, validating=self.__validating)
        self.__enumerator = enumerator
        self.__archetype = archetype

    def exc(self) -> None:
        """

        :return:
        """

        # bert
        # src.models.bert.steps.Steps(
        #     enumerator=self.__enumerator, archetype=self.__archetype, frames=self.__frames).exc()

        # electra
        # src.models.electra.steps

        # distil
        src.models.distil.steps.Steps(
            enumerator=self.__enumerator, archetype=self.__archetype, frames=self.__frames).exc()

        # Transfer Learning with BiLSTM, BERT and CRF
        # https://link.springer.com/article/10.1007/s42979-024-02835-z
