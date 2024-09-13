"""Module steps.py"""
import logging

import src.elements.frames as fr
import src.elements.variable as vr
import src.models.distil.architecture
import src.models.distil.measurements
import src.models.distil.tokenizer
import src.models.distil.validation
import src.models.structures


class Steps:
    """
    Steps
    """

    def __init__(self, enumerator: dict, archetype: dict, frames: fr.Frames):
        """

        :param enumerator:
        :param archetype:
        :param frames:
        """

        # Inputs
        self.__enumerator = enumerator
        self.__archetype = archetype
        self.__frames = frames

        # A set of values for machine learning model development
        self.__variable = vr.Variable()
        self.__variable = self.__variable._replace(EPOCHS=4, TRAIN_BATCH_SIZE=16, VALID_BATCH_SIZE=16)

        # ...
        self.__tokenizer = src.models.distil.tokenizer.Tokenizer()()

        # Logging
        logging.basicConfig(level=logging.INFO,
                            format='\n\n%(message)s\n%(asctime)s.%(msecs)03d',
                            datefmt='%Y-%m-%d %H:%M:%S')
        self.__logger = logging.getLogger(__name__)

    def __structures(self):
        """

        :return:
        """

        structures = src.models.structures.Structures(
            enumerator=self.__enumerator, variable=self.__variable,
            frames=self.__frames, tokenizer=self.__tokenizer)

        # The data
        training = structures.training()
        validating = structures.validating()

        return training, validating

    def exc(self):
        """

        :return:
        """

        training, validating = self.__structures()

        # Modelling
        architecture = src.models.distil.architecture.Architecture(
            variable=self.__variable, enumerator=self.__enumerator, archetype=self.__archetype)
        model = architecture(training=training, validating=validating, tokenizer=self.__tokenizer)
        self.__logger.info(dir(model))

        # Evaluating: vis-à-vis best model
        originals, predictions = src.models.distil.validation.Validation(
            validating=validating, archetype=self.__archetype).exc(model=model)
        self.__logger.info(originals)
        self.__logger.info(predictions)

        # Evaluation Metrics
        src.models.distil.measurements.Measurements().exc(
            originals=originals, predictions=predictions)
