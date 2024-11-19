"""Module steps.py"""
import logging
import os
import typing

import datasets
import transformers

import src.elements.arguments as ag
import src.elements.hyperspace as hp
import src.elements.vault as vu
import src.models.bert.structures
import src.models.bert.tokenizer
import src.models.hyperpoints
import src.models.measurements
import src.models.prime
import src.models.validation


class Steps:
    """
    The BERT steps.
    """

    def __init__(self, enumerator: dict, archetype: dict, arguments: ag.Arguments,
                 hyperspace: hp.Hyperspace, vault: vu.Vault):
        """

        :param enumerator: Of tags; key &rarr; identifier, value &rarr; label<br>
        :param archetype: Of tags; key &rarr; label, value &rarr; identifier<br>
        :param arguments: A suite of values/arguments for machine learning model development<br>
        :param hyperspace: The hyperparameters alongside their starting values or number spaces.<br>
        :param vault: The data frames for modelling stages, i.e., the training, validating, and testing stages.<br>
        """

        # Inputs
        self.__enumerator = enumerator
        self.__archetype = archetype
        self.__arguments = arguments
        self.__hyperspace = hyperspace
        self.__vault = vault

        # A set of values for machine learning model development
        self.__arguments = self.__arguments._replace(
            N_TRAIN=self.__vault.training.shape[0], N_VALID=self.__vault.validating.shape[0],
            N_TEST=self.__vault.testing.shape[0])

        # Get tokenizer
        self.__tokenizer: transformers.tokenization_utils_base.PreTrainedTokenizerBase = (
            src.models.bert.tokenizer.Tokenizer(arguments=self.__arguments)())

        # Storage Section
        self.__section = self.__arguments.model_output_directory

    def __structures(self) -> typing.Tuple[datasets.Dataset, datasets.Dataset, datasets.Dataset]:
        """
        structures.training(), structures.validating(), structures.testing()

        :return:
        """

        structures = src.models.bert.structures.Structures(
            enumerator=self.__enumerator, arguments=self.__arguments,
            vault=self.__vault, tokenizer=self.__tokenizer)

        return structures.training(), structures.validating(), structures.testing()

    def exc(self):
        """

        :return:
        """

        training, validating, _ = self.__structures()

        # The path for hyperparameter artefacts
        self.__arguments = self.__arguments._replace(
            model_output_directory=os.path.join(self.__section, 'hyperparameters'))

        # Determining the optimal hyperparameters
        hyperpoints = src.models.hyperpoints.Hyperpoints(
            arguments=self.__arguments, hyperspace=self.__hyperspace,
            enumerator=self.__enumerator, archetype=self.__archetype)
        best = hyperpoints(training=training, validating=validating, tokenizer=self.__tokenizer)
        logging.info(best)

        # Hence, update the modelling variables
        self.__arguments = self.__arguments._replace(
            LEARNING_RATE=best.hyperparameters.get('learning_rate'),
            WEIGHT_DECAY=best.hyperparameters.get('weight_decay'))

        # Additionally, prepare the artefacts storage area for the best model, vis-à-vis best hyperparameters set.
        self.__arguments = self.__arguments._replace(
            model_output_directory=os.path.join(self.__section, 'prime'),
            EPOCHS=2*self.__arguments.EPOCHS, save_total_limit=1)

        # The prime model
        model = src.models.prime.Prime(
            enumerator=self.__enumerator, archetype=self.__archetype, arguments=self.__arguments).exc(
            training=training, validating=validating, tokenizer=self.__tokenizer)

        # Evaluating: vis-à-vis model & validation data
        originals, predictions = src.models.validation.Validation(
            validating=validating, archetype=self.__archetype).exc(model=model)

        src.models.measurements.Measurements(
            originals=originals, predictions=predictions, arguments=self.__arguments).exc(segment='prime')
