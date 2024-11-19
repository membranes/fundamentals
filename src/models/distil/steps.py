"""Module steps.py"""
import logging
import os.path

import datasets
import transformers.tokenization_utils_base

import src.elements.arguments as ag
import src.elements.hyperspace as hp
import src.elements.vault as vu
import src.models.distil.tokenizer
import src.models.distil.yields
import src.models.hyperpoints
import src.models.prime
import src.models.validation
import src.models.measurements


class Steps:
    """
    Steps
    ref. https://huggingface.co/docs/transformers/tasks/token_classification
    """

    def __init__(self, enumerator: dict, archetype: dict, arguments: ag.Arguments, hyperspace: hp.Hyperspace, vault: vu.Vault):
        """

        :param enumerator: Of tags; key &rarr; identifier, value &rarr; label<br>
        :param archetype: Of tags; key &rarr; label, value &rarr; identifier<br>
        :param arguments: A suite of values/arguments for machine learning model development.<br>
        :param hyperspace: The hyperparameters alongside their starting values or number spaces.<br>
        :param vault: An object of dataframes, consisting of the training, validating, and testing data sets.<br>
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
            src.models.distil.tokenizer.Tokenizer(arguments=self.__arguments)())

        # Storage Section
        self.__section = self.__arguments.model_output_directory

    def exc(self):
        """

        :return:
        """

        yields: datasets.DatasetDict = src.models.distil.yields.Yields(
            vault=self.__vault, tokenizer=self.__tokenizer).exc()

        # The path for hyperparameter artefacts
        self.__arguments = self.__arguments._replace(
            model_output_directory=os.path.join(self.__section, 'hyperparameters'))

        # Determining the optimal hyperparameters
        hyperpoints = src.models.hyperpoints.Hyperpoints(
            arguments=self.__arguments, hyperspace=self.__hyperspace,
            enumerator=self.__enumerator, archetype=self.__archetype)
        best = hyperpoints(training=yields['training'], validating=yields['validating'], tokenizer=self.__tokenizer)
        logging.info(best)

        # Hence, update the modelling variables
        self.__arguments = self.__arguments._replace(
            LEARNING_RATE=best.hyperparameters.get('learning_rate'),
            WEIGHT_DECAY=best.hyperparameters.get('weight_decay'),
            TRAIN_BATCH_SIZE=best.hyperparameters.get('per_device_train_batch_size'))

        # Additionally, prepare the artefacts storage area for the best model, vis-à-vis best hyperparameters
        # set, and save a checkpoint at the optimal training point only by setting save_total_limit = 1.
        self.__arguments = self.__arguments._replace(
            model_output_directory=os.path.join(self.__section, 'prime'),
            EPOCHS=2*self.__arguments.EPOCHS, save_total_limit=1)

        # The prime model
        model = src.models.prime.Prime(
            enumerator=self.__enumerator, archetype=self.__archetype, arguments=self.__arguments).exc(
            training=yields['training'], validating=yields['validating'], tokenizer=self.__tokenizer)

        # Save
        model.save_model(output_dir=os.path.join(self.__arguments.model_output_directory, 'model'))

        # Evaluating: vis-à-vis model & validation data
        originals, predictions = src.models.validation.Validation(
            validating=yields['validating'], archetype=self.__archetype).exc(model=model)

        src.models.measurements.Measurements(
            originals=originals, predictions=predictions, arguments=self.__arguments).exc(segment='prime')
