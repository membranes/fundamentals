"""Module prime.py"""
import os

import datasets
import transformers

import src.elements.arguments as ag
import src.models.algorithm
import src.models.metrics
import src.models.training_arguments
import src.models.tunnel


class Prime:
    """
    Notes<br>
    ------<br>

    Determines the prime model vis-à-vis the best set of hyperparameters.
    """

    def __init__(self, enumerator: dict, archetype: dict, arguments: ag.Arguments):
        """

        :param enumerator: Of tags; key &rarr; identifier, value &rarr; label<br>
        :param archetype: Of tags; key &rarr; label, value &rarr; identifier<br>
        :param arguments: A suite of values/arguments for machine learning model development.<br>
        """

        self.__enumerator = enumerator
        self.__archetype = archetype
        self.__arguments = arguments

    def exc(self, training: datasets.Dataset, validating: datasets.Dataset,
            tokenizer: transformers.tokenization_utils_base.PreTrainedTokenizerBase):
        """

        :param training: The training data.
        :param validating: The validation data.
        :param tokenizer: The tokenizer of text.<br>
        :return:
        """

        # The transformers.Trainer
        tunnel = src.models.tunnel.Tunnel(arguments=self.__arguments, enumerator=self.__enumerator, archetype=self.__archetype)
        trainer = tunnel(training=training, validating=validating, tokenizer=tokenizer)

        model = trainer.train()

        model.save_model(output_dir=os.path.join(self.__arguments.model_output_directory, 'model'))

        return model
