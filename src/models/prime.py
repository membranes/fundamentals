"""Module prime.py"""
import datasets
import transformers

import src.elements.arguments as ag
import src.models.algorithm
import src.models.metrics
import src.models.training_arguments
import src.models.prerequisites


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

        trainer = src.models.prerequisites.Prerequisites(
            arguments=self.__arguments, enumerator=self.__enumerator, archetype=self.__archetype)(
            training=training, validating=validating, tokenizer=tokenizer)

        trainer.train()

        return trainer
