"""Module prerequisites.py"""
import datasets
import transformers

import src.elements.arguments as ag
import src.models.algorithm
import src.models.metrics
import src.models.training_arguments


class Prerequisites:
    """
    Prerequisites
    """

    def __init__(self, arguments: ag.Arguments, enumerator: dict, archetype: dict):
        """

        :param arguments:
        :param enumerator:
        :param archetype:
        """

        self.__arguments = arguments
        self.__enumerator = enumerator
        self.__archetype = archetype

        # Intelligence
        self.__algorithm = src.models.algorithm.Algorithm(architecture=self.__arguments.architecture)

    def __call__(self, training: datasets.Dataset, validating: datasets.Dataset,
                 tokenizer: transformers.tokenization_utils_base.PreTrainedTokenizerBase) -> transformers.Trainer:
        """

        :param training:
        :param validating:
        :param tokenizer:
        :return:
        """

        # Training Arguments
        args = src.models.training_arguments.TrainingArguments(arguments=self.__arguments).exc()

        # Model
        algorithm = self.__algorithm.exc(
            arguments=self.__arguments, enumerator=self.__enumerator, archetype=self.__archetype)

        # Metrics
        metrics = src.models.metrics.Metrics(archetype=self.__archetype)

        # Data Collator
        data_collator: transformers.DataCollatorForTokenClassification = (
            transformers.DataCollatorForTokenClassification(tokenizer=tokenizer))

        # Hence
        trainer = transformers.Trainer(
            model_init=algorithm.model,
            args=args,
            data_collator=data_collator,
            train_dataset=training,
            eval_dataset=validating,
            tokenizer=tokenizer,
            compute_metrics=metrics.exc
        )

        return trainer
