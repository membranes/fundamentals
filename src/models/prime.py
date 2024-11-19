"""Module prime.py"""
import logging
import os

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

        # Training Arguments
        self.__args = src.models.training_arguments.TrainingArguments(arguments=self.__arguments).exc()

        # Intelligence
        self.__algorithm = src.models.algorithm.Algorithm(architecture=self.__arguments.architecture)

    def exc(self, training: datasets.Dataset, validating: datasets.Dataset,
            tokenizer: transformers.tokenization_utils_base.PreTrainedTokenizerBase):
        """

        :param training: The training data.
        :param validating: The validation data.
        :param tokenizer: The tokenizer of text.<br>
        :return:
        """

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
            args=self.__args,
            data_collator=data_collator,
            train_dataset=training,
            eval_dataset=validating,
            tokenizer=tokenizer,
            compute_metrics=metrics.exc
        )

        model = trainer.train()

        # Save
        model.save_model(output_dir=os.path.join(self.__arguments.model_output_directory, 'model'))

        return model
