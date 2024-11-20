"""Module hyperpoints.py"""
import os

import datasets
import transformers

import src.elements.arguments as ag
import src.elements.hyperspace as hp
import src.functions.directories


import src.models.training_arguments
import src.models.prerequisites
import src.models.tuning


class Hyperpoints:
    """
    Architecture
    """

    def __init__(self, arguments: ag.Arguments, hyperspace: hp.Hyperspace, enumerator: dict, archetype: dict):
        """

        :param arguments: A suite of values/arguments for machine learning model development.<br>
        :param hyperspace: The hyperparameters alongside their starting values or number spaces.<br>
        :param enumerator: Of tags; key &rarr; identifier, value &rarr; label<br>
        :param archetype: Of tags; key &rarr; label, value &rarr; identifier<br>
        """

        self.__arguments = arguments
        self.__hyperspace = hyperspace
        self.__enumerator = enumerator
        self.__archetype = archetype

        # Directory preparation
        src.functions.directories.Directories().cleanup(path=self.__arguments.model_output_directory)

    def __call__(self, training: datasets.Dataset, validating: datasets.Dataset,
                 tokenizer: transformers.tokenization_utils_base.PreTrainedTokenizerBase) -> (
            transformers.trainer_utils.BestRun):
        """
        https://huggingface.co/docs/transformers/v4.41.3/en/main_classes/trainer#transformers.Trainer
        https://docs.ray.io/en/latest/tune/api/doc/ray.tune.run.html

        :param training: The training data object.<br>
        :param validating: The validation data object.<br>
        :param tokenizer: The tokenizer of text.<br>
        :return:
        """

        # The transformers.Trainer
        trainer = src.models.prerequisites.Prerequisites(
            arguments=self.__arguments, enumerator=self.__enumerator, archetype=self.__archetype)(
            training=training, validating=validating, tokenizer=tokenizer)

        # Tuning
        tuning = src.models.tuning.Tuning(arguments=self.__arguments, hyperspace=self.__hyperspace)

        # Best
        best = trainer.hyperparameter_search(
            hp_space=tuning.hp_space,
            compute_objective=tuning.compute_objective,
            n_trials=self.__arguments.N_TRIALS,
            direction='minimize',
            backend='ray',

            # scaling configuration
            resources_per_trial={'cpu': self.__arguments.N_CPU, 'gpu': self.__arguments.N_GPU},

            # tune configuration
            search_alg=tuning.algorithm(),
            scheduler=tuning.scheduler(), reuse_actors=True,

            # check point configuration
            # keep_checkpoints_num=8, checkpoint_score_attr='training_iteration',

            # run configuration: local_dir -> storage_path
            name='default', storage_path=os.path.join(self.__arguments.model_output_directory, 'ray'),
            verbose=0, progress_reporter=tuning.reporting, log_to_file=True
        )

        return best
