"""Module settings"""

import ray
import ray.tune
import ray.tune.schedulers as rts

import src.elements.variable as vr


class Settings:
    """
    Class Settings
    """

    def __init__(self, variable: vr.Variable):
        """

        :param variable: A suite of values for machine learning
                         model development
        """

        self.__variable = variable

        # Re-visit
        self.__perturbation_interval = 2

    def hp_space(self):
        """
        Initialises

        :return:
        """

        return {
            'learning_rate': self.__variable.LEARNING_RATE,
            'weight_decay': self.__variable.WEIGHT_DECAY,
            'per_device_train_batch_size': self.__variable.TRAIN_BATCH_SIZE
        }

    def scheduler(self):
        """
        https://docs.ray.io/en/latest/tune/api/doc/ray.tune.schedulers.PopulationBasedTraining.html

        Leads on from hp_space

        :return:
        """

        return rts.PopulationBasedTraining(
            time_attr='training_iteration',
            metric='eval_loss', mode='min',
            perturbation_interval=self.__perturbation_interval,
            hyperparam_mutations={
                'learning_rate': ray.tune.uniform(lower=0.002, upper=0.1),
                'weight_decay': ray.tune.uniform(lower=0.02, upper=0.2),
                'per_device_train_batch_size': [32, 64]
            },
            quantile_fraction=0.25,
            resample_probability=0.25
        )

    @staticmethod
    def reporting():
        """
        https://docs.ray.io/en/latest/tune/api/doc/ray.tune.CLIReporter.html

        :return:
        """

        return ray.tune.CLIReporter(
            parameter_columns=['learning_rate', 'weight_decay', 'per_device_training_batch_size'],
            metric_columns=['eval_loss', 'precision', 'recall', 'f1']
        )
