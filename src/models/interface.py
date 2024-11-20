"""Module interface.py"""
import logging
import os

import pandas as pd

import config
import src.elements.arguments as ag
import src.elements.hyperspace as hp
import src.elements.vault
import src.functions.directories
import src.functions.streams
import src.models.bert.steps
import src.models.distil.steps
import src.models.splittings


class Interface:
    """
    An interface to each model.
    """

    def __init__(self, data: pd.DataFrame, enumerator: dict, archetype: dict):
        """

        :param data: The data set for the training, validating, etc., stages
        :param enumerator: The tags and their identification codes.
        :param archetype: The inverse dict of enumerator.
        """

        self.__training, self.__validating, self.__testing = src.models.splittings.Splittings().exc(data=data)
        self.__vault = src.elements.vault.Vault(
            training=self.__training, validating=self.__validating, testing=self.__testing)

        # Objects in relation to tagging
        self.__enumerator = enumerator
        self.__archetype = archetype

        # Configurations
        self.__configurations = config.Config()

    def __store(self, architecture: str):
        """

        :param architecture:
        :return:
        """

        path = os.path.join(self.__configurations.artefacts_, architecture, 'data')
        directories = src.functions.directories.Directories()
        directories.create(path=path)

        streams = src.functions.streams.Streams()
        streams.write(blob=self.__training, path=os.path.join(path, 'training.csv'))
        streams.write(blob=self.__validating, path=os.path.join(path, 'validating.csv'))
        streams.write(blob=self.__testing, path=os.path.join(path, 'testing.csv'))

    def exc(self, architecture: str, arguments: ag.Arguments, hyperspace: hp.Hyperspace) -> None:
        """

        :param architecture:
        :param arguments:
        :param hyperspace:
        :return:
        """

        # Store the training, validating, and testing
        self.__store(architecture=architecture)

        # Hence
        match architecture:
            case 'bert':
                src.models.bert.steps.Steps(
                    enumerator=self.__enumerator, archetype=self.__archetype,
                    arguments=arguments, hyperspace=hyperspace, vault=self.__vault)
            case 'electra':
                logging.info('ELECTRA: Future')
            case 'roberta':
                logging.info('ROBERTA: Future')
            case 'distil':
                src.models.distil.steps.Steps(
                    enumerator=self.__enumerator, archetype=self.__archetype,
                    arguments=arguments, hyperspace=hyperspace, vault=self.__vault).exc()
            case 'ensemble':
                logging.info('BiLSTM, BERT, & CRF: Future\nhttps://link.springer.com/article/10.1007/s42979-024-02835-z')
            case _:
                logging.info('Unknown architecture')
