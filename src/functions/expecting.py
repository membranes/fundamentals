"""Module expecting.py"""
import argparse

import config


class Expecting:
    """
    Parses the expected input arguments.
    """

    def __init__(self):
        """
        Constructor
        """

        self.__configurations = config.Config()

    def architecture(self, value) -> str:
        """

        :param value: The name of the architecture in focus.
        :return:
        """

        if str(value) not in self.__configurations.architectures:
            raise argparse.ArgumentTypeError(f"The string must be a member of {self.__configurations.architectures}")

        return str(value)
