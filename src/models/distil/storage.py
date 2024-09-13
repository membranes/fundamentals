"""Module storage.py"""
import src.functions.directories

class Storage:
    """
    Class Storage
    """

    def __init__(self):
        """
        Constructor
        """

        self.__directories = src.functions.directories.Directories()

    def exc(self, path: str):
        """

        :param path:
        :return:
        """

        self.__directories.cleanup(path=path)
