"""Module dataset.py"""
import numpy as np
import pandas as pd
import torch
import torch.utils.data
import transformers

import src.elements.arguments as ag


class Dataset(torch.utils.data.Dataset):
    """
    Dataset builder, vis-à-vis the tokenization method of the architecture in question.
    """

    def __init__(self, frame: pd.DataFrame, arguments: ag.Arguments,
                 enumerator: dict, tokenizer: transformers.tokenization_utils_base) -> None:
        """
        Parameters<br>
        -----------<br>
        :param frame: The data object within which the data being tokenized resides.<br>
        :param arguments: A suite of values/arguments for machine learning model development.<br>
        :param enumerator: Of tags; key &rarr; identifier, value &rarr; label<br>
        :param tokenizer: The tokenizer of text.<br>
        """

        super().__init__()

        self.__frame = frame
        self.__length = len(self.__frame)

        self.__arguments = arguments
        self.__enumerator = enumerator
        self.__tokenizer = tokenizer

    def __getitem__(self, index) -> dict:
        """

        :param index: A row index
        :return:
        """

        # A sentence's words, and the tokenization of words
        words: list[str] = self.__frame['sentence'][index].split()
        encoding: dict = self.__tokenizer(words, padding='max_length', truncation=True,
                                          is_split_into_words=True,
                                          max_length=self.__arguments.MAX_LENGTH,
                                          return_offsets_mapping=True)

        # placeholder array of labels for the encoding dict
        ela: np.ndarray = np.ones(shape=self.__arguments.MAX_LENGTH, dtype=int) * -100

        # The corresponding tags of a sentence's words, and the code of each tag
        tags: list[str] = self.__frame['tagstr'][index].split(',')
        labels = [self.__enumerator[tag] for tag in tags]

        # Herein, per word index cf. offset pairings.  There are <max_length> tokens.
        # (maximum number of tokens, 2)
        limit = len(labels)
        for iteration, mapping in enumerate(encoding['offset_mapping']):
            if mapping[0] == 0 and mapping[1] != 0 and iteration < limit:
                ela[iteration] = labels[iteration]

        encoding['labels'] = ela
        item = {key: torch.as_tensor(value) for key, value in encoding.items()}

        return item

    def __len__(self):
        """

        :return:
        """

        return self.__length
