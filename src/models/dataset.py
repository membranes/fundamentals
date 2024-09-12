import numpy as np
import pandas as pd
import torch
import torch.utils.data
import transformers

import src.elements.variable as vr


class Dataset(torch.utils.data.Dataset):

    def __init__(self, frame: pd.DataFrame, variable: vr.Variable,
                 enumerator: dict, tokenizer: transformers.tokenization_utils_base) -> None:
        """

        :param frame:
        :param variable:
        :param enumerator:
        """

        super().__init__()

        self.__frame = frame
        self.__length = len(self.__frame)

        self.__variable = variable
        self.__enumerator = enumerator
        self.__tokenizer = tokenizer
    
    def __getitem__(self, index) -> dict:
        """

        :param index: A row index
        :return:
        """

        # A sentence's words, and the tokenization of words
        words: list[str] = self.__frame['sentence'][index].strip().split()
        encoding: dict = self.__tokenizer(words, padding='max_length', truncation=True,
                                          is_split_into_words=True,
                                          max_length=self.__variable.MAX_LENGTH,
                                          return_offsets_mapping=True)

        # placeholder array of labels for the encoding dict
        ela: np.ndarray = np.ones(shape=self.__variable.MAX_LENGTH, dtype=int) * -100

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
