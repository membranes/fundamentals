"""Module dataset.py"""
import pandas as pd
import torch.utils.data
import transformers


class Dataset(torch.utils.data.Dataset):
    """
    Dataset builder, vis-à-vis the tokenization method of the architecture in question.
    """

    def __init__(self, frame: pd.DataFrame, enumerator: dict,
                 tokenizer: transformers.tokenization_utils_base.PreTrainedTokenizerBase):
        """
        Parameters<br>
        -----------<br>
        :param frame: The data object within which the data being tokenized resides.<br>
        :param enumerator: Of tags; key &rarr; identifier, value &rarr; label<br>
        :param tokenizer: The tokenizer of text.<br>
        """

        super().__init__()

        self.__frame = frame
        self.__length = self.__frame.shape[0]

        self.__enumerator = enumerator
        self.__tokenizer = tokenizer

    def __getitem__(self, index):
        """

        :param index: A row index
        :return:
        """

        words: list[str] = self.__frame['sentence'][index].split()
        encodings = self.__tokenizer(
            words, truncation=True, is_split_into_words=True, padding='max_length', max_length=self.__length)

        tags: list[str] = self.__frame['tagstr'][index].split(',')
        numerals: list[int] = [self.__enumerator[tag] for tag in tags]

        identifiers = encodings.word_ids()

        previous = None
        register = []
        for identifier in identifiers:
            if identifier is None:
                register.append(-100)
            elif identifier != previous:
                register.append(numerals[identifier])
            else:
                register.append(-100)
            previous = identifier

        encodings['labels'] = register

        item = {key: torch.as_tensor(value) for key, value in encodings.items()}

        return item

    def __len__(self):
        """

        :return:
        """

        return self.__length
