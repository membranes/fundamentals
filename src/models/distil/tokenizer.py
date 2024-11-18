"""Module tokenizer"""
import transformers

import src.elements.arguments as ag


class Tokenizer:
    """
    Class Tokenizer: <a href="https://arxiv.org/abs/1910.01108" target="_blank">Distil BERT</a>
     (Bidirectional Encoder Representations from Transformers)
    """

    def __init__(self, arguments: ag.Arguments):
        """

        :param arguments: A suite of values/arguments for machine learning model development.<br>
        """

        self.__arguments = arguments

    def __call__(self) -> transformers.tokenization_utils_base.PreTrainedTokenizerBase:
        """

        :return:
        """

        # Tokenizer
        return transformers.DistilBertTokenizerFast.from_pretrained(
            pretrained_model_name_or_path=self.__arguments.pretrained_model_name,
            clean_up_tokenization_spaces=True)
