"""Module tokenizer"""
import transformers

import src.models.distil.parameters as pr

class Tokenizer:

    def __init__(self):
        """
        Constructor
        """

        self.__parameters = pr.Parameters()

    def __call__(self):
        """

        :return:
        """


        # Tokenizer
        return transformers.DistilBertTokenizerFast.from_pretrained(
            pretrained_model_name_or_path=self.__parameters.pretrained_model_name,
            clean_up_tokenization_spaces=True)
