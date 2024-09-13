"""Module intelligence.py"""
import transformers

import src.models.distil.parameters as pr


class Intelligence:
    """
    Intelligence
    """

    def __init__(self, enumerator: dict):
        """

        :param enumerator:
        """

        self.__enumerator = enumerator

        # Parameters
        self.__parameters = pr.Parameters()

    @staticmethod
    def collator(tokenizer: transformers.PreTrainedTokenizerBase) -> transformers.DataCollatorForTokenClassification:
        """

        :param tokenizer:
        :return:
        """

        return transformers.DataCollatorForTokenClassification(tokenizer=tokenizer)

    def model(self):
        """
        https://huggingface.co/docs/transformers/v4.41.3/en/model_doc/auto#transformers.AutoModel

        :return:
        """

        return transformers.AutoModelForTokenClassification.from_pretrained(
            pretrained_model_name_or_path=self.__parameters.pretrained_model_name,
            **{'num_labels': len(self.__enumerator)})
