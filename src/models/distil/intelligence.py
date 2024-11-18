"""Module intelligence.py"""
import transformers

import src.elements.arguments as ag


class Intelligence:
    """
    Intelligence
    """

    def __init__(self, enumerator: dict, archetype: dict, arguments: ag.Arguments):
        """

        :param enumerator: {label: id, ...}<br>
        :param archetype: {id: label, ...}<br>
        :param arguments: A suite of values/arguments for machine learning model development.<br>
        """

        self.__enumerator = enumerator
        self.__archetype = archetype

        # Parameters
        self.__arguments = arguments

    def model(self):
        """
        https://huggingface.co/docs/transformers/v4.41.3/en/model_doc/auto#transformers.AutoModel

        transformers.AutoModelForTokenClassification.from_pretrained(
            pretrained_model_name_or_path=self.__parameters.pretrained_model_name,
            config=config
        )

        :return:
        """

        config = transformers.DistilBertConfig(dropout=0.1, activation='gelu').from_pretrained(
            pretrained_model_name_or_path=self.__arguments.pretrained_model_name,
            **{'num_labels': len(self.__enumerator),
               'label2id': self.__enumerator,
               'id2label': self.__archetype})

        return transformers.DistilBertForTokenClassification.from_pretrained(
            pretrained_model_name_or_path=self.__arguments.pretrained_model_name,
            config=config
        )
