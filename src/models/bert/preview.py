import logging

import torch
import torch.utils.data as tu

import src.models.bert.parameters


class Preview:

    def __init__(self):
        """
        Constructor
        """

        self.__tokenizer = src.models.bert.parameters.Parameters().tokenizer

        # Logging
        logging.basicConfig(level=logging.INFO,
                            format='\n\n%(message)s\n%(asctime)s.%(msecs)03d',
                            datefmt='%Y-%m-%d %H:%M:%S')
        self.__logger = logging.getLogger(__name__)
        
    @staticmethod
    def __content(segment: dict):
        """

        :param segment:
        :return:
        """

        inputs_: torch.Tensor = segment['input_ids']
        labels_: torch.Tensor = segment['labels']
        token_type_identifiers_: torch.Tensor = segment['token_type_ids']
        attention_mask_: torch.Tensor = segment['attention_mask']
        offset_mapping_: torch.Tensor = segment['offset_mapping']
        
        return inputs_, labels_, token_type_identifiers_, attention_mask_, offset_mapping_

    def __details(self, name: str, item: torch.Tensor):
        """

        :param name:
        :param item:
        :return:
        """

        self.__logger.info('%s: %s', name, item.shape)
        self.__logger.info(item.data)

    def exc(self, dataset: tu.Dataset):
        """

        :param dataset:
        :return:
        """

        # A segment of dataset
        self.__logger.info(dataset.__doc__)
        self.__logger.info(type(dataset.__getitem__(0)))
        segment: dict = dataset.__getitem__(0)

        self.__logger.info('Previewing an instance of the data: ...')
        self.__logger.info(segment.keys())
        
        # The content of the segment
        inputs_, labels_, token_type_identifiers_, _, offset_mapping_ = self.__content(segment=segment)
        self.__details(name='inputs', item=inputs_)
        self.__details(name='labels', item=labels_)
        self.__details(name='token type identifiers', item=token_type_identifiers_)
        self.__details(name='offset mapping', item=offset_mapping_)
