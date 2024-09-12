"""Module parameters.py"""
import torch
import transformers


class Parameters:
    """

    [from_pretrained](https://huggingface.co/docs/transformers/main_classes/model#transformers.PreTrainedModel.from_pretrained)
    [PretrainedConfig](https://huggingface.co/docs/transformers/main_classes/configuration#transformers.PretrainedConfig)
    """

    def __init__(self):
        """
        Constructor
        """

        # The device for computation
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Pretrained model
        self.pretrained_model_name: str = 'google-bert/bert-base-uncased'

        # Tokenizer
        self.tokenizer = transformers.BertTokenizerFast.from_pretrained(
            pretrained_model_name_or_path=self.pretrained_model_name)
