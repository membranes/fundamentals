"""Module validation.py"""
import logging
import typing

import sklearn.metrics as sm
import torch
import torch.utils.data as tu
import transformers
import transformers.modeling_outputs as tm

import src.models.bert.parameters


class Validation:
    """
    For validation stage calculations
    """

    def __init__(self, model: transformers.PreTrainedModel, archetype: dict, dataloader: tu.DataLoader):
        """

        :param model: The trained model
        :param archetype: The identifiers to label mappings
        :param dataloader: The validation data DataLoader
        """

        # Model, DataLoader, Tag Mappings
        self.__model = model
        self.__archetype = archetype
        self.__dataloader = dataloader

        # Parameters
        self.__parameters = src.models.bert.parameters.Parameters()

    def __validating(self) -> typing.Tuple[list, list]:
        """

        :return:
        """

        # Preparing for validation stage ...
        self.__model.eval()

        # For measures & metrics
        step_ = 0
        loss_ = 0
        accuracy_ = 0
        __originals: list[torch.Tensor] = []
        __predictions: list[torch.Tensor] = []

        with torch.no_grad():

            # By batch
            index: int
            batch: dict
            for index, batch in enumerate(self.__dataloader):

                step_ += 1

                # Parts of the dataset
                inputs_: torch.Tensor = batch['input_ids'].to(self.__parameters.device, dtype = torch.long)
                labels_: torch.Tensor = batch['labels'].to(self.__parameters.device, dtype = torch.long)
                attention_mask_: torch.Tensor = batch['attention_mask'].to(self.__parameters.device, dtype = torch.long)

                # https://huggingface.co/docs/transformers/main_classes/output#transformers.modeling_outputs.TokenClassifierOutput
                bucket: tm.TokenClassifierOutput = self.__model(input_ids=inputs_, attention_mask=attention_mask_, labels=labels_)

                # Loss
                loss = bucket.loss
                loss_ += loss.item()

                # Targets, active targets.
                targets = labels_.view(-1)
                active = labels_.view(-1).ne(-100)
                original = torch.masked_select(targets, active)
                __originals.extend(original)

                # Predictions, of active targets
                logits = bucket.logits.view(-1, self.__model.config.num_labels)
                maxima = torch.argmax(logits, dim=1)
                prediction = torch.masked_select(maxima, active)
                __predictions.extend(prediction)

                # Accuracy
                # Replace this metric; inappropriate, and probably incorrect arithmetic.
                score: float = sm.accuracy_score(original.cpu().numpy(), prediction.cpu().numpy())
                accuracy_ += score

        logging.info(loss_/step_)
        logging.info(accuracy_/step_)

        originals_ = [self.__archetype[code.item()] for code in __originals]
        predictions_ = [self.__archetype[code.item()] for code in __predictions]

        return originals_, predictions_

    def exc(self) -> typing.Tuple[list, list]:
        """

        :return:
        """

        return self.__validating()
