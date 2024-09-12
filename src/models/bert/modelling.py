"""Module modelling.py"""
import logging

import sklearn.metrics as sm
import torch
import torch.utils.data as tu
import tqdm.auto
import transformers
import transformers.modeling_outputs as tm

import src.elements.variable as vr
import src.models.bert.parameters


class Modelling:
    """
    BERT
    """

    def __init__(self, variable: vr.Variable,
                 enumerator: dict, dataloader: tu.DataLoader):
        """

        :param variable:
        :param enumerator: The labels enumerator
        :param dataloader:
        """

        self.__variable = variable
        self.__dataloader = dataloader
        self.__n_steps = self.__variable.EPOCHS * len(self.__dataloader)
        self.__progress = tqdm.auto.tqdm(iterable=range(self.__n_steps))

        # Parameters
        self.__parameters = src.models.bert.parameters.Parameters()

        # Model
        self.__model = transformers.BertForTokenClassification.from_pretrained(
            pretrained_model_name_or_path=self.__parameters.pretrained_model_name,
            **{'num_labels': len(enumerator)})
        self.__model.to(self.__parameters.device)

        # Optimizing & Scheduling
        self.__optimizer = torch.optim.AdamW(params=self.__model.parameters(), lr=self.__variable.LEARNING_RATE)
        self.__scheduler = transformers.get_scheduler(name='linear', optimizer=self.__optimizer,
                                                      num_warmup_steps=0, num_training_steps=self.__n_steps)

    def __train(self) -> transformers.modeling_utils.PreTrainedModel:
        """

        :return:
        """

        # Preparing for training ...
        self.__model.train()

        # Train
        for epoch in range(self.__variable.EPOCHS):

            logging.info('Epoch: %s', epoch)

            # For measures & metrics
            step_ = 0
            loss_ = 0
            accuracy_ = 0
            __labels: list[torch.Tensor] = []
            __predictions: list[torch.Tensor] = []

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
                loss.backward()

                # Targets, active targets.
                targets = labels_.view(-1)
                active = labels_.view(-1).ne(-100)
                original = torch.masked_select(targets, active)
                __labels.extend(original)

                # Predictions, of active targets
                logits = bucket.logits.view(-1, self.__model.config.num_labels)
                maxima = torch.argmax(logits, dim=1)
                predictions = torch.masked_select(maxima, active)
                __predictions.extend(predictions)

                # Accuracy
                # Replace this metric; inappropriate, and probably incorrect arithmetic.
                score: float = sm.accuracy_score(original.cpu().numpy(), predictions.cpu().numpy())
                accuracy_ += score

                # Gradient: Ambiguous
                # torch.nn.utils.clip_grad_norm(parameters=self.__model.parameters(),
                #                               max_norm=self.__variable.MAX_GRADIENT_NORM)
                self.__optimizer.step()
                self.__scheduler.step()
                self.__optimizer.zero_grad()
                self.__progress.update(1)

            logging.info('EPOCH %s -> Rough loss: %s', epoch, loss_/step_)
            logging.info('EPOCH %s -> Rough accuracy: %s', epoch, accuracy_/step_)

        return self.__model


    def exc(self) -> transformers.modeling_utils.PreTrainedModel:

        return self.__train()
