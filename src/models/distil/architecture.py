import logging

import transformers

import config
import src.elements.structures as sr
import src.elements.variable as vr
import src.models.distil.intelligence
import src.models.distil.metrics
import src.models.distil.parameters as pr
import src.models.distil.settings
import src.models.distil.storage


class Architecture:

    def __init__(self, variable: vr.Variable, enumerator: dict, archetype: dict):
        """

        :param variable:
        :param enumerator:
        :param archetype:
        """

        self.__variable = variable
        self.__enumerator = enumerator
        self.__archetype = archetype

        # Parameters
        self.__parameters = pr.Parameters()

        # Settings
        self.__settings = src.models.distil.settings.Settings(variable=self.__variable)

        # Directory preparation
        src.models.distil.storage.Storage().exc(path=self.__parameters.path)

        # https://huggingface.co/docs/transformers/v4.41.3/en/model_doc/auto#transformers.AutoModel
        self.__model = transformers.AutoModelForTokenClassification.from_pretrained(
            pretrained_model_name_or_path=self.__parameters.pretrained_model_name,
            **{'num_labels': len(self.__enumerator), 'id2label': self.__enumerator, 'label2id': self.__archetype})
        self.__model.to(self.__parameters.device)

    def __args(self):
        """
        https://huggingface.co/docs/transformers/v4.41.3/en/main_classes/trainer#transformers.TrainingArguments

        :return:
        """

        return transformers.TrainingArguments(
            output_dir=self.__parameters.path,
            eval_strategy='epoch',
            save_strategy='epoch',
            learning_rate=self.__variable.LEARNING_RATE,
            weight_decay=self.__variable.WEIGHT_DECAY,
            per_device_train_batch_size=self.__variable.TRAIN_BATCH_SIZE,
            per_device_eval_batch_size=self.__variable.VALID_BATCH_SIZE,
            num_train_epochs=self.__variable.EPOCHS,
            max_steps=-1,
            warmup_steps=0,
            no_cuda=False,
            seed=config.Config().seed,
            skip_memory_metrics=True,
            load_best_model_at_end=True,
            fp16=True,
            push_to_hub=False
            )

    def __call__(self, training: sr.Structures, validating: sr.Structures,
                 tokenizer: transformers.PreTrainedTokenizerBase):
        """
        https://huggingface.co/docs/transformers/v4.41.3/en/main_classes/trainer#transformers.Trainer

        :return:
        """

        # Collator
        intelligence = src.models.distil.intelligence.Intelligence(enumerator=self.__enumerator)
        data_collator = intelligence.collator(tokenizer=tokenizer)

        # Metrics
        metrics = src.models.distil.metrics.Metrics(archetype=self.__archetype)

        # Hence
        trainer = transformers.Trainer(
            model=self.__model,
            args=self.__args(),
            data_collator=data_collator,
            train_dataset=training.dataset,
            eval_dataset=validating.dataset,
            tokenizer=tokenizer,
            compute_metrics=metrics.exc)

        trainer.train()

        return trainer
