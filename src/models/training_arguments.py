"""Module training_arguments.py"""
import os

import transformers

import config
import src.elements.arguments as ag


class TrainingArguments:
    """
    Class Arguments
    """

    def __init__(self, arguments: ag.Arguments):
        """

        :param arguments: A suite of values/arguments for machine learning model development.<br>
        """

        self.__arguments = arguments

    def exc(self):
        """

        Notes<br>
        -----<br>
        output_dir: Is the model_output_directory<br>
        logging_dir: Is the storage_path, the
        <a href="https://huggingface.co/docs/transformers/v4.45.2/en/main_classes/
        trainer#transformers.TrainingArguments.logging_dir" target="_blank">TensorBoard logging
        directory</a>.<br><br>

        References<br>
        ----------<br>
        <a href="https://huggingface.co/docs/transformers/v4.41.3/en/main_classes/trainer#transformers.TrainingArguments"
         target="_blank">TrainingArguments</a><br><br>

        :return:
        """

        return transformers.TrainingArguments(
            output_dir=self.__arguments.model_output_directory,
            eval_strategy='epoch',
            save_strategy='epoch',
            learning_rate=self.__arguments.LEARNING_RATE,
            weight_decay=self.__arguments.WEIGHT_DECAY,
            per_device_train_batch_size=self.__arguments.TRAIN_BATCH_SIZE,
            per_device_eval_batch_size=self.__arguments.VALID_BATCH_SIZE,
            num_train_epochs=self.__arguments.EPOCHS,
            max_steps=-1,
            warmup_steps=0,
            no_cuda=False,
            seed=config.Config().seed,
            save_total_limit=self.__arguments.save_total_limit,
            skip_memory_metrics=True,
            metric_for_best_model='eval_loss',
            greater_is_better=False,
            load_best_model_at_end=True,
            logging_dir=os.path.join(self.__arguments.model_output_directory, 'logs'),
            fp16=True,
            push_to_hub=False)
