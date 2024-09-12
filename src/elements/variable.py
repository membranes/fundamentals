import typing

class Variable(typing.NamedTuple):
    """

    Attributes
    ----------

    MAX_LENGTH : int
        The maximum number of tokens

    TRAIN_BATCH_SIZE : int
        The batch size for the training stage

    VALID_BATCH_SIZE : int
        The batch size for the validation stage

    TEST_BATCH_SIZE : int
        The batch size for the testing stage

    EPOCHS : int
        The number of epochs

    LEARNING_RATE : float
        The learning rate

    WEIGHT_DECAY : float
        Weight decay

    MAX_GRADIENT_NORM : int
        The maximum gradient norm
    """

    MAX_LENGTH: int = 128
    TRAIN_BATCH_SIZE: int = 4
    VALID_BATCH_SIZE: int = 2
    TEST_BATCH_SIZE: int = 2
    EPOCHS: int = 1
    LEARNING_RATE: float = 0.001
    WEIGHT_DECAY: float = 0.01
    MAX_GRADIENT_NORM: int = 10
