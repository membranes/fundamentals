"""config.py"""
import os


class Config:
    """
    Config
    """

    def __init__(self) -> None:
        """
        Constructor
        """

        self.datapath = os.path.join(os.getcwd(), 'data')
        self.warehouse = os.path.join(os.getcwd(), 'warehouse')

        # Addressing category imbalance ...
        self.minimum_category_frequency = 1000

        # Splitting fractions:
        # * 80% of the data for training
        # * A tenth of the remaining 20% for testing, i.e., 18%
        #   for validating, 2% for testing
        self.fraction = 0.8
        self.aside = 0.1

        # Seed: All cases
        self.seed = 5
