"""Module main.py"""
import logging
import os
import sys

import pandas as pd
import torch


def main():
    """
    Entry point

    :return:
    """

    logger: logging.Logger = logging.getLogger(__name__)

    # Device Selection: Setting a graphics processing unit as the default device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(msg=device)

    # Ray
    # ray.init(dashboard_host='172.17.0.2', dashboard_port=8265)

    # The Data
    data: pd.DataFrame = src.data.source.Source().exc()
    logger.info(data.head())

    # Tags
    elements, enumerator, archetype = src.data.tags.Tags(data=data).exc()
    logger.info(elements)
    logger.info(enumerator)
    logger.info(archetype)

    # Balance/Imbalance
    data = data.copy().loc[data['category'].isin(values=elements['category'].unique()), :]

    # Sentences & Labels
    frame: pd.DataFrame = src.data.demarcations.Demarcations(data=data).exc()
    logger.info(frame.head())

    # Temporary
    frame = frame.loc[:4000, :]
    src.models.interface.Interface(frame=frame, enumerator=enumerator, archetype=archetype).exc()

    # Delete Cache Points
    src.functions.cache.Cache().exc()


if __name__ == '__main__':

    # Paths
    root = os.getcwd()
    sys.path.append(root)
    sys.path.append(os.path.join(root, 'src'))

    # Logging
    logging.basicConfig(level=logging.INFO,
                        format='\n\n%(message)s\n%(asctime)s.%(msecs)03d',
                        datefmt='%Y-%m-%d %H:%M:%S')

    # Activate graphics processing units
    os.environ['CUDA_VISIBLE_DEVICES']='0'
    os.environ['TOKENIZERS_PARALLELISM']='true'

    # Modules
    import src.data.source
    import src.data.tags
    import src.data.demarcations
    import src.functions.cache
    import src.models.interface

    main()
