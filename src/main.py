"""Module main.py"""
import argparse
import logging
import os
import sys

import boto3
import pandas as pd
import ray
import torch


def main():
    """
    Entry Point

    :return:
    """

    logger: logging.Logger = logging.getLogger(__name__)

    # Set up
    setup: bool = src.setup.Setup(service=service, s3_parameters=s3_parameters, architecture=architecture).exc()
    if not setup:
        src.functions.cache.Cache().exc()
        sys.exit('No Executions')

    # Device Selection: Setting a graphics processing unit as the default device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info('Device: %s', device)

    # Ray
    ray.init(dashboard_host='172.17.0.2', dashboard_port=8265)

    # The Data
    interface = src.data.interface.Interface(s3_parameters=s3_parameters)
    data: pd.DataFrame = interface.data()

    # Temporary, min(arguments.N_INSTANCES, data.shape[0])
    data.info()
    data = data if arguments.N_INSTANCES is None else data.loc[:min(arguments.N_INSTANCES, data.shape[0]), :]
    data.info()

    # Hence
    src.models.interface.Interface(
        data=data, enumerator=interface.enumerator(), archetype=interface.archetype()).exc(
        architecture=architecture, arguments=arguments, hyperspace=hyperspace)

    # Transfer
    src.data.transfer.Transfer(
        service=service, s3_parameters=s3_parameters, architecture=architecture).exc()

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
    os.environ['RAY_USAGE_STATS_ENABLED']='0'
    os.environ['HF_HOME']='/tmp'

    # Modules
    import src.data.interface
    import src.data.transfer
    import src.elements.arguments

    import src.functions.cache
    import src.functions.expecting
    import src.functions.service
    import src.models.interface

    import src.s3.s3_parameters
    import src.settings.arguments
    import src.settings.hyperspace
    import src.setup

    expecting = src.functions.expecting.Expecting()
    parser = argparse.ArgumentParser()
    parser.add_argument('--architecture', type=expecting.architecture,
                        help='The name of the architecture in focus.')
    args = parser.parse_args()

    # Default architecture?
    architecture = 'distil' if args.architecture is None else args.architecture

    # S3 S3Parameters, Service Instance
    connector = boto3.session.Session()
    s3_parameters = src.s3.s3_parameters.S3Parameters(connector=connector).exc()
    service = src.functions.service.Service(connector=connector, region_name=s3_parameters.region_name).exc()

    arguments = src.settings.arguments.Arguments(s3_parameters=s3_parameters).exc(
        node=f'{architecture}/arguments.json')

    hyperspace = src.settings.hyperspace.Hyperspace(service=service, s3_parameters=s3_parameters).exc(
        node=f'{architecture}/hyperspace.json'
    )

    main()
