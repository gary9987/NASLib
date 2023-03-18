import logging
import os
import sys
from argparse import ArgumentParser

import keras
import tensorflow as tf
import numpy as np

from naslib.search_spaces.core.query_metrics import Metric
from naslib.optimizers import (
    RandomSearch,
    RegularizedEvolution,
    LocalSearch,
    Bananas,
    BasePredictor,
    HB,
    Npenas
)

from naslib.search_spaces import (
    NasBench101SearchSpace,
    NasBench301SearchSpace,
    SimpleCellSearchSpace,
    NasBench201SearchSpace,
    HierarchicalSearchSpace
)

# from naslib.search_spaces.nasbench101 import graph
from naslib import utils
from naslib.search_spaces.nasbench101.graph import MODEL_TYPE
from naslib.search_spaces.nasbench201.conversions import convert_naslib_to_str
from naslib.utils import setup_logger, get_project_root
import pickle
from xgboost import XGBRegressor


if __name__ == '__main__':
    weights = ['nb201_model_bpr_train900_valid100', 'nb201_model_mae_train900_valid100']

    # Read args and config, setup logger
    config = utils.get_config_from_args()
    logger = setup_logger(config.save + "/log.log")
    utils.log_args(config)

    if config.optimizer_fidelity == 'single':
        from naslib.defaults.trainer import Trainer
        algo_list = ['rs', 're', 'ls', 'bananas', 'npnas', 'bp']
    elif config.optimizer_fidelity == 'multi':
        # If using HB:
        from naslib.defaults.trainer_x11 import Trainer
        algo_list = ['hb']
    else:
        raise ValueError('optimizer_fidelity is not supported')

    result_dict = {}

    for seed in range(11, 21):
        for algo in algo_list:
            for weight_path in weights:
                utils.set_seed(seed)

                supported_optimizers = {
                    "rs": RandomSearch(config),
                    "re": RegularizedEvolution(config),
                    "ls": LocalSearch(config),
                    "bananas": Bananas(config),
                    "bp": BasePredictor(config),
                    "hb": HB(config),
                    "npnas": Npenas(config)
                }

                if config.search_space == 'nasbench101':
                    search_space = NasBench101SearchSpace(model_type=MODEL_TYPE.KERAS)
                    dataset_api = utils.get_dataset_api('nasbench101', dataset='cifar10')
                elif config.search_space == 'nasbench201':
                    search_space = NasBench201SearchSpace(model_type=MODEL_TYPE.KERAS)
                    dataset_api = utils.get_dataset_api('nasbench201', dataset='cifar10')
                else:
                    raise ValueError('search space is not supported')

                dataset_api['surrogate'] = keras.models.load_model(weight_path, custom_objects={'bpr_loss': None})
                '''
                # lgbm
                weight_path = 'lgb_size80500_0'
                with open(weight_path, 'rb') as f:
                    dataset_api['surrogate'] = pickle.load(f)
                '''

                '''
                # xgb
                weight_path = 'xgb_size120500_0'
                model = XGBRegressor()
                model.load_model(weight_path)
                dataset_api['surrogate'] = model
                '''

                # Changing the optimizer is one line of code
                # optimizer = supported_optimizers[config.optimizer]
                optimizer = supported_optimizers[algo]
                optimizer.adapt_search_space(search_space, dataset_api=dataset_api)

                # Start the search and evaluation
                trainer = Trainer(optimizer, config)
                trainer.search()

                #
                # trainer.evaluate(dataset_api=dataset_api, metric=Metric.VAL_ACCURACY)
                # For trainer x11
                # trainer.evaluate(dataset_api=dataset_api)

                if config.search_space == 'nasbench101':
                    if config.optimizer_fidelity == 'multi':
                        best_arch_hash = trainer.optimizer.get_final_architecture()[0].get_hash()
                    elif config.optimizer_fidelity == 'single':
                        best_arch_hash = trainer.optimizer.get_final_architecture().get_hash()

                    model_spec = dataset_api['api'].ModelSpec(matrix=np.array(best_arch_hash[: 7*7]).reshape((7, 7)),
                                                              ops=list(best_arch_hash[7*7: ]))
                    logger.info(f'Result for weight: {weight_path} algo:{algo}')

                    for i in range(3):
                        data = dataset_api['nb101_data'].query(model_spec, query_idx=i)
                        if result_dict.get(f'{weight_path}_{algo}') is None:
                            result_dict[f'{weight_path}_{algo}'] = []
                        result_dict[f'{weight_path}_{algo}'].append(data['test_accuracy'])
                        logger.info(f'index {i} test_accuracy {data["test_accuracy"]}')

                elif config.search_space == 'nasbench201':
                    if config.optimizer_fidelity == 'multi':
                        best_arch, best_arch_epoch = trainer.optimizer.get_final_architecture()
                        logger.info(f'best arch epoch {best_arch_epoch}')
                    elif config.optimizer_fidelity == 'single':
                        best_arch = trainer.optimizer.get_final_architecture()

                    arch_str = convert_naslib_to_str(best_arch)
                    logger.info(f'best arch_str {arch_str}')
                    idx = dataset_api['api'].query_index_by_arch(arch_str)
                    for seed in [777, 888]:
                        data = dataset_api['api'].get_more_info(idx, 'cifar10-valid', iepoch=199, hp='200', is_random=seed)
                        if result_dict.get(f'{weight_path}_{algo}') is None:
                            result_dict[f'{weight_path}_{algo}'] = []
                        result_dict[f'{weight_path}_{algo}'].append(data['valid-accuracy'])
                        logger.info(f'index {seed} test_accuracy {data["valid-accuracy"]}')

        for key, item in result_dict.items():
            logger.info(f'{key}, mean test acc: {sum(item)/len(item)}, std: {np.std(item)}')