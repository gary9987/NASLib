import logging
import sys
import keras
import tensorflow as tf
import numpy as np
# If using HB:
#   from naslib.defaults.trainer_x11 import Trainer
from naslib.defaults.trainer import Trainer
from naslib.search_spaces.core.query_metrics import Metric
from naslib.optimizers import (
    DARTSOptimizer,
    GDASOptimizer,
    DrNASOptimizer,
    RandomSearch,
    RegularizedEvolution,
    LocalSearch,
    Bananas,
    BasePredictor,
    HB
)

from naslib.search_spaces import (
    NasBench101SearchSpace,
    NasBench301SearchSpace,
    SimpleCellSearchSpace,
    NasBench201SearchSpace,
    HierarchicalSearchSpace,
)

# from naslib.search_spaces.nasbench101 import graph
from naslib import utils
from naslib.search_spaces.nasbench101.graph import MODEL_TYPE
from naslib.utils import setup_logger
import pickle
from xgboost import XGBRegressor


if __name__ == '__main__':
    algo_list = ['rs', 're', 'ls', 'bananas', 'bp']
    weights = ['gin25500_mse', 'gin25500_bpr']

    # Read args and config, setup logger
    config = utils.get_config_from_args()
    logger = setup_logger(config.save + "/log.log")
    utils.log_args(config)

    result_dict = {}
    for algo in algo_list:
        for weight_path in weights:
            utils.set_seed(config.seed)

            supported_optimizers = {
                "darts": DARTSOptimizer(config),
                "gdas": GDASOptimizer(config),
                "drnas": DrNASOptimizer(config),
                "rs": RandomSearch(config),
                "re": RegularizedEvolution(config),
                "ls": LocalSearch(config),
                "bananas": Bananas(config),
                "bp": BasePredictor(config),
                "hb": HB(config)
            }

            # search_space = NasBench201SearchSpace(model_type=MODEL_TYPE.KERAS)
            search_space = NasBench101SearchSpace(model_type=MODEL_TYPE.KERAS)

            dataset_api = utils.get_dataset_api('nasbench101', dataset='cifar10')

            '''
            # keras 201
            weight_path = 'nb201_gin_model'
            dataset_api['surrogate'] = keras.models.load_model(weight_path)
            '''

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

            trainer.evaluate(dataset_api=dataset_api, metric=Metric.VAL_ACCURACY)

            best_arch_hash = trainer.optimizer.get_final_architecture().get_hash()
            model_spec = dataset_api['api'].ModelSpec(matrix=np.array(best_arch_hash[: 7*7]).reshape((7, 7)),
                                                      ops=list(best_arch_hash[7*7: ]))
            logger.info(f'Result for weight: {weight_path} algo:{algo}')
            test_acc_list = []
            for i in range(3):
                data = dataset_api['nb101_data'].query(model_spec, query_idx=i)
                test_acc_list.append(data['test_accuracy'])
                logger.info(f'index {i} test_accuracy {data["test_accuracy"]}')

            result_dict[f'{weight_path}_{algo}'] = test_acc_list

    for key, item in result_dict.items():
        logger.info(f'{key}, {item}')