import logging
import sys
import keras
import tensorflow as tf
from naslib.defaults.trainer import Trainer
from naslib.optimizers import (
    DARTSOptimizer,
    GDASOptimizer,
    DrNASOptimizer,
    RandomSearch,
    RegularizedEvolution,
    LocalSearch,
    Bananas,
    BasePredictor,
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
    # Read args and config, setup logger
    config = utils.get_config_from_args()
    utils.set_seed(config.seed)

    logger = setup_logger(config.save + "/log.log")
    # logger.setLevel(logging.INFO)   # default DEBUG is very verbose

    utils.log_args(config)

    supported_optimizers = {
        "darts": DARTSOptimizer(config),
        "gdas": GDASOptimizer(config),
        "drnas": DrNASOptimizer(config),
        "rs": RandomSearch(config),
        "re": RegularizedEvolution(config),
        "ls": LocalSearch(config),
        "bananas": Bananas(config),
        "bp": BasePredictor(config),
    }

    # Changing the search space is one line of code
    # search_space = SimpleCellSearchSpace()

    search_space = NasBench201SearchSpace(model_type=MODEL_TYPE.KERAS)
    # search_space = graph.NasBench101SearchSpace()
    # search_space = HierarchicalSearchSpace()
    # search_space = NasBench301SearchSpace()
    # search_space = NasBench201SearchSpace()

    dataset_api = utils.get_dataset_api('nasbench201', dataset='cifar10')

    # keras 201
    weight_path = 'nb201_gin_model'
    dataset_api['surrogate'] = keras.models.load_model(weight_path)

    '''
    # keras
    weight_path = 'gin_conv_batch_filterTrue_a1_size95500_r1_m64_b256_dropout0.2_lr0.001_mlp(64, 64, 64, 64)_0'
    dataset_api['surrogate'] = keras.models.load_model(weight_path, custom_objects={'weighted_mse': tf.keras.losses.MeanSquaredError()})
    '''

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
    optimizer = supported_optimizers["bananas"]
    optimizer.adapt_search_space(search_space, dataset_api=dataset_api)

    # Start the search and evaluation
    trainer = Trainer(optimizer, config)
    trainer.search()

    #trainer.evaluate(dataset_api=dataset_api)
