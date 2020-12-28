import logging
import sys
import os
import naslib as nl


from naslib.defaults.predictor_evaluator import PredictorEvaluator

from naslib.predictors import Ensemble, FeedforwardPredictor, GBDTPredictor, \
EarlyStopping, GCNPredictor, BonasGCNPredictor, jacobian_cov, SoLosspredictor, \
SVR_Estimator, XGBoost, NGBoost, RandomForestPredictor, DNGOPredictor, \
BOHAMIANN, BayesianLinearRegression, LCNetPredictor, FeedforwardKerasPredictor, \
SemiNASPredictor, GPPredictor, SparseGPPredictor, VarSparseGPPredictor

from naslib.search_spaces import NasBench201SearchSpace, DartsSearchSpace
from naslib.search_spaces.core.query_metrics import Metric

from naslib.utils import utils, setup_logger
from naslib.utils.utils import get_project_root

from fvcore.common.config import CfgNode


config = utils.get_config_from_args(config_type='predictor')

utils.set_seed(config.seed)
logger = setup_logger(config.save + "/log.log")
logger.setLevel(logging.INFO)

utils.log_args(config)

supported_predictors = {
    'bananas': Ensemble(encoding_type='path',
                        predictor_type='feedforward'),
    'feedforward': FeedforwardPredictor(encoding_type='adjacency_one_hot'),
    'ff_keras': FeedforwardKerasPredictor(encoding_type='adjacency_one_hot'),
    'gbdt': GBDTPredictor(encoding_type='adjacency_one_hot'),
    'gcn': GCNPredictor(encoding_type='gcn'),
    'bonas_gcn': BonasGCNPredictor(encoding_type='bonas_gcn'),
    'valloss': EarlyStopping(dataset=config.dataset, metric=Metric.VAL_LOSS),
    'valacc': EarlyStopping(dataset=config.dataset, metric=Metric.VAL_ACCURACY),
    'jacov': jacobian_cov(config, task_name='nas201_cifar10', batch_size=256),
    'sotl': SoLosspredictor(dataset=config.dataset, metric=Metric.TRAIN_LOSS, sum_option='SoTL'),
    'lcsvr': SVR_Estimator(dataset=config.dataset, metric=Metric.VAL_ACCURACY),
    'xgb': XGBoost(encoding_type='adjacency_one_hot'),
    'ngb': NGBoost(encoding_type='adjacency_one_hot'),
    'rf': RandomForestPredictor(encoding_type='adjacency_one_hot'),
    'dngo': DNGOPredictor(encoding_type='adjacency_one_hot'),
    'bohamiann': BOHAMIANN(encoding_type='adjacency_one_hot'),
    'lcnet': LCNetPredictor(encoding_type='adjacency_one_hot'),
    'bayes_lin_reg': BayesianLinearRegression(encoding_type='adjacency_one_hot'),
    'seminas': SemiNASPredictor(encoding_type='seminas'),
    'gp': GPPredictor(encoding_type='adjacency_one_hot'),
    'sparse_gp': SparseGPPredictor(encoding_type='adjacency_one_hot',
                                   optimize_gp_hyper=True, num_steps=100),
    'var_sparse_gp': VarSparseGPPredictor(encoding_type='adjacency_one_hot',
                                          optimize_gp_hyper=True, num_steps=200),
}

supported_search_spaces = {
    'nasbench201': NasBench201SearchSpace(),
    'darts': DartsSearchSpace()
}

load_labeled = (False if config.search_space == 'nasbench201' else True)

# set up the search space and predictor
predictor = supported_predictors[config.predictor]
search_space = supported_search_spaces[config.search_space]
predictor_evaluator = PredictorEvaluator(predictor, config=config)
predictor_evaluator.adapt_search_space(search_space, load_labeled=load_labeled)

# evaluate the predictor
predictor_evaluator.evaluate()