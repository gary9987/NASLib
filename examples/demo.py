import logging
import sys
import numpy as np
from naslib.search_spaces.core.query_metrics import Metric
from naslib.defaults.trainer import Trainer
from naslib.optimizers import (
    RandomSearch,
    RegularizedEvolution,
    LocalSearch,
    Bananas,
    BasePredictor,
)
from naslib.search_spaces.nasbench201.conversions import (
    convert_op_indices_to_naslib,
    convert_naslib_to_op_indices,
    convert_naslib_to_str,
    convert_op_indices_to_str,
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
from naslib.utils import setup_logger, get_dataset_api

# Read args and config, setup logger
config = utils.get_config_from_args()

val_list = []
hash_list = []
test_list = []
for seed in range(10):
    config.seed = seed
    config.search.seed = seed
    utils.set_seed(config.seed)

    logger = setup_logger(config.save + "/log.log")
    # logger.setLevel(logging.INFO)   # default DEBUG is very verbose

    utils.log_args(config)

    supported_optimizers = {
        "rs": RandomSearch(config),
        "re": RegularizedEvolution(config),
        "ls": LocalSearch(config),
        "bananas": Bananas(config),
        "bp": BasePredictor(config),
    }

    # Changing the search space is one line of code
    #search_space = SimpleCellSearchSpace()
    search_space = NasBench101SearchSpace()
    # search_space = HierarchicalSearchSpace()
    # search_space = NasBench301SearchSpace()
    #search_space = NasBench201SearchSpace()

    # Changing the optimizer is one line of code
    # optimizer = supported_optimizers[config.optimizer]
    dataset_api = get_dataset_api(config.search_space, config.dataset)
    optimizer = supported_optimizers[config.optimizer]
    optimizer.adapt_search_space(search_space, dataset_api=dataset_api)

    # Start the search and evaluation
    trainer = Trainer(optimizer, config)
    
    if not config.eval_only:
        checkpoint = utils.get_last_checkpoint(config) if config.resume else ""
        trainer.search(resume_from=checkpoint)

    checkpoint = utils.get_last_checkpoint(config, search=False) if config.resume else ""
    val_acc, test_acc = trainer.evaluate(resume_from=checkpoint, dataset_api=dataset_api)
    #test_acc = trainer.evaluate(resume_from=checkpoint, dataset_api=dataset_api, metric=Metric.TEST_ACCURACY)
    #best = optimizer.get_final_architecture()
    #print(convert_naslib_to_str(best))
    val_list.append(val_acc)
    test_list.append(test_acc)
    #hash_list.append(convert_naslib_to_str(best))

print(val_list)
print(test_list)

print(np.mean(val_list), np.std(val_list))
print(np.mean(test_list), np.std(test_list))

