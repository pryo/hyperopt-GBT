import utility
import numpy as np
#https://www.kaggle.com/sandeepkumar121995/magic-parameters
name = 'magic'
param= {
    'num_iterations':99999999,
    'early_stopping_rounds':3500,
    'bagging_freq': 5,
    'bagging_fraction': 0.335,
    'boost_from_average':'false',
    'boost': 'gbdt',
    'feature_fraction': 0.041,
    'learning_rate': 0.0083,
    'max_depth': -1,
    'metric':'auc',
    'min_data_in_leaf': 80,
    'min_sum_hessian_in_leaf': 10.0,
    'num_leaves': 13,
    'num_threads': 8,
    'tree_learner': 'serial',
    'objective': 'binary',
    'verbosity': -1,
    'device_type' : 'gpu',
    'gpu_platform_id' : 0,
    'gpu_device_id' : 0,}

utility.param2json(param,name)

#catboost
#https://www.kaggle.com/bogorodvo/starter-code-saving-and-loading-lgb-xgb-cb
name = 'catboost'
param={
    'iterations':999999,
    'max_depth':2,
    'early_stopping_rounds':2000,
    'learning_rate':0.02,
    'colsample_bylevel':0.03,
    'objective':'YetiRank',
    'task_type':'GPU'
}

utility.param2json(param,name)

# name = 'GridSearchLgReg'
# param = {
#     'penalty': ['l1', 'l2'],
#     'C':np.logspace(0, 4, 10),
#     'class_weight':['balanced']
# }
# utility.param2json(param,name)