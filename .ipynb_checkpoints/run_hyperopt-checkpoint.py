#load data
import numpy as np
import pandas as pd
from hyperopt import fmin, tpe, hp
from hyperopt.mongoexp import MongoTrials


#define objective function
import pickle as pkl
import os.path
import Hyperopt_lightgbm
training_frame = pd.read_csv("./data/train.csv")

training_frame=training_frame.drop(columns=["ID_code"])
y = training_frame["target"]
X = training_frame.drop(columns=["target"])

X = X.to_numpy()
y = y.to_numpy()
space = {

            'bagging_freq': hp.choice('bagging_freq',[5,10,30]),
            'bagging_fraction': hp.uniform('bagging_fraction',0.1,0.8),
            'boost_from_average': 'false',
            'boost': 'gbdt',
            'feature_fraction': hp.uniform('feature,fraction',0.01,0.9),
            'learning_rate': hp.loguniform('lr',-2,-10),
            'max_depth': -1,
            'metric': 'auc',
            'min_data_in_leaf': hp.quniform('min_data_in_leaf',50,150,10),
            'num_leaves': hp.uniform('num_leaves',10,100),
            'tree_learner': 'data',
            'objective': 'binary',
            'verbose': -1,
            'device_type': 'gpu',
            'gpu_platform_id': 0,
            'gpu_device_id': 0
        }


def objective(params):
    from sklearn.model_selection import cross_val_score
    # put fixed(non-hp object in here)
    decoded_param = {
        'bagging_freq': params['bagging_freq'],
        'bagging_fraction': params['bagging_fraction'],
        'boost_from_average': 'false',
        'boost': 'gbdt',
        'feature_fraction': params['feature_fraction'],
        'learning_rate': params['learning_rate'],
        'max_depth': -1,
        'metric': 'auc',
        'min_data_in_leaf': params['min_data_in_leaf'],
        'num_leaves': params['num_leaves'],
        'tree_learner': 'data',
        'objective': 'binary',
        'verbose': -1,
        'device_type': 'gpu',
        'gpu_platform_id': 0,
        'gpu_device_id': 0
    }
    # the stuff fixed
    import lightgbm as lgb
    clf = lgb.LGBMClassifier(**decoded_param)

    mean_auc = cross_val_score(clf, X, y, scoring='roc_auc', cv=10).mean()
    print("auc {:.3f} params {}".format(mean_auc, params))
    return {'loss': 1 - mean_auc, 'result': mean_auc}

def save(best,param_path):
    #save the best
    fp = open(param_path,'w+')
    pkl.dump(best,fp)
    fp.close()




#make the trail

exp_key ='exp3'
trials = MongoTrials('mongo://localhost:27017/santander/jobs', exp_key=exp_key)
#start running
best = fmin(objective,space=space, trials=trials, algo=tpe.suggest, max_evals=200)

param_prefix='~/Developer/santander/santander/params/'

save(best,os.path.join(param_prefix,exp_key))

# #make optmizer
#
# optimizer =Hyperopt_lightgbm.Optimizer(X,y)
#
# param_prefix='~/Developer/santander/santander/params/'
# exp_key = 'exp1'
# #run
# optimizer.start(exp_key,os.path.join(param_prefix,exp_key))

# hyperopt-mongo-worker --mongo=localhost:27017/santander