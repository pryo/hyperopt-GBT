from hyperopt import fmin, tpe, hp
from hyperopt.mongoexp import MongoTrials
from sklearn.model_selection import cross_val_score
import lightgbm as lgb
#define objective function
import pickle as pkl

class Optimizer:
    def __init__(self,X,y,folds =10):
        self.folds = folds
        self.X = X
        self.y = y
        #put hp generator in here
        self.space = {

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


    #     self.space = {
    #     'max_depth': hp.quniform('max_depth', 2, 8, 1),
    #     'colsample_bytree': hp.uniform('colsample_bytree', 0.3, 1.0),
    #     'gamma': hp.uniform('gamma', 0.0, 0.5),
    # }

        self.OPTIMIZED = False

    def objective(self,params):
        #put fixed(non-hp object in here)
        decoded_param = {
            'bagging_freq': params['bagging_freq'],
            'bagging_fraction': params['bagging_fraction'],
            'boost_from_average': 'false',
            'boost': 'gbdt',
            'feature_fraction':params['feature_fraction'],
            'learning_rate': params['learning_rate'],
            'max_depth': -1,
            'metric': 'auc',
            'min_data_in_leaf': params['min_data_in_leaf'],
            'num_leaves':params['num_leaves'],
            'tree_learner': 'data',
            'objective': 'binary',
            'verbose': -1,
            'device_type': 'gpu',
            'gpu_platform_id': 0,
            'gpu_device_id': 0
        }
        # the stuff fixed

        clf = lgb.LGBMClassifier(**decoded_param)


        # clf = xgb.XGBClassifier(
        #     n_estimators=250,
        #     learning_rate=0.05,
        #     n_jobs=4,
        #     **params
        # )

        mean_auc = cross_val_score(clf, self.X, self.y, scoring='roc_auc', cv=self.folds).mean()
        print("auc {:.3f} params {}".format(mean_auc, params))
        return {'loss':1-mean_auc,'result':mean_auc}

    def getTrials(self,exp_key,mongo=True):
        if mongo:
            return MongoTrials('mongo://localhost:27017/santander/jobs', exp_key=exp_key)
        else:
            raise Exception("you have to make use of mongodb!")
    def save(self,param_path):
        #save the best
        fp = open(param_path,'w+')
        pkl.dump(self.best,fp)
        fp.close()

    def start(self,exp_key,param_path):
        self.trials=self.getTrials(exp_key=exp_key)
        self.best = fmin(self.objective,space=self.space, trials=self.trials, algo=tpe.suggest, max_evals=500)
        print("experiment finished!")
        self.OPTIMIZED = True
        self.save(param_path)




