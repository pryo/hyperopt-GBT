#lgb kernel

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
import os
print(os.listdir("../input"))
import sys


sys.stdout.write('Let me do this real quick! \n')
train_df = pd.read_csv('./data/train0.csv')
test_df = pd.read_csv('../data/test.csv')
features = [c for c in train_df.columns if c not in ['ID_code', 'target']]
target = train_df['target']
param = {
    'bagging_freq': 5,          'bagging_fraction': 0.38,   'boost_from_average':'false',   'boost': 'gbdt',
    'feature_fraction': 0.045,   'learning_rate': 0.0105,     'max_depth': -1,                'metric':'auc',
    'min_data_in_leaf': 80,     'min_sum_hessian_in_leaf': 10.0,'num_leaves': 13,           'num_threads': 8,
    'tree_learner': 'serial',   'objective': 'binary',      'verbosity': 1,
    'device_type' : 'gpu',    'gpu_platform_id' : 0,    'gpu_device_id' : 0
}

folds = StratifiedKFold(n_splits=12, shuffle=False, random_state=44000)
oof = np.zeros(len(train_df))
predictions = np.zeros(len(test_df))

for fold_, (trn_idx, val_idx) in enumerate(folds.split(train_df.values, target.values)):
    print("Fold :{}".format(fold_ + 1))
    trn_data = lgb.Dataset(train_df.iloc[trn_idx][features], label=target.iloc[trn_idx])
    val_data = lgb.Dataset(train_df.iloc[val_idx][features], label=target.iloc[val_idx])
    clf = lgb.train(param, trn_data, 100000, valid_sets = [trn_data, val_data], verbose_eval=1000, early_stopping_rounds = 1000)
    oof[val_idx] = clf.predict(train_df.iloc[val_idx][features], num_iteration=clf.best_iteration)
    predictions += clf.predict(test_df[features], num_iteration=clf.best_iteration) / folds.n_splits
sys.stdout.write("CV score: {:<8.5f}".format(roc_auc_score(target, oof)))
print("CV score: {:<8.5f}".format(roc_auc_score(target, oof)))

sub = pd.DataFrame({"ID_code": test_df.ID_code.values})
sub["target"] = predictions
sub.to_csv('submission.csv', index=False)