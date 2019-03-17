exec(open("./make_configs.py").read())

import numpy as np
import pandas as pd
import utility

training_frame = pd.read_csv("./data/train.csv")

training_frame=training_frame.drop(columns=["ID_code"])
y = training_frame["target"]
X = training_frame.drop(columns=["target"])

X = X.to_numpy()
y = y.to_numpy()

# creat lgbm model
import lightgbm as lgb

# param = {
#     'bagging_freq': 5,          'bagging_fraction': 0.38,   'boost_from_average':'false',   'boost': 'gbdt',
#     'feature_fraction': 0.045,   'learning_rate': 0.0105,     'max_depth': -1,                'metric':'auc',
#     'min_data_in_leaf': 80,     'min_sum_hessian_in_leaf': 10.0,'num_leaves': 13,           'num_threads': 8,
#     'tree_learner': 'serial',   'objective': 'binary',      'verbosity': 1,
#     'device_type' : 'gpu',    'gpu_platform_id' : 0,    'gpu_device_id' : 0,
#     'early_stopping_rounds':1000
# }
lgb_param = utility.json2param('magic')
lgb_clf0 = lgb.LGBMClassifier(**lgb_param)
lgb_clf1 = lgb.LGBMClassifier(**lgb_param)
lgb_clf2 = lgb.LGBMClassifier(**lgb_param)
#catboost
import catboost
cat_param = utility.json2param('catboost')
#cat_clf= catboost.CatBoostClassifier(**cat_param)

# create NB clf
from sklearn.preprocessing import QuantileTransformer
from sklearn.pipeline import make_pipeline
from sklearn.naive_bayes import GaussianNB
#NB_clf = make_pipeline(QuantileTransformer(output_distribution='normal'), GaussianNB(n_classes=2))
NB_clf = make_pipeline(QuantileTransformer(output_distribution='normal'), GaussianNB())
import Stacking
# make stacking
# from sklearn.model_selection import StratifiedKFold
# kfold = StratifiedKFold(n_splits=2, random_state=999).split(X,y)
#kfold is generator that will destory itself after one usage
clf_list = [lgb_clf0,lgb_clf1,lgb_clf2,NB_clf]
layer0= Stacking.layering(clf_list)
layer0_out = layer0.fit_blend(X,y)

# #last layer(meta)
# from sklearn.linear_model import LogisticRegression
# meta_clf = LogisticRegression(n_jobs=4,random_state=123)
#
# # meta_clf.fit(layer0_out,y.reshape(-1,1))
# meta_clf.fit(layer0_out,y)

#last layer with Grid search
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
hyperparameters = utility.json2param('GridSearchLgReg')
logistic=LogisticRegression(n_jobs=4,random_state=123)
meta_clf=GridSearchCV(logistic, hyperparameters, cv=3, verbose=0)
best_meta = meta_clf.fit(layer0_out, y.reshape(-1,1))



#get test
test_frame = pd.read_csv("./data/test.csv")
X_test = test_frame.drop(columns=["ID_code"]).to_numpy()

#get prediction
y_test=best_meta.predict_proba(layer0.predict(X_test))
result =pd.DataFrame(dict(ID_code = test_frame["ID_code"], target = y_test[:,1]))
#save file
print('writting file...')
result.to_csv("stack_result.csv",index = False)
#txt my phone
utility.notify('+14435629996','Training complete!')
