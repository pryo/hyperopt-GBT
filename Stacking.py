import lightgbm as lgb
import numpy as np
from sklearn.model_selection import StratifiedKFold
from multiprocessing import Process
from multiprocessing import Pool
from sklearn.pipeline import Pipeline
import random
from datetime import datetime
class layering:
    def __init__(self,clf_list):# get foldings from sklean StratifiedKFold
        self.clfs = clf_list

        self.output =None



    # def para_blend(self,clf,X,y,cv=2):
    #     if cv<=4:
    #
    #         num_process  = cv
    #     else:
    #         num_process=4
    #     output = np.ndarray([len(X), 1])
    #     kfolds = StratifiedKFold(n_splits=cv, random_state=datetime.now().microsecond).split(X, y)
    #     #when set size
    #     pool = Pool(processes=num_process)
    #     if isinstance(clf, lgb.LGBMClassifier):
    #         pool.starmap(
    #             clf.fit,
    #             [(X[train_idx], y[train_idx],
    #                         eval_set=[(X[val_idx], y[val_idx])]).predict_proba(X[val_idx]
    #                                                                            for train_idx,val_idx in kfolds )])
    #         y_val = clf.fit(X[train_idx], y[train_idx],
    #                         eval_set=[(X[val_idx], y[val_idx])]).predict_proba(X[val_idx])[:, 0]
    #         # clf.set_params(eval_set=list(zip(X[val_idx],y[val_idx])))
    #     else:
    #         y_val = clf.fit(X[train_idx], y[train_idx]).predict_proba(X[val_idx])[:, 0]
    #
    #
    #
    #
    #     for p in range(num_process):
    #         p = Process(target=f, args=('bob',))
    #


    def blend(self,clf,X,y,cv=2):
        output = np.ndarray([len(X),1])
        kfolds = StratifiedKFold(n_splits=cv, random_state=datetime.now().microsecond).split(X, y)
        for train_idx,val_idx in kfolds:
            if isinstance(clf, lgb.LGBMClassifier):
                y_val = clf.fit(X[train_idx], y[train_idx],
                                eval_set=[(X[val_idx], y[val_idx])]).predict_proba(X[val_idx])[:,0]
                    #clf.set_params(eval_set=list(zip(X[val_idx],y[val_idx])))
            else:
                y_val=clf.fit(X[train_idx],y[train_idx]).predict_proba(X[val_idx])[:,0]

            output[val_idx,:] = np.reshape(y_val,[len(y_val),1])
        return output

    # def fit(self,X,y,folds):
    #     for i in range(len(self.clfs)):
    #         if isinstance(self.clfs[i],lgb.LGBMClassifier):
    #             self.clfs[i].set_params()


    def fit_blend(self,X,y,cv=2):
        output = np.ndarray([len(X),len(self.clfs)])
        for i in range(len(self.clfs)):
            blend_output = self.blend(self.clfs[i],X,y,cv=cv)
            output[:,i]=np.reshape(blend_output,[len(blend_output),])
        self.output = output
        return output

    def predict(self,X):
        output = np.ndarray([len(X), len(self.clfs)])

        for i in range(len(self.clfs)):

            predict= self.clfs[i].predict_proba(X)[:,0]
            output[:,i] = predict

        return output
