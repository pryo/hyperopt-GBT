#load data
import numpy as np
import pandas as pd
import os.path
import Hyperopt_lightgbm
training_frame = pd.read_csv("./data/train.csv")

training_frame=training_frame.drop(columns=["ID_code"])
y = training_frame["target"]
X = training_frame.drop(columns=["target"])

X = X.to_numpy()
y = y.to_numpy()
#make optmizer

optimizer =Hyperopt_lightgbm.Optimizer(X,y)

param_prefix='~/Developer/santander/santander/params/'
exp_key = 'exp1'
#run
optimizer.start(exp_key,os.path.join(param_prefix,exp_key))