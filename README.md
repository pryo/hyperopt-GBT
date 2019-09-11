# hyperopt-GBT
This is a custom staking framework based on hyperopt that support XGB and lightgbm model.
When using the hypeopt to build a stacking of models. If early stopping parameter was used for lightGBM or XGB model when using their Sklearn API. The kstrightfolding iterator will be consumed during training, so that you receive no validation set error from the L or X model. When creating the blend of dataset or creating stack layers(wrapped with Skleanr API), please use Stack.py, example provide in lgb_debug.py.
