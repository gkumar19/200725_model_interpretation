# -*- coding: utf-8 -*-
"""
Created on Sun Jul  5 23:33:44 2020

@author: KGU2BAN
"""

# Feature importance
# In multicollinearity, permutation importance under-value the importance
# Since when one feature is randomize the other feature still maitains the model score
# In that case removal of redundant feature is the option to go about it

# impurity-based feature importance can inflate the importance of numerical features or categorical
# feature with high unique calues [high cardinality]

# Permutation importance evalue the score after randomizing and returns the reduction in score
# as a proxy for importance

import numpy as np
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Lasso, LinearRegression, SGDRegressor
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf

#%%
m = 1000
x1 = np.random.normal(size=(m,1))
x2 = np.random.randint(0,6,size=(m,1))
x3 = np.random.normal(size=(m,1))
x4 = np.random.normal(size=(m,1))
x5 = np.random.normal(size=(m,1))

X = np.concatenate([x1, x2, x3, x4, x5], axis=-1)
y1 = 4*x1**2 + 4*x2 + 3*x3 + 3*x4 + 4*x5
y2 = 4*x1**2 + 4*x2 + 3*x3 + 4*x4 + 4*x5
y = np.concatenate([y1, y2], axis=-1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=47)

# models = dict(ExtraTreesRegressor = ExtraTreesRegressor(),
#               RandomForestRegressor = RandomForestRegressor(),
#               GradientBoostingRegressor = GradientBoostingRegressor(),
#               Lasso = Lasso(),
#               LinearRegression = LinearRegression(),
#               SGDRegressor = SGDRegressor(),
#               SVR = SVR())

models = dict(ExtraTreesRegressor = ExtraTreesRegressor())

# def build_keras_model():
#     model = tf.keras.models.Sequential([tf.keras.layers.Dense(10,input_shape=(5,),
#                                                                 name='input', activation='sigmoid'),
#                                           tf.keras.layers.Dense(1,name='output', activation='linear')])
#     model.compile(optimizer = 'adam', loss='mse')
#     return model
# keras_model = tf.keras.wrappers.scikit_learn.KerasRegressor(build_keras_model,
#                                                             epochs=500, batch_size=100)

# models = dict(ExtraTreesRegressor = ExtraTreesRegressor(),
#               KerasModel = keras_model)


def scorer(n_feature=0):
    from sklearn.metrics import r2_score
    from sklearn.metrics import make_scorer
    def my_custom_loss_func(y_true, y_pred):
        return r2_score(y_true[:,n_feature], y_pred[:,n_feature])
    return make_scorer(my_custom_loss_func)
    

for model_name, model in models.items():
    model.fit(X_train, y_train)
    
    #feature_importance_train = permutation_importance(model, X_train, y_train,
    #                        n_repeats=30,
    #                        random_state=0, scoring=score)
    feature_importance_test = permutation_importance(model, X_test, y_test,
                            n_repeats=30,
                            random_state=0, scoring=scorer(1))
    #plt.figure('train'+model_name)
    #plt.boxplot(feature_importance_train['importances'].T)
    plt.figure('test'+model_name)
    plt.boxplot(feature_importance_test['importances'].T)
#plt.figure()
#plt.bar(range(X.shape[-1]), models['ExtraTreesRegressor'].feature_importances_)

#%% check the meaning of importance in permutation importance
model = models['ExtraTreesRegressor'] #select one model for further evaluation
n_importance = 1
print(model.score(X_test, y_test))
X_test_copy = X_test.copy()
np.random.shuffle(X_test_copy[:,n_importance])
print(model.score(X_test_copy, y_test))

#%% Partial Dependence plots
from matplotlib import pyplot as plt
from pdpbox import pdp, get_dataset, info_plots
import pandas as pd

# Create the data that we will plot
pdp_data = pdp.pdp_isolate(model=model, dataset=pd.DataFrame(X_test),
                           model_features=range(X.shape[-1]),
                           feature=0)

# plot it
pdp.pdp_plot(pdp_data, 'lol')
plt.show()

#%%
from pdpbox import pdp, get_dataset, info_plots
test_titanic = get_dataset.titanic()
titanic_data = test_titanic['data']
titanic_features = test_titanic['features']
titanic_model = test_titanic['xgb_model']
titanic_target = test_titanic['target']
