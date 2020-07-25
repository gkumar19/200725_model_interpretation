# -*- coding: utf-8 -*-
"""
Created on Fri Jul 10 23:04:54 2020

@author: KGU2BAN
"""

#import tensorflow as tf
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

#%%
m = 1000
x1 = np.random.normal(size=(m,1))
x2 = np.random.randint(0,6,size=(m,1))
x3 = np.random.normal(size=(m,1))
x4 = np.random.normal(size=(m,1))
x5 = np.random.normal(size=(m,1))

X = np.concatenate([x1, x2, x3, x4, x5], axis=-1)
X = pd.DataFrame(X, columns=['x'+str(i) for i in range(1,6)])
y = x1 + 2*x2 + 3*x3 + 4*x4 - 5*x5
y = y.ravel()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=47)

model = LinearRegression()
model.fit(X_train, y_train)
print(model.score(X_test, y_test))

#%% plotting the pdp through basic principles
pdp_feature = 'x1'
num_discrete_points = 10
pdp_feature_values = X[pdp_feature].values
percentile_linspace = np.linspace(0, 100, num_discrete_points)
percentile_values = [np.percentile(pdp_feature_values, i) for i in percentile_linspace.tolist()]
predictions = np.zeros((X.shape[0], num_discrete_points))
for i, percentile_value in enumerate(percentile_values):
    X_temp = X.copy()
    X_temp[pdp_feature] = percentile_value
    predictions[:, i] = model.predict(X_temp)

predictions = predictions.T

mean_predictions = predictions.mean(axis=1, keepdims=True)
std_predictions = predictions.std(axis=1, keepdims=True)
mean_std_predictions = np.concatenate([mean_predictions-std_predictions,
                                       mean_predictions,
                                       mean_predictions+std_predictions], axis=1)
plt.figure()
plt.plot(percentile_values, mean_std_predictions)
plt.scatter(x5, -x5**3)

#%% 1D pdp
from pdpbox import pdp
pdp_feature = 'x5'
pdp_data = pdp.pdp_isolate(model=model, dataset=X,
                           model_features=X.columns,
                           feature=pdp_feature,
                           num_grid_points=20, grid_type='percentile')
pdp.pdp_plot(pdp_data, 'plot for x1', center=False, plot_pts_dist=True,
             plot_lines=True, frac_to_plot=0.5,
             cluster=False, n_cluster_centers=5,
             x_quantile=False, show_percentile=False)

#conclusion: pdp plots are like keep predicting over X, with pdp_feature as sequentially
#constant multiple number of times at different percentile

#%% 2D pdp
pdp_data = pdp.pdp_interact(model, dataset=X,
                        model_features=X.columns,
                        features=['x4', 'x5'],
                        num_grid_points=[5, 5],
                        grid_types=['percentile', 'percentile'])

pdp.pdp_interact_plot(pdp_data, ['x4_plot', 'x5_plot'], plot_type='contour',
                      x_quantile=False, plot_pdp=False,
                      which_classes=None, ncols=2,
                      plot_params=None)

#%%
#y = x1**2 + x2**3 + x3**4 + 2*x4**2 - x5**3
i = 1.3
print(i, 2*i , 3*i, 4*i, -5*i)
print(i + 2*i + 3*i + 4*i -5*i)
import shap
data_for_prediction = np.array([[i, i, i, i, i]])
print(model.predict(data_for_prediction))

k_explainer = shap.KernelExplainer(model.predict, X)
k_shap_values = k_explainer.shap_values(data_for_prediction)
shap.force_plot(k_explainer.expected_value[1], k_shap_values[1], data_for_prediction,
                matplotlib=True)

#explainer = shap.TreeExplainer(model)
l_explainer = shap.LinearExplainer(model, X_train)

#shap_values = explainer.shap_values(data_for_prediction)
l_shap_values = l_explainer.shap_values(data_for_prediction)

#shap.initjs()
shap.force_plot(l_explainer.expected_value[0], l_shap_values[0], data_for_prediction,
                matplotlib=True)

#%%
plt.scatter(y_test, model.predict(X_test))
model.score(X_test, y_test)
