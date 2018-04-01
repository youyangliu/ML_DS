
# coding: utf-8

# # Problem 1

# In[1]:

from sklearn.linear_model import Ridge, LogisticRegression, Lasso
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import make_scorer, mean_squared_error
from sklearn.model_selection import cross_val_score, GridSearchCV
import numpy as np
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import make_pipeline


# In[2]:

#load data
from scipy.io import loadmat
data = loadmat('hw4data.mat')


# In[3]:

#split data into train, test, and quiz

split = int(data['data'].shape[0]*0.75)
len = data['data'].shape[0]

x_train = data['data'][0:split,]
x_test = data['data'][split:len,]

y_train = data['labels'][0:split,]
y_test = data['labels'][split:len,]

x_quiz = data['quiz']


# In[4]:

mse=make_scorer(mean_squared_error)


# ## Ridge - default params - CV

# In[5]:

#ridge cross validation using original training data and default parameters
ridge = Ridge()
mse=make_scorer(mean_squared_error)
print(np.mean(cross_val_score(ridge,x_train,y_train,scoring ='neg_mean_squared_error')))


# ## Ridge - scaled and polynomial expansion - CV

# In[8]:

#scale training data and apply polynomial expansion

scaler = StandardScaler()
x_tr_scaled = scaler.fit_transform(x_train)

poly = PolynomialFeatures()
x_tr_poly = poly.fit_transform(x_tr_scaled)
print(x_tr_scaled.shape)
print(x_tr_poly.shape)

poly_pipe = make_pipeline(StandardScaler(), PolynomialFeatures(), Ridge())
print(np.mean(cross_val_score(poly_pipe,x_train,y_train,scoring =mse)))

poly_pipe.fit(x_train,y_train)
y_tr_hat=poly_pipe.predict(x_train)
print(mean_squared_error(y_train,y_tr_hat))

y_te_hat=poly_pipe.predict(x_test)
print(mean_squared_error(y_test,y_te_hat))


# ## Ridge - scaled and polynomial expansion - grid search

# In[67]:

#grid search on polynomially transformed data

poly_pipe = make_pipeline(StandardScaler(), PolynomialFeatures(), Ridge())

ridge_grid = GridSearchCV(poly_pipe,{'ridge__alpha':np.logspace(-3, 3, 7)},
                          return_train_score=True, scoring = 'mean_squared_error')

ridge_grid.fit(x_train, y_train)

print(ridge_grid.best_params_)
print(ridge_grid.best_score_)


# In[73]:

#calculate training and test mean square risk

y_tr_hat=ridge_grid.predict(x_train)
print(mean_squared_error(y_train,y_tr_hat))

y_te_hat=ridge_grid.predict(x_test)
print(mean_squared_error(y_test,y_te_hat))


# In[69]:

#predict quiz label
y_quiz_hat = ridge_grid.predict(x_quiz)


# In[70]:

#construct conditional probablity estimator

eta_ti = (1+y_quiz_hat)*0.5

np.place(eta_ti,eta_ti>1,1)
np.place(eta_ti,eta_ti<0,0)

P_Q = sum(eta_ti)/eta_ti.shape[0]
print(P_Q)


# ## Random Forest - default - CV

# In[71]:

#grid search random forest regressor using original data

pRFR = make_pipeline(RandomForestRegressor(random_state=0))

gRFR = GridSearchCV(pRFR,{'randomforestregressor__n_estimators':[10,15,20],
                          'randomforestregressor__max_depth':[10,12,15]},
                    return_train_score=True,scoring ='neg_mean_squared_error')

gRFR.fit(x_train, y_train)

print("Random forest regressor best parameter:"+str(gRFR.best_params_))
print("Random forest regressor best mean cv score: {:.10f}".format(gRFR.best_score_))


# In[72]:

#calculate training and test mean square risk
y_tr_hat=gRFR.predict(x_train)
print(mean_squared_error(y_train,y_tr_hat))

y_te_hat=gRFR.predict(x_test)
print(mean_squared_error(y_test,y_te_hat))

#consrtuct conditional probability estimator
h_hat = gRFR.predict(x_quiz)
eta_ti = (1+h_hat)*0.5

np.place(eta_ti,eta_ti>1,1)
np.place(eta_ti,eta_ti<0,0)

P_Q = sum(eta_ti)/eta_ti.shape[0]
print(P_Q)

