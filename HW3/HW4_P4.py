
# coding: utf-8

# In[52]:


import numpy as np
from scipy.io import loadmat
data = loadmat('hw3data.mat')
import pandas as pd
from sklearn import preprocessing


############################  realization of soft_svm function  ############################
def SSVM(n, c, cir, X, y):
    
    alpha = np.zeros((n,1))
    K = np.matmul(X, X.transpose())
    
    ######  calculate the value of alpha  ######
    for m in range(cir):
        for i in range(n):
            
            bound = (1 - 2 * y[i] * (np.matmul(K[i, :],np.multiply(alpha, y)) 
                                - (K[i,i]*alpha[i]*y[i])))/(2 * y[i] * y[i] * K[i,i])
            
            if c <= bound:
                alpha[i] = c
            elif 0 >= bound:
                alpha[i] = 0
            else:
                alpha[i] = bound
                
    ######  calculate the value of object value  ######    
    obj = np.sum(alpha)
    obj -= sum(sum(np.multiply(np.matmul(np.multiply(y, alpha), 
                                         np.multiply(y, alpha).transpose()),
                               np.matmul(X, X.transpose()))))
    
    return alpha, obj



############################  standerization and clean the input data  ############################
x = data['data']
y = np.int8(data['labels'])
y[y == 0] = -1


######  set parameters  ######
n = len(y)
c = 10.0/n
cir = 2

######  standerization  ######
X_rescaled = preprocessing.scale(x)


######  get the object value of data  ######
alpha, obj = SSVM(n, c, cir, X_rescaled, y)
print (obj)


######  get the weight value of data  ######
w1, w2, w3 = 0,0,0
for i in range(n):
    w1 += alpha[i]* y[i]* X_rescaled[i, 0]
    w2 += alpha[i]* y[i]* X_rescaled[i, 1]
    w3 += alpha[i]* y[i]* X_rescaled[i, 2]
print(w1, w2, w3) 


