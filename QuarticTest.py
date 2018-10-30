
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import math
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.stats import *

import re
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


filepath = 'Assignment/'
df_train = pd.read_csv(filepath + 'data_train.csv')
df_train.shape
df_test = pd.read_csv(filepath + 'data_test.csv')
df_test.shape


# In[3]:


df_train.describe(include='all')


# In[4]:


df_test.describe(include='all')


# In[5]:


for i in range(1,15):
        name = 'cat' + str(i)
        df_train[name] = pd.Categorical(df_train[name])
        df_test[name] = pd.Categorical(df_test[name])       


# In[6]:


def display_all(df):
    with pd.option_context("display.max_rows",1000):
        with pd.option_context("display.max_columns",1000):
            display(df)
        


# In[7]:


display_all(df_train.head().T)


# In[8]:


df_train.info(verbose=True, null_counts=True)


# In[9]:


null_columns=df_train.columns[df_train.isnull().any()]
df_train[null_columns].isnull().sum()


# In[10]:


def is_numeric_dtype(col):
    if col.dtype == 'int64' or col.dtype == 'float64' or col.dtype == 'bool':
        return 1
    else :
        return 0

def preprocesstrain(df):
    meddict =  {}
#     for i in range(1,20):
#         name = 'der' + str(i)
#         df = df.drop(columns=name)
    
    for n,c in df.items():
        if is_numeric_dtype(c):
            med = c.median()
            if pd.isnull(c).sum(): 
                df[n] = c.fillna(med)
            meddict[n] = med
                
    return df,meddict

def preprocesstest(df,meddict):
    
#     for i in range(1,20):
#         name = 'der' + str(i)
#         df = df.drop(columns=name)
        
    for n,c in df.items():
        if is_numeric_dtype(c):
            if pd.isnull(c).sum():
                med = meddict[n]
                df[n] = c.fillna(med)
                
    return df

def numericalize(df,col,name):
    if not is_numeric_dtype(col):
        df[name] = df[name].cat.codes+1
        
def split_vals(a,n): return a[:n].copy(), a[n:].copy()   


# In[11]:


def GetPCA(df,var):
    scaler = StandardScaler()
    scaler.fit(df)
    df_img = scaler.transform(df)
    pca = PCA(var)
    pca.fit(df_img)
    return pca


# In[12]:


def GetTrainData(df_raw):
    
    median_dict = {}
    
    df = df_raw.copy()
    y = df['target'].values
    df.drop(['target'],axis=1,inplace=True)
    df.head().T
    
    df,median_dict = preprocesstrain(df)
    
    for n,c in df.items():
        numericalize(df,c,n)
        
    return df,y,median_dict

def GetTrainValid(df,y,n_valid):
   
    n_train = len(df) - n_valid
    X_train, X_valid = split_vals(df,n_train)
    y_train, y_valid = split_vals(y,n_train)
    
    return X_train,y_train,X_valid,y_valid


# In[13]:


mediandict = {}
df,y,mediandict = GetTrainData(df_train)
X_train,y_train,X_valid,y_valid = GetTrainValid(df,y,25000)


# In[14]:


corrdata = df.iloc[:,0:42]
corrdata.apply(lambda x : pd.factorize(x)[0]).corr(method='pearson', min_periods=1)


# In[15]:


i = 9
h = df.iloc[:,i]
h = np.sort(h)
hmean = np.mean(h)
hstd = np.std(h)
pdf = norm.pdf(h, hmean, hstd)
plt.plot(h, pdf)


# In[16]:


h = df.iloc[:,47]
h.value_counts().plot(kind='bar')


# In[17]:


chitestmat = list()
colcount = len(df.columns)
for i in range(42,colcount) :
    ind_chi_test = chisquare(df.iloc[:,i])
    chitestmat.append(ind_chi_test[1])
np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
display_all(chitestmat)


# In[18]:


colcount = len(df.columns)
ttestmat = np.zeros((colcount,colcount))
for i in range(42,colcount) :
    for j in range(42,colcount) :
        if i != j :
            ind_t_test = ttest_ind(df.iloc[:,i],df.iloc[:,j])
            ttestmat[i][j] = ind_t_test[1]
np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
display_all(ttestmat)


# In[19]:


h = pd.Series(y.tolist())
h.value_counts().plot(kind='bar')


# In[20]:


def TrainRandomForest(X_train,y_train,X_valid,y_valid):

    m = RandomForestClassifier(n_estimators=40,n_jobs=-1,oob_score=True)
    get_ipython().run_line_magic('time', 'm.fit(X_train,y_train)')
    res = [m.score(X_train,y_train),m.score(X_valid,y_valid)]
    if hasattr(m,'oob_score_'): res.append(m.oob_score_)
    print(res)
    
    return m


# In[21]:


def TrainLogisticRegression(X_train,y_train,X_valid,y_valid):
    m = LogisticRegression()
    get_ipython().run_line_magic('time', 'm.fit(X_train, y_train)')
    res = [m.score(X_train,y_train),m.score(X_valid,y_valid)]
    print(res)
    
    return m


# In[22]:


def ReadyTestData(df_test,median_dict) :
    predmat = np.zeros((df_test.shape[0],2))
    
    df = df_test.copy()
    ids = df['id'].values
    #df.drop(['id'],axis=1,inplace=True)
    
    df = preprocesstest(df,median_dict)
    
    for n,c in df.items():
        numericalize(df,c,n)
        
    return df,ids


def PredictTestSet(X_test, ids, predicted_model):
          
    y = predicted_model.predict(X_test)
    predmat = np.column_stack((ids,y)) 
    
    return predmat

def GetImportances(predictedmodel,df) :
    importances = predictedmodel.feature_importances_
    indices = np.argsort(importances)[::-1]
    df_imp = pd.DataFrame(columns=['cols','imp'])
    j = 0
    for f in range(df.shape[1]):
        df_imp.loc[j] = [df.columns[indices[f]],importances[indices[f]]]
        j = j + 1
    return df_imp
    


# In[23]:


# pca = GetPCA(X_train,0.95)
# X_train = pca.transform(X_train)
# X_valid = pca.transform(X_valid)


# In[24]:


predictedmodel = TrainRandomForest(X_train,y_train,X_valid,y_valid)


# In[25]:


yvalid_predict = predictedmodel.predict(X_valid)
print(classification_report(y_valid, yvalid_predict))


# In[26]:


X_test,ids = ReadyTestData(df_test,mediandict)
# X_test = pca.transform(X_test)
predmat = PredictTestSet(X_test, ids, predictedmodel)
df_print = pd.DataFrame(predmat,columns=["id","target"])
df_print.to_csv("C:\\Users\\Admin\\Downloads\\QuarticPred.csv",index=False)


# In[27]:


get_ipython().run_line_magic('time', 'preds = np.stack([t.predict(X_valid) for t in predictedmodel.estimators_])')
np.mean(preds[:,0]),np.std(preds[:,0])


# In[28]:


df_imp = GetImportances(predictedmodel,df); df_imp[:10]


# In[29]:


df_imp.plot('cols','imp',figsize=(10,6),legend=False)


# In[30]:


to_keep = df_imp[df_imp.imp > 0.02].cols;len(to_keep)


# In[31]:


mediandict = {}
df,y,mediandict = GetTrainData(df_train)
df_keep = df[to_keep].copy()
X_train_new,y_train_new,X_valid_new,y_valid_new = GetTrainValid(df_keep,y,250000)
newmodel = TrainRandomForest(X_train_new,y_train_new,X_valid_new,y_valid_new)


# In[32]:


fi = GetImportances(newmodel,df_keep); fi[:]


# In[33]:


fi.plot('cols','imp','barh',figsize=(12,7),legend=False)


# In[34]:


predictedmodel = TrainLogisticRegression(X_train,y_train,X_valid,y_valid)


# In[35]:


yvalid_predict = predictedmodel.predict(X_valid)
print(classification_report(y_valid, yvalid_predict))

