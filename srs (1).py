#!/usr/bin/env python
# coding: utf-8

# In[6]:


import numpy as np
import pandas as pd
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import seaborn as sns
color = sns.color_palette()
sns.set_style('darkgrid')
import warnings
warnings.filterwarnings('ignore') # отключаем сообщения об ошибках
import matplotlib


# In[12]:


iris_data = pd.read_csv("Iris.csv")
iris_data.head(3)


# In[13]:


target = iris_data["Species"]
iris_data.drop("Species", axis=1, inplace = True)


# In[14]:


print("Типы данных в колонках {}".format(iris_data.dtypes.unique()))
print("размерность данных {}".format(iris_data.shape))
iris_data.describe(percentiles=[])


# In[15]:


from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder


# In[16]:


# создаем синтетический набор данных
X, y = make_blobs(random_state=0)

plt.scatter(x=X[:,0],y=X[:,1], 
            color = ['red' if l == 0 else ('blue' if l==1 else 'green') for l in y])


# In[17]:


# разобъем данные на обучающий и тестовый наборы
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
# создаем экземпляр модели и подгоняем его на обучающем наборе
logreg = LogisticRegression().fit(X_train, y_train)
# оцениваем качество модели на тестовом наборе
print("Правильность на тестовом наборе: {:.2f}".format(logreg.score(X_test, y_test)))


# In[18]:


plt.scatter(x=X_train[:,0],y=X_train[:,1], 
            color = ['red' if l == 0 else ('blue' if l==1 else 'green') for l in y_train])


# In[19]:


plt.scatter(x=X_test[:,0],y=X_test[:,1], 
            color = ['red' if l == 0 else ('blue' if l==1 else 'green') for l in y_test])


# In[23]:


X_train, X_test, y_train, y_test =     train_test_split(iris_data, target, random_state=42, train_size=0.8)
assert X_train.shape[0] + X_test.shape[0] == iris_data.shape[0]
print ("train size={}, test_size={}, total_size={}"        .format(X_train.shape[0], X_test.shape[0], iris_data.shape[0]))


# In[24]:


from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import cross_val_score

log_model = LogisticRegression().fit(X_train, y_train)
#y_pred = log_model.predict(X_test)

scores = cross_val_score(log_model, iris_data, target)
print("accuracy: {}".format(scores))


# In[25]:


scores = cross_val_score(log_model, iris_data, target, cv=5)
print("accuracy: {}".format(scores))


# In[26]:


print("mean accuracy: {:.2f}".format(scores.mean()))


# In[27]:


target.head(3)


# In[28]:


# перекодируем метки классов
encoded_target = LabelEncoder().fit_transform(target)
encoded_target


# In[44]:


from sklearn.model_selection import LeaveOneOut
loo = LeaveOneOut()
scores = cross_val_score(log_model, iris_data, target, cv=loo)
print("Количество итераций cv: ", len(scores))
print("Mean accuracy: {:.2f}".format(scores.mean()))


# In[ ]:


'''Чтобы оценить изменение обобщающей способности, используется метрика качества модели. Метрика качества должна быть выбрана до этапа подготовки данных и обучения модели.
Простой способ оценить обобщающую способность модели - отложенный контроль (hold-out).

Мы разбиваем обучающую выборку случайным образом на 2 подвыборки - обучающую и контрольную. Обычно используется соотношение 80 % (обучение) на 20% (контроль).

В библиотеке sklearn для разбиения выборки используем функцию train_test_split.

Для решения задачи классификации будем использовать логистическую регрессию


# In[ ]:





# In[ ]:




