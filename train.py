#!/usr/bin/env python
# coding: utf-8

# *Karim Ladouani examen final A57

# In[1]:


# Lecture et creation du dataframe


# In[2]:


import pandas as pd
import numpy as np


# In[4]:


df = pd.read_csv('pointure.data')
df.head()


# In[5]:


import numpy as np
from sklearn import preprocessing
label_encoder = preprocessing.LabelEncoder()
input_classes = ['masculin','féminin']
label_encoder.fit(input_classes)

# transformer un ensemble de classes
encoded_labels = label_encoder.transform(df['Genre'])
print(encoded_labels)
df['Genre'] = encoded_labels

df.head()


# In[6]:


# Creation du X,Y
X = df.iloc[:, lambda df: [1, 2, 3]]
y = df.iloc[:, 0]


# In[19]:


# #Split des données
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=44)


# In[28]:


from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
def apprendre_mlflow(i):
#    mlflow.set_experiment(experiment_name='Experiment')
 #   mlflow.set_tracking_uri("http://localhost:5000")
  #  with mlflow.start_run():
   gnb = GaussianNB()
# Entrainer
   gnb.fit(X_train, y_train)
# Predire
   y_naive_bayes1 = gnb.predict(X_train)
# Métriques
   accuracy = metrics.accuracy_score(y_train, y_naive_bayes1)
   print("Accuracy du modele Naive Bayes predit: " + str(accuracy))
        #mlflow.log_param('Accuracy', accuracy)

   recall_score = metrics.recall_score(y_train, y_naive_bayes1)
   print("recall score du modele Naive Bayes predit: " + str(recall_score))
      #  mlflow.log_param('recall_score', recall_score)
    
   f1_score = metrics.f1_score(y_train, y_naive_bayes1)
   print("F1 score du modele Naive Bayes predit: " + str(f1_score))
       # mlflow.log_param('f1_score', f1_score)
       # mlflow.sklearn.log_model(gnb, "model")
   return gnb


# In[29]:


for i in range(3):
    print('Experience', i)
    gnb = apprendre_mlflow(i)


# In[30]:


# Prediction d'une valeur
d = {'Taille(cm)':[183], 'Poids(kg)':[59], 'Pointure(cm)':[20]}
dfToPredict = pd.DataFrame(data=d)
yPredict = gnb.predict(dfToPredict)
print('La classe predite est : ', yPredict)


# In[ ]:




