# -*- coding: utf-8 -*-
"""
Created on Thu Nov  2 20:32:21 2017

@author: linkw
"""

import pandas as pd
import numpy as np

#read indto df. take care of missing values and combine text.
data=pd.read_csv("D:\\Datasets\\kickstarter\\train.csv")
data.iloc[:,2].replace(np.NAN, '---', inplace=True)
data.iloc[:,4].replace(np.NAN, '---', inplace=True)
data.iloc[:,2] = data.iloc[:,2] + " " + data.iloc[:,4]
#data=data.dropna()
#data=data.dropna(subset=['desc'])

#clean txt data. Get rid of dummy's and non words
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
stopwords=stopwords.words('english')
stemmer=SnowballStemmer('english')
import re
r=re.compile(r'[\W]', re.U)
data.iloc[:,2]=data.iloc[:,2].apply(lambda x : ' '.join(stemmer.stem(w) for w in re.sub('[\\s]+',' ',r.sub(' ',x.lower())).split() if w not in stopwords))
data=data.assign(txt_len=data.iloc[:,2].apply(lambda x : len(x.split())).values)

#get rid of trivial stuff. Also calculate time diffs
txt_data=data.iloc[:,2].values[...,None]
X=data.iloc[:,[3,5,6,8,10,11,14]].values
Y=data.iloc[:,13].values
launch_to_deadline=np.subtract(X[:,3],X[:,5])[...,None]
launch_to_deadline=np.divide(launch_to_deadline,86400)
creation_to_deadline=np.subtract(X[:,3],X[:,4])[...,None]
creation_to_deadline=np.divide(creation_to_deadline,86400)
creation_to_launch=np.subtract(X[:,5],X[:,4])[...,None]
creation_to_launch=np.divide(creation_to_launch,86400)
X=np.concatenate((X,launch_to_deadline,creation_to_deadline,creation_to_launch),1)
X=np.delete(X, 3,1)
X=np.delete(X, 3,1)
X=np.delete(X, 3,1)

#encode values
from sklearn.preprocessing import LabelEncoder
labelencoder_1 = LabelEncoder()
X[:,1]=labelencoder_1.fit_transform(X[:,1])
labelencoder_2 = LabelEncoder()
labelencoder_2.fit(X[:,2])
#take care of unknown labels during prediction phase
import bisect
countries = labelencoder_2.classes_.tolist()
bisect.insort_left(countries, 'unknown')
labelencoder_2.classes_ = countries
X[:,2]=labelencoder_2.transform(X[:,2])
#1H
from sklearn.preprocessing import OneHotEncoder
onehotencoder = OneHotEncoder(categorical_features=[1,2],n_values=[2,12])
X=onehotencoder.fit_transform(X).toarray()
X=np.concatenate((X,txt_data),1)

#unfortunately do an early split to hide test data from feature transformers
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X, Y, train_size=0.9,random_state=141289)

del X,Y,txt_data,creation_to_deadline,creation_to_launch,launch_to_deadline,countries

#oops backers are not there in test data of the challenge. gotta find another way
txt=X_train[:,19]
txt2=X_test[:,19]
X_train=np.delete(X_train,19,1)
X_test=np.delete(X_test,19,1)

#get ready to clusterize text
from sklearn.feature_extraction.text import CountVectorizer
cvt = CountVectorizer()
text_features = cvt.fit_transform(txt)
text_features2 = cvt.transform(txt2)

from sklearn.feature_extraction.text import TfidfTransformer
tf = TfidfTransformer(use_idf=False)
text_features= tf.fit_transform(text_features)
text_features2= tf.transform(text_features2)

#dr
from sklearn.decomposition import TruncatedSVD
tsvd= TruncatedSVD(n_components= 450,n_iter=12, random_state=141289)
text_features=tsvd.fit_transform(text_features)
#print(tsvd.explained_variance_ratio_.sum())
text_features2=tsvd.transform(text_features2)

#normalize for use in K means
from sklearn.preprocessing import Normalizer
nrm= Normalizer()
nrm.fit(text_features)
text_features=nrm.transform(text_features)
text_features2=nrm.transform(text_features2)

#clusterize text for use as a feature
from sklearn.cluster import KMeans
km = KMeans(n_clusters=18, n_jobs=8, algorithm='full')
clusters=km.fit_predict(text_features)[...,None]
cluster_test=km.predict(text_features2)[...,None]

#1h clusters to be used with XGB
onehotencoder2 = OneHotEncoder()
clusters=onehotencoder2.fit_transform(clusters).toarray()
cluster_test=onehotencoder2.transform(cluster_test).toarray()

X_train=np.concatenate((X_train,clusters,text_features),1)
X_test=np.concatenate((X_test,cluster_test,text_features2),1)

del clusters,cluster_test,text_features,text_features2,txt,txt2

#from sklearn.ensemble import RandomForestClassifier
#rf = RandomForestClassifier(n_estimators = 85, criterion = 'entropy', random_state = 141289,n_jobs=8)
#rf.fit(X_train, Y_train)
#Y_pred = rf.predict(X_test)

from xgboost import XGBClassifier
xgb = XGBClassifier(random_state=141289,n_jobs=8,max_depth=5,n_estimators=245,subsample=0.9,colsample_bytree=0.9)
xgb.fit(X_train, Y_train)
Y_pred = xgb.predict(X_test)
Y_pred=[round(k) for k in Y_pred]

#check accuracy of model
from sklearn.metrics import accuracy_score
print(accuracy_score(Y_test, Y_pred))

#cleanup
del X_test,X_train,Y_test,Y_train,Y_pred,data

###########################################################################################
#predict the test dataset for submission.
test_data=pd.read_csv("D:\\Datasets\\kickstarter\\test.csv")
test_data.iloc[:,2].replace(np.NAN, '---', inplace=True)
test_data.iloc[:,4].replace(np.NAN, '---', inplace=True)
test_data.iloc[:,2] = test_data.iloc[:,2] + " " + test_data.iloc[:,4]
test_data.iloc[:,6] = test_data.iloc[:,6].map(lambda x: 'unknown' if x not in labelencoder_2.classes_ else x)
test_data.iloc[:,2]=test_data.iloc[:,2].apply(lambda x : ' '.join(stemmer.stem(w) for w in re.sub('[\\s]+',' ',r.sub(' ',x.lower())).split() if w not in stopwords))
test_data=test_data.assign(txt_len=test_data.iloc[:,2].apply(lambda x : len(x.split())).values)
txt_data=test_data.iloc[:,2].values[...,None]
pred_x=test_data.iloc[:,[3,5,6,8,10,11,12]].values
launch_to_deadline=np.subtract(pred_x[:,3],pred_x[:,5])[...,None]
launch_to_deadline=np.divide(launch_to_deadline,86400)
creation_to_deadline=np.subtract(pred_x[:,3],pred_x[:,4])[...,None]
creation_to_deadline=np.divide(creation_to_deadline,86400)
creation_to_launch=np.subtract(pred_x[:,5],pred_x[:,4])[...,None]
creation_to_launch=np.divide(creation_to_launch,86400)
pred_x=np.concatenate((pred_x,launch_to_deadline,creation_to_deadline,creation_to_launch),1)
pred_x=np.delete(pred_x, 3,1)
pred_x=np.delete(pred_x, 3,1)
pred_x=np.delete(pred_x, 3,1)
pred_x[:,1]=labelencoder_1.transform(pred_x[:,1])
pred_x[:,2]=labelencoder_2.transform(pred_x[:,2])
pred_x=onehotencoder.transform(pred_x).toarray()
pred_x=np.concatenate((pred_x,txt_data),1)
txt2=pred_x[:,19]
pred_x=np.delete(pred_x,19,1)
text_features2 = cvt.transform(txt2)
text_features2= tf.transform(text_features2)
text_features2=tsvd.transform(text_features2)
text_features2=nrm.transform(text_features2)
cluster_test=km.predict(text_features2)[...,None]

#1h clusters for use with XGB
cluster_test=onehotencoder2.transform(cluster_test).toarray()

pred_x=np.concatenate((pred_x,cluster_test,text_features2),1)
#predictions = rf.predict(pred_x)
predictions = xgb.predict(pred_x)
predictions=[round(k) for k in predictions]

#save csv for upload
test_data=test_data.assign(final_status=predictions)
sub=test_data.iloc[:,[0,13]]
#sub.to_csv("D:\\Datasets\\kickstarter\\subrf.csv",index=False)
sub.to_csv("D:\\Datasets\\kickstarter\\subxgb.csv",index=False)

#cleanup
del txt_data,creation_to_deadline,creation_to_launch,launch_to_deadline,txt2, text_features2,cluster_test,pred_x,predictions,test_data,sub