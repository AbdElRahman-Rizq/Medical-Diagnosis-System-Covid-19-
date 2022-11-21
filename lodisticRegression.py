import pandas as pd

data = pd.read_csv ('G:Dataset.csv')
X = data.iloc[:, :22].values
y = data.iloc[:, 22].values
#df = pd.DataFrame(data,columns=['Fever','Tiredness','Dry-Cough','Difficulty-in-Breathing','Sore-Throat','None_Sympton','Pains','Nasal-Congestion','Runny-Nose','Diarrhea','None_Experiencing','Age_0-9','Age_10-19','Age_20-24','Age_25-59','Age_60+','Gender_Female','Gender_Male','Gender_Transgender','Contact_Dont-Know','Contact_No','Contact_Yes','Severity'])

#*reg=linear_model.LinearRegression()
#reg.fit(data[[ 'Fever','Tiredness',	'Dry-Cough','Difficulty-in-Breathing','Sore-Throat','None_Sympton','Pains','Nasal-Congestion','Runny-Nose','Diarrhea','None_Experiencing','Age_0-9','Age_10-19','Age_20-24','Age_25-59','Age_60+','Gender_Female','Gender_Male','Gender_Transgender','Contact_Dont-Know','Contact_No','Contact_Yes',
#]],data.Severity)
#from sklearn.datasets import load_iris

#/////////////
#Import Libraries
from sklearn.preprocessing import Binarizer
#----------------------------------------------------

#Binarizing Data

scaler = Binarizer(threshold = 1.0)
X = scaler.fit_transform(X)
#Scalling
#Data_Scaling
from sklearn.datasets import load_digits
from sklearn.feature_selection import SelectKBest, chi2
X, y = load_digits(return_X_y=True)
X.shape

X_new = SelectKBest(chi2, k=30).fit_transform(X, y)

X_new.shape


#EndOf_FeatureSelection
#/////////////

from sklearn.linear_model import LogisticRegressionCV
#X, y = load_iris(return_X_y=True)
clf = LogisticRegressionCV(cv=4, random_state=0).fit(X, y)
clf.predict(X[:22, :])

clf.predict_proba(X[:22, :]).shape

clf.score(X, y)
print(data.shape)