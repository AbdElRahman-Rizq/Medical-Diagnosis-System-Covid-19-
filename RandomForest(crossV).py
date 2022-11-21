# Import libraries and modules
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.datasets import make_blobs
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
data = pd.read_csv ('G:Covid Dataset.csv')
X = data.iloc[:, :20]
y = data.iloc[:, 20]

# Split data



#from sklearn.preprocessing import StandardScaler
#sc = StandardScaler()
#X_train = sc.fit_transform(X_train)
#X_test = sc.transform(X_test)




classifier = RandomForestClassifier(n_estimators = 100)

scores = cross_val_score(classifier, X, y, cv=5)

print('RandomForest scores.mean: ',scores.mean() )
print('RandomForest scores.std: ',scores.std() )
