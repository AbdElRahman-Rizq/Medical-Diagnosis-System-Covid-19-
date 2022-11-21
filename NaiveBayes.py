import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv('G:Dataset.csv')
X = data.iloc[:, :22].values
y = data.iloc[:, 22].values
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)



from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)


classifier.score(X_test, y_test)
y_pred = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

from sklearn.metrics import confusion_matrix,classification_report,accuracy_score

accuracy_score(y_test,y_pred, normalize=True)

#print(classification_report(y_test,y_pred,y))

y_pred[:30]
y_test[:30]

