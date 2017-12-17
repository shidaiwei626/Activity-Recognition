from xgboost import XGBClassifier
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score

print('Loading data...')
train_df = pd.read_csv('input/train.csv')
test_df = pd.read_csv('input/test.csv')

X_train, y_train = train_df.iloc[:, 0: -1].values, train_df.iloc[:, -1].values
X_test, y_test = test_df.iloc[:, 0: -1].values, test_df.iloc[:, -1].values

transformation_dict = {
    'WALKING': 0,
    'WALKING_UPSTAIRS': 1,
    'WALKING_DOWNSTAIRS': 2,
    'SITTING': 3,
    'STANDING': 4,
    'LAYING': 5
}

y_train = np.vectorize(transformation_dict.get)(y_train)
y_test = np.vectorize(transformation_dict.get)(y_test)

print('Start training...')
clf = XGBClassifier().fit(X_train, y_train)

print('Start predicting...')
y_pred = clf.predict(X_test)
print('Accuracy:', accuracy_score(y_test, y_pred))
