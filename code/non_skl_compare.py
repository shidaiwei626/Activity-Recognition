import pandas as pd
import numpy as np

import lightgbm as lgb
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

print('Loading data...')
y_train = pd.read_csv('input/train_feature_sel/labels.csv', header=None, index_col=0).values#; print(y_train)
test_df = pd.read_csv('input/test.csv')

transformation_dict = {
    'WALKING': 0,
    'WALKING_UPSTAIRS': 1,
    'WALKING_DOWNSTAIRS': 2,
    'SITTING': 3,
    'STANDING': 4,
    'LAYING': 5
}

y_train = np.vectorize(transformation_dict.get)(y_train)

lgb_params = {
    'boosting_type': 'gbdt',
    'objective': 'multiclass',
    'num_class': 6,
    'verbose': 0,
}

for method_id in [1, 2, 4, 5]:
    print('Starting reading training set from method %i...' % method_id)
    X_train = pd.read_csv('input/train_feature_sel/train_v%i.csv' % method_id)
    temp_train_df = X_train
    temp_train_df['Activity'] = y_train
    temp_test_df = test_df[temp_train_df.columns]

    X_train, y_train = temp_train_df.iloc[:, 0: -1].values, temp_train_df.iloc[:, -1].values
    X_test, y_test = temp_test_df.iloc[:, 0: -1].values, temp_test_df.iloc[:, -1].values
    y_test = np.vectorize(transformation_dict.get)(y_test)

    lgb_train = lgb.Dataset(X_train, y_train)
    gbm = lgb.train(lgb_params, lgb_train)
    predicted_prob = gbm.predict(X_test)
    y_pred = np.argmax(predicted_prob, axis=1)
    print('Accuracy for LightGBM:', accuracy_score(y_test, y_pred))

    xgb_clf = clf = XGBClassifier().fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print('Accuracy for XGBoost:', accuracy_score(y_test, y_pred))
