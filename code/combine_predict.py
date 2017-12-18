import pandas as pd
import numpy as np

'''from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

import lightgbm as lgb
from xgboost import XGBClassifier

prediction_df = pd.DataFrame()

train_df = pd.read_csv('input/train.csv')
test_df = pd.read_csv('input/test.csv')

X_train, y_train = train_df.iloc[:, 0:len(train_df.columns) - 1], train_df.iloc[:, -1]
X_test, y_test = test_df.iloc[:, 0:len(test_df.columns) -1], test_df.iloc[:, -1]

# train models that do best without feature selection
without_feature_sel = [
    RandomForestClassifier(n_estimators=200),
    GradientBoostingClassifier(n_estimators=200),
    SVC(kernel='linear', C=0.025),
]

for clf in without_feature_sel:
    print(clf.__class__.__name__)
    fitted_clf = clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    prediction_df[clf.__class__.__name__] = y_pred

# train models that do better with L1-based feature selection
with_l1_based = [
    MLPClassifier(),
    QuadraticDiscriminantAnalysis(),
]

X_new = pd.read_csv('input/train_feature_sel/train_v4.csv')

for clf in with_l1_based:
    print(clf.__class__.__name__)
    fitted_clf = clf.fit(X_new, y_train)
    y_pred = clf.predict(X_test[X_new.columns])

    prediction_df[clf.__class__.__name__] = y_pred

print(prediction_df)

# LightGBM and XGBoost stuff
label_to_num = {
    'WALKING': 0,
    'WALKING_UPSTAIRS': 1,
    'WALKING_DOWNSTAIRS': 2,
    'SITTING': 3,
    'STANDING': 4,
    'LAYING': 5
}

num_to_label = {
    0: 'WALKING',
    1: 'WALKING_UPSTAIRS',
    2: 'WALKING_DOWNSTAIRS',
    3: 'SITTING',
    4: 'STANDING',
    5: 'LAYING'
}

y_train_num = np.vectorize(label_to_num.get)(y_train)
y_test_num = np.vectorize(label_to_num.get)(y_test)

# train LightGBM
print('LightGBM')
lgb_train = lgb.Dataset(X_train, y_train_num)

params = {
    'boosting_type': 'gbdt',
    'objective': 'multiclass',
    'num_class': 6,
    'verbose': 0,
}

gbm = lgb.train(params, lgb_train)

predicted_prob = gbm.predict(X_test)
y_pred = np.argmax(predicted_prob, axis=1)

prediction_df['LightGBM'] = np.vectorize(num_to_label.get)(y_pred)

# train XGBoost
print('XGBoost')
clf = XGBClassifier().fit(X_train, y_train_num)
y_pred = clf.predict(X_test)

prediction_df['XGBoost'] = np.vectorize(num_to_label.get)(y_pred)


print('Writing to file...')
prediction_df.to_csv('output/combine_v1.csv')'''



from sklearn.metrics import accuracy_score

prediction_df = pd.read_csv('output/combine_v1.csv', index_col=0)
prediction_df['Mode'] = prediction_df.mode(axis=1)[0]
print(prediction_df)

test_df = pd.read_csv('input/test.csv')
y_test = test_df.iloc[:, -1]

print(accuracy_score(y_test, prediction_df['Mode']))
