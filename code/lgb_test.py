import lightgbm as lgb
import pandas as pd
import numpy as np; np.random.seed(13)
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

lgb_train = lgb.Dataset(X_train, y_train)

params = {
    'boosting_type': 'gbdt',
    'objective': 'multiclass',
    #'objective': 'multiclassova',
    'num_class': 6,
    'verbose': 0,
}

print('Start training...')
gbm = lgb.train(params, lgb_train)

#print('Saving model...')
#gbm.save_model('model.txt')

print('Start predicting...')
predicted_prob = gbm.predict(X_test)
y_pred = np.argmax(predicted_prob, axis=1)
print('Accuracy:', accuracy_score(y_test, y_pred))
