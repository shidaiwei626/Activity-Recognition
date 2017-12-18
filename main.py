import pandas as pd
import numpy as np

from sklearn.feature_selection import SelectFromModel
from sklearn.svm import SVC

from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.metrics import classification_report, accuracy_score

def read_train(file_name='input/train.csv'):
    print('Reading in training set...')
    train_df = pd.read_csv(file_name)
    print('Training set dimensionality: (%i, %i).\n' % train_df.shape)

    return train_df

def read_test(file_name='input/'):
    print('Reading in testing set...')
    test_df = pd.read_csv('input/test.csv')
    print('Testing set dimensionality: (%i, %i).\n' % test_df.shape)

    return test_df

print('Performing SVC prediction...')
print('Training...')

def param_select(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=13)
    print('Training set shape:', X_train.shape, y_train.shape)
    print('Testing set shape:', X_test.shape, y_test.shape)

    parameters = {
        'kernel': ['linear'],
        'C': [20, 1, 0.5]
    }

    print('Tuning hyper-parameters...')
    selector = GridSearchCV(SVC(), parameters, scoring='accuracy')
    selector.fit(X_train, y_train)

    print('Best parameter set found on development set:')
    print(selector.best_params_)
    print('Grid scores on development set:')
    means = selector.cv_results_['mean_test_score']
    stds = selector.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, selector.cv_results_['params']):
        print('%0.3f (+/-%0.03f) for %r' % (mean, std * 2, params))
        print()

    print('Detailed classification report:')
    print("The model is trained on the full development set.")
    print("The scores are computed on the full evaluation set.")
    y_true = y_test
    y_pred = selector.predict(X_test)
    print(classification_report(y_true, y_pred))

def predict(clf):
    test_df = read_test()
    X_test, y_test = test_df.iloc[:, 0:len(test_df.columns) -1], test_df.iloc[:, -1]

    print('Predicting...')
    y_pred = clf.predict(X_test)
    print('Accuracy score:', accuracy_score(y_test, y_pred))

if __name__ == '__main__':
    train_df = read_train()
    X, y = train_df.iloc[:, 0:len(train_df.columns) - 1], train_df.iloc[:, -1]

    param_select(X, y)

    #clf = SVC(kernel='linear', C=1).fit(X, y)
    #predict(clf)
