# for general organization
import pandas as pd
import numpy as np

# for feature selection
from sklearn.feature_selection import VarianceThreshold, SelectKBest, RFE, SelectFromModel
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import ExtraTreesClassifier

# for predicting group 1
'''from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier'''

# for predicting group 2
'''from sklearn.neighbors import KNeighborsClassifier'''

# for predicting group 3
'''from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier'''

# for predicting group 4
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

# for evaluation
from sklearn.metrics import accuracy_score
from timeit import default_timer as timer

def read_input():
    print('Reading training set...')
    train_df = pd.read_csv('input/train.csv')
    print('Training set dimensionality: (%i, %i).\n' % train_df.shape)

    return train_df

def read_result():
    print('Reading testing set...')
    test_df = pd.read_csv('input/test.csv')
    print('Testing set dimensionality: (%i, %i).\n' % test_df.shape)

    return test_df

# method_id:
# 1 --> variance threshold
# 2 --> univariate feature selection
# 3 --> recursive feature elimination TODO: currently takes too long
# 4 --> L1-based feature selection
# 5 --> tree-based feature selection
def feature_select(X, y, method_id):
    # apparently higher p results in lower number of features dropped
    def remove_low_variance(p, X=X):
        print('Removing features with low variance, p is %f.' % p)
        print('Initial shape of X: (%i, %i).' % X.shape)

        selector = VarianceThreshold(threshold=p*(1-p)).fit(X)
        keep_columns = [X.columns[i] for i in range(len(X.columns)) if selector.get_support()[i]]
        X_new = X[keep_columns]
        print('Resulting shape of X: (%i, %i).' % X_new.shape)

        return X_new

    def univariate_select(k, X=X, y=y):
        print('Performing univariate selection, k is %i.' % k)
        print('Initial shape of X: (%i, %i).' % X.shape)

        # score function:
        # not using chi2 because of negative values
        # not using mutual_info_classif because takes too long
        selector = SelectKBest(k=k).fit(X, y)
        keep_columns = [X.columns[i] for i in range(len(X.columns)) if selector.get_support()[i]]
        X_new = X[keep_columns]
        print('Resulting shape of X: (%i, %i).' % X_new.shape)

        return X_new

    def recursive_eliminate(k, X=X, y=y):
        '''print('Recursively eliminating features, k is %i.' % k)
        print('Initial shape of X: (%i, %i).' % X.shape)

        clf = SVC(kernel='linear')
        selector = RFE(clf, k, step=1)
        X_new = selector.fit_transform(X, y)
        print('Resulting shape of X: (%i, %i).' % X_new.shape)

        return X_new'''
        return

    def l1_select(X=X, y=y):
        print('Performing L1-based selection.')
        print('Initial shape of X: (%i, %i).' % X.shape)

        lsvc = LinearSVC(C=0.01, penalty='l1', dual=False).fit(X, y)
        model = SelectFromModel(lsvc, prefit=True)
        keep_columns = [X.columns[i] for i in range(len(X.columns)) if model.get_support()[i]]
        X_new = X[keep_columns]
        print('Resulting shape of X: (%i, %i).' % X_new.shape)

        return X_new

    def tree_select(X=X, y=y):
        print('Performing tree-based feature selection.')
        print('Initial shape of X: (%i, %i).' % X.shape)

        clf = ExtraTreesClassifier().fit(X, y)
        model = SelectFromModel(clf, prefit=True)
        keep_columns = [X.columns[i] for i in range(len(X.columns)) if model.get_support()[i]]
        X_new = X[keep_columns]
        print('Resulting shape of X: (%i, %i).' % X_new.shape)

        return X_new

    if method_id == 1:
        X_new = remove_low_variance(0.8) # 0.8 results in 212
    elif method_id == 2:
        X_new = univariate_select(100)
    elif method_id == 3:
        X_new = recursive_eliminate(100)
    elif method_id == 4:
        X_new = l1_select() # result in 110 features
    elif method_id == 5:
        X_new = tree_select() # result in 92 features
    else:
        print('Please use correct method ID (1-5).')
        exit()

    return X_new

# using:
# decision tree
# random forest
# gradient boosting
def predict_and_evaluate(X_train, y_train):
    classifiers = [
        # predicting group 1
        #DecisionTreeClassifier(),
        #RandomForestClassifier(n_estimators=200),
        #GradientBoostingClassifier(n_estimators=200),

        # predicting group 2
        #KNeighborsClassifier(7),
        #SVC(kernel='linear', C=0.025),
        #SVC(gamma=2, C=1),

        # predicting group 3
        #AdaBoostClassifier(),
        #GaussianNB(),
        #MLPClassifier(),

        # predicting group 4
        QuadraticDiscriminantAnalysis(),
    ]

    test_df = read_result()
    X_test, y_test = test_df.iloc[:, 0:len(test_df.columns) -1], test_df.iloc[:, -1]

    # without feature selection
    '''time_0 = []
    models_0 = []
    accuracy_0 = []
    for clf in classifiers:
        print('Performing prediction with %s...' % clf.__class__.__name__)
        start = timer()
        fitted_clf = clf.fit(X_train, y_train)
        prediction = fitted_clf.predict(X_test)
        duration = timer() - start
        time_0.append(duration)
        models_0.append(clf.__class__.__name__)
        accuracy_0.append(accuracy_score(y_test, prediction))

        print('Current time list:')
        print(time_0)
        print('Current model list:')
        print(models_0)
        print('Current accuracy list:')
        print(accuracy_0)
        print()'''

    # try different feature selection methods
    # change method_id to change method
    method_id = 5
    time_1 = []
    models_1 = []
    accuracy_1 = []
    X_new = feature_select(X_train, y_train, method_id)
    keep_columns = X_new.columns
    for clf in classifiers:
        print('Performing prediction with %s...' % clf.__class__.__name__)
        start = timer()
        fitted_clf = clf.fit(X_new, y_train)
        prediction = fitted_clf.predict(X_test[keep_columns])
        duration = timer() - start
        time_1.append(duration)
        models_1.append(clf.__class__.__name__)
        accuracy_1.append(accuracy_score(y_test, prediction))

        print('Current time list:')
        print(time_1)
        print('Current model list:')
        print(models_1)
        print('Current accuracy list:')
        print(accuracy_1)
        print()

if __name__ == '__main__':
    train_df = read_input()
    X_train, y_train = train_df.iloc[:, 0:len(train_df.columns) - 1], train_df.iloc[:, -1]

    for method_id in [1, 2, 4, 5]:
        print('Writing %i...' % method_id)
        X_new = feature_select(X_train, y_train, method_id)
        X_new.to_csv('input/train_feature_sel/train_v%i.csv' % method_id, index=False)

    #y_train.to_csv('input/train_feature_sel/labels.csv')

    #predict_and_evaluate(X_train, y_train)

    print('Finished.')
