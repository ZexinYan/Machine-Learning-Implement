import pandas as pd
import numpy as np
import math
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score


class DecisionStump:
    """
    Decision Stump
    Used to define weak decision function.
    :arg
    alpha: the importance of such feature
    feature_index: the index of selected feature
    threshold: the threshold of feature value
    """

    def __init__(self):
        self.alpha = None
        self.feature_index = None
        self.model = None


class adaBoost:
    """
    adaboost model

    n_clf: int
    the num of weak decision function
    """

    def __init__(self, n_clf=10, learning_rate=0.5):
        self.n_clf = n_clf
        self.learning_rate = learning_rate
        self.clf_array = []

    """
    fit function

    :arg
    X: pd.DataFrame(value must be discrete)
    y: pd.DataFrame(binary value)
    """

    def fit(self, X, y):
        n_samples, _ = X.shape
        features = X.columns

        y = np.array(y.values).flatten()
        weights = np.full(n_samples, 1 / n_samples)

        for _ in range(self.n_clf):
            clf = DecisionStump()
            min_error = 1

            for feature_i in features:
                feature_values = np.reshape(X[feature_i], (-1, 1))

                model = tree.DecisionTreeClassifier()
                model.fit(feature_values, y)
                pred = model.predict(feature_values)
                error = np.sum(weights[pred != y])

                if error <= min_error:
                    clf.feature_index = feature_i
                    min_error = error
                    clf.model = model

            clf.alpha = self.learning_rate * math.log((1 - min_error + 1e-10) / (min_error + 1e-10))
            pred = clf.model.predict(np.reshape(X[clf.feature_index], (-1, 1)))
            weights *= np.exp(-1 * clf.alpha * pred * y)
            weights /= np.sum(weights)
            self.clf_array.append(clf)

    """
    predict function

    :arg
    X: pd.DataFrame

    :return
    prediction: pd.DataFrame
    """

    def predict(self, X):
        return pd.DataFrame(np.sign(self._predict_helper(X)))

    """
    predict probability(not real probability but confidence)
    if probability is much larger than 1, 
    then it will be more confident that the predicted value will be positive.

    :arg
    X: pd.DataFrame    
    """

    def predict_prob(self, X):
        return pd.DataFrame(self._predict_helper(X))

    def _predict_helper(self, X):
        n_samples = X.shape[0]
        y_pred = np.zeros((1, n_samples))

        for clf in self.clf_array:
            prediction = clf.model.predict(np.reshape(X[clf.feature_index], (-1, 1)))
            y_pred += prediction * clf.alpha
        return y_pred[0]

    def score(self, X, y, mode='accuracy'):
        pred = self.predict(X)
        if mode == 'accuracy':
            return accuracy_score(y, pred)
        elif mode == 'f1':
            return f1_score(y, pred)

    """
    :return weak functions array
    """

    def decision_function(self):
        return self.clf_array
