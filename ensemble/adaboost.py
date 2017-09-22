import pandas as pd
import numpy as np
import math
from sklearn.metrics import accuracy_score


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
        self.direction = 1
        self.alpha = None
        self.feature_index = None
        self.threshold = None


class adaBoost:
    """
    adaboost model

    n_clf: int
    the num of weak decision function
    """
    def __init__(self, n_clf=10):
        self.n_clf = n_clf
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
                feature_values = X[feature_i]
                unique_feature = X[feature_i].unique()
                for _threshold in unique_feature:
                    direction = 1
                    pred = np.ones(n_samples)
                    pred[feature_values < _threshold] = -1
                    error = np.sum(weights[pred != y])
                    if error > 0.5:
                        error = 1 - error
                        direction = -1
                    if error <= min_error:
                        clf.direction = direction
                        clf.feature_index = feature_i
                        clf.threshold = _threshold
                        min_error = error

            clf.alpha = 0.5 * math.log((1 - min_error + 1e-10) / (min_error + 1e-10))
            pred = np.ones(n_samples)
            pred[clf.direction * X[clf.feature_index] < clf.direction * clf.threshold] = -1
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
        y_pred = np.zeros((n_samples, 1))

        for clf in self.clf_array:
            prediction = np.ones((n_samples, 1))
            prediction[(clf.direction * X[clf.feature_index] < clf.direction * clf.threshold)] = -1
            y_pred += prediction * clf.alpha

        return y_pred

    def score(self, X, y):
        pred = self.predict(X)
        return accuracy_score(y, pred)

    """
    :return weak functions array
    """
    def decision_function(self):
        return self.n_clf
