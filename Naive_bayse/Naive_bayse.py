from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
import math
import json

'''
Machine Learning Model-Naive Bayse
feature_type: represent the type of features
'continuous': if features are continuous.
'discrete': if features are discrete.
'''


class Naive_Bayse():
    def __init__(self, feature_type=None):
        self.X = None
        self.y = None
        self.data = None
        self.features = None
        self.parameter = {}
        self.prior_prob = {}
        self.classse = []
        if feature_type:
            self.feature_type = feature_type
        else:
            self.feature_type = 'continuous'

    def fit(self, X, y):
        self.X = X
        self.y = y
        self.data = self.X.copy()
        self.data['class'] = self.y
        self.features = self.X.columns
        self.classse = self.y.unique()
        self._compute_prior_prob()
        if self.feature_type == 'discrete':
            self._construct_model_discrete()
        else:
            self._construct_model_continuous()

    '''
    compute the prior probability.
    '''
    def _compute_prior_prob(self):
        for each_class in self.classse:
            self.prior_prob[each_class] = (self.y[self.y == each_class].shape[0] + 1) / (self.y.shape[0] + len(self.classse))

    def _construct_model_discrete(self):
        for each_feature in self.features:
            for each_value in np.unique(self.X[each_feature]):
                for each_class in self.classse:
                    numerator = self.data[(self.data[each_feature] == each_value)
                                          & (self.data['class'] == each_class)].shape[0] + 1
                    denominator = self.data[self.data['class'] == each_class].shape[0] \
                                  + len(np.unique(self.X[each_feature]))
                    self.parameter[each_feature + '=' + str(each_value) + '|' + 'label=' + str(each_class)] \
                        = numerator / denominator

    def _construct_model_continuous(self):
        for each_feature in self.features:
            self.parameter[each_feature] = {}
            for each_class in self.classse:
                self.parameter[each_feature][each_class] = {}
                self.parameter[each_feature][each_class]['mean'] \
                    = np.mean(self.data[self.data['class'] == each_class][each_feature])
                self.parameter[each_feature][each_class]['var'] = np.var(self.data[self.data['class'] == each_class][each_feature])

    def _compute_gaussian(self, x, mean, var):
        return np.exp(-1 * math.pow(x - mean, 2) / (2 * var)) / (math.sqrt(2 * math.pi * var))

    def predict(self, x):
        if self.feature_type == 'discrete':
            def _predict_helper(_x):
                score = {}
                label = None
                max_score = 0
                for each_class in self.classse:
                    score[each_class] = self.prior_prob[each_class]
                    for each_feature in self.features:
                        score[each_class] *= self.parameter[each_feature + '=' + str(_x[each_feature])
                                                            + '|' + 'label=' + str(each_class)]
                    if score[each_class] > max_score:
                        label = each_class
                        max_score = score[each_class]
                return label

            return x.apply(_predict_helper, axis=1)
        else:
            def _predict_helper(_x):
                score = {}
                label = None
                max_score = 0
                for each_class in self.classse:
                    score[each_class] = self.prior_prob[each_class]
                    for each_feature in self.features:
                        score[each_class] *= self._compute_gaussian(_x[each_feature],
                                                                    self.parameter[each_feature][each_class]['mean'],
                                                                    self.parameter[each_feature][each_class]['var'])
                    if score[each_class] > max_score:
                        label = each_class
                        max_score = score[each_class]
                return label
            return x.apply(_predict_helper, axis=1)

    def score(self):
        pred = self.predict(self.X)
        return accuracy_score(self.y, pred)
