import pandas as pd
import numpy as np
import json


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

    def predict(self, x):
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


if __name__ == '__main__':
    import pandas as pd
    train_data = pd.read_csv('data.csv', index_col=0)

    model = Naive_Bayse(feature_type='discrete')
    model.fit(train_data[['factor1', 'factor2']], train_data['RTN'])
    pred = model.predict(train_data[['factor1', 'factor2']])
    print(pred)
