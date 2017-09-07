import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


def divide_features(train_data, feature, threshold):
    if isinstance(threshold, int) or isinstance(threshold, float):
        x_1 = train_data[train_data[feature] >= threshold]
        x_2 = train_data[train_data[feature] < threshold]
    else:
        x_1 = train_data[train_data[feature] == threshold]
        x_2 = train_data[train_data[feature] != threshold]
    return x_1, x_2


def calculate_entropy(y):
    values = y['label'].unique()
    entropy = 0
    for each in values:
        prob = y[y['label'] == each].shape[0] / y.shape[0]
        entropy += -prob * np.log2(prob)
    return entropy


def calculate_info_gain(y, y1, y2):
    prob = len(y1) / len(y)
    entropy = calculate_entropy(y)
    info_gain = entropy - prob * calculate_entropy(y1) - (1 - prob) * calculate_entropy(y2)
    return info_gain


def majority_vote(y):
    most_common = 0
    max_count = 0
    for label in y['label'].unique():
        count = y[y['label'] == label].shape[0]
        if count > max_count:
            max_count = count
            most_common = label
    return most_common


def accuracy_score(y_true, y_pred):
    return np.sum(y_true == y_pred) / y_true.shape[0]


def plot_2d(X, y):
    model = PCA(n_components=2)
    X = pd.DataFrame(model.fit_transform(X), columns=['features_1', 'features_2'], index=y.index)
    plt.figure()
    cmap = plt.get_cmap('viridis')
    colors = [cmap(i) for i in np.linspace(0, 1, y['label'].unique().shape[0])]
    for i, each_value in enumerate(y['label'].unique()):
        x_1 = X.loc[y[y['label'] == each_value].index]['features_1']
        x_2 = X.loc[y[y['label'] == each_value].index]['features_2']
        plt.scatter(x_1, x_2, color=colors[i])
    plt.show()


"""
Decision Node is the class for tree node.
"""


class Decision_Node():
    """
    Class that represents a decision node or leaf in the decision tree

    Parameters:
    -----------
    feature_index: int
        Feature index which we want to use as the threshold measure.
    threshold: float
        The value that we will compare feature values at feature_i against to
        determine the prediction.
    value: float
        The class prediction if classification tree, or float value if regression tree.
    true_branch: DecisionNode
        Next decision node for samples where features value met the threshold.
    false_branch: DecisionNode
        Next decision node for samples where features value did not meet the threshold.
    """
    def __init__(self, feature_index=None, threshold=None, true_branch=None, false_branch=None, value=None):
        self.feature_index = feature_index
        self.threshold = threshold
        self.true_branch = true_branch
        self.false_branch = false_branch
        self.value = value


class Decision_tree(object):
    def __init__(self, min_samples_split=2, min_impurity=1e-7,
                 max_depth=100, loss=None):
        self.min_samples_split = min_samples_split
        self.min_impurity = min_impurity
        self.max_depth = max_depth
        print(self.max_depth)
        self.loss = loss
        self.root = None
        # Function to calculate impurity (classif.=>info gain, regr=>variance reduct.)
        self._impurity_calculation = calculate_info_gain
        # Function to determine prediction of y at leaf
        self._leaf_value_calculation = majority_vote
        # If Gradient Boost
        self.loss = loss

    def fit(self, X, y, loss=None):
        self.root = self._build_tree(X, y)
        self.loss = loss

    def _build_tree(self, X, y, current_depth=0):
        largest_impurity = 0

        n_samples = X.shape[0]
        n_features = X.columns
        best_criteria = {}
        best_sets = {}

        if n_samples >= self.min_samples_split and current_depth < self.max_depth:
            for each_features in n_features:

                values = X[each_features].unique()

                for each_threshold in values:
                    X_1, X_2 = divide_features(X, each_features, each_threshold)

                    if X_1.shape[0] > 0 and X_2.shape[0] > 0:
                        y1 = y.loc[X_1.index]
                        y2 = y.loc[X_2.index]

                        impurity = self._impurity_calculation(y, y1, y2)

                        if impurity > largest_impurity:
                            best_criteria = {
                                "feature_index": each_features, "threshold": each_threshold}
                            best_sets = {
                                "left_X": X_1,
                                "left_y": y1,
                                "right_X": X_2,
                                "right_y": y2
                            }
                            largest_impurity = impurity

        if largest_impurity >= self.min_impurity:
            true_branch = self._build_tree(best_sets['left_X'], best_sets['left_y'], current_depth + 1)
            false_branch = self._build_tree(best_sets['right_X'], best_sets['right_y'], current_depth + 1)
            return Decision_Node(best_criteria['feature_index'], best_criteria['threshold'],
                                 true_branch, false_branch, None)
        leaf_value = self._leaf_value_calculation(y)
        return Decision_Node(value=leaf_value)

    def predict_value(self, X, root=None):
        if root is None:
            root = self.root

        if root.value is not None:
            return root.value

        feature_value = X[root.feature_index]
        branch = None
        if feature_value >= root.threshold:
            branch = root.true_branch
        else:
            branch = root.false_branch

        return self.predict_value(X, branch)

    def predict(self, X):
        return X.apply(lambda x: self.predict_value(x), axis=1)


def main():
    print("-- Classification Tree --")

    data = datasets.load_iris()
    X = data.data
    y = data.target

    X = pd.DataFrame(X, columns=['features_1', 'features_2', 'features_3', 'features_4'])
    y = pd.DataFrame(y, columns=['label'])

    plot_2d(X, y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)

    clf = Decision_tree(max_depth=20)
    clf.fit(X_train, y_train)
    y_pred_train = clf.predict(X_train)
    y_pred_test = clf.predict(X_test)
    print(accuracy_score(pd.Series(y_train['label']), y_pred_train))
    print(accuracy_score(pd.Series(y_test['label']), y_pred_test))

if __name__ == '__main__':
    main()
