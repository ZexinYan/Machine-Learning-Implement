import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno


class DataDescription:
    """
    Record the useful function which can be used to describe data.
    """
    def __init__(self, data, paths):
        self.data = data
        self.paths = paths

    def head(self, n):
        """
        Display the head 'n' rows of data.
        :param n: rows
        :return: None
        """
        print(self.data.head(n))

    def tail(self, n):
        """
        Display the tail 'n' rows of data.
        :param n: rows
        :return: None
        """
        print(self.data.tail(n))

    def description(self):
        """
        Display the description of data.
        :return: None
        """
        print(self.data.describe())

    def showMissValue(self):
        """
        Show the ratio of missing value of each features.
        It will not save fig.
        :return: None
        """
        msno.bar(self.data)

    def showDistribution(self):
        """
        Show the distribution of each features.
        For continuous data, it will just show the plot of data, which can be easily classified the distribution.
        For concrete data, it will show the bar of data.
        :return: Saved images.
        """
        self.data.hist()
        plt.show()
        plt.savefig('Total Distribution')

        for each in self.data.columns:
            if self.data[each].dtype is np.dtype(float) or self.data[each].dtype is np.dtype(int):
                sns.distplot(self.data[each], kde=True)
                plt.savefig(self.paths + '/' + each + ' distribution')
                plt.show()
            else:
                values = data[each].unique()
                nums = []
                for value in values:
                    nums.append(data[data[each] == value].shape[0])
                plt.bar(np.arange(len(values)), nums, color='y', align='center')
                plt.xticks(np.arange(len(values)), values)
                plt.savefig(self.paths + '/' + each + ' distribution')
                plt.show()

    def showBoxPlot(self):
        """
        Show the box plot of each feature which is continuous.
        Save the figs.
        :return: None
        """
        self.data.boxplot()
        plt.savefig(self.paths + '/' + 'Total BoxPlot')
        plt.show()

        for each in self.data.columns:
            if self.data[each].dtype is np.dtype(int) or self.data[each].dtype is np.dtype(float):
                sns.boxplot(self.data[each])
                plt.savefig(self.paths + '/' + each + ' BoxPlot')
                plt.show()

if __name__ == '__main__':
    data = pd.read_excel('消费贷款数据.xlsx')
    describer = DataDescription(data, '.')
    describer.head(10)
    describer.tail(10)
    describer.description()
    describer.showMissValue()
    describer.showDistribution()
    describer.showBoxPlot()
