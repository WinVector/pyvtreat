"""base class for user transforms"""

import abc


class UserTransform(abc.ABC):
    """base class for user transforms, should express taking a set of k inputs to k outputs independently"""

    def __init__(self, treatment):
        self.y_aware_ = True
        self.treatment_ = treatment
        self.incoming_vars_ = []
        self.derived_vars_ = []

    # noinspection PyPep8Naming
    @abc.abstractmethod
    def fit(self, X, y):
        """
        sklearn API

        :param X: explanatory values
        :param y: dependent values
        :return: self for method chaining
        """

    # noinspection PyPep8Naming
    @abc.abstractmethod
    def transform(self, X):
        """
        sklearn API

        :param X: explanatory values
        :return: transformed data
        """

    # noinspection PyPep8Naming
    def fit_transform(self, X, y):
        """
        sklearn API

        :param X: explanatory values
        :param y: dependent values
        :return: transformed data
        """

        self.fit(X, y)
        return self.transform(X)
