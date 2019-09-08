class UserTransform:
    """base class for user transforms, should express taking a set of k inputs to k outputs independently"""

    def __init__(self, treatment):
        self.y_aware_ = True
        self.treatment_ = treatment
        self.incoming_vars_ = []
        self.derived_vars_ = []

    # noinspection PyPep8Naming
    def fit(self, X, y):
        raise NotImplementedError("base method called")

    # noinspection PyPep8Naming
    def transform(self, X):
        raise NotImplementedError("base method called")

    # noinspection PyPep8Naming
    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform(X)
