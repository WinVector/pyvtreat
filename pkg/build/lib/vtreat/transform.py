

class UserTransform:
    """base class for user transforms, should express taking a set of k inputs to k outputs independently"""

    def __init__(self, treatment):
        self.y_aware_ = True
        self.treatment_ = treatment
        self.incoming_vars_ = []
        self.derived_vars_ = []

    def fit(self, X, y):
        raise Exception("base method called")

    def transform(self, X):
        raise Exception("base method called")

    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform(X)
