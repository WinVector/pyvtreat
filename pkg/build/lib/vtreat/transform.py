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

    def __repr__(self):
        return ("vtreat.transform.UserTransform("
                + "treatment=" + self.treatment_.__repr__()
                + ") {"
                + "'y_aware_': " + str(self.y_aware_)
                + ", " + "'treatment_': " + str(self.treatment_)
                + ", " + "'incoming_vars_': " + str(self.incoming_vars_)
                + "}")

    def __str__(self):
        return self.__repr__()
