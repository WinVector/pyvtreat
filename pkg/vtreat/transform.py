

import pandas


class UserTransform:
    """base class for user transforms"""

    def __init__(self, incoming_column_name, derived_column_names, treatment):
        self.incoming_column_name_ = incoming_column_name
        self.derived_column_names_ = derived_column_names.copy()
        self.treatment_ = treatment
        self.need_cross_treatment_ = True

    def fit(self, X, y):
        raise Exception("base method called")

    def transform(self, X):
        raise Exception("base method called")

    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform(X)
