import numpy as np
import pandas as pd


class EffectScaler:
    """
    Effect scaler. First step of Y-Aware PCA (ref: https://win-vector.com/2022/09/08/y-aware-pca/ )
    or Y-Aware L2-regularization.
    """

    def _clear(self) -> None:
        self._n_columns = None
        self._colnames = None
        self._x_means = None
        self._x_scales = None

    def __init__(self):
        self._clear()

    # noinspection PyPep8Naming
    def fit(self, X, y, sample_weight=None) -> None:
        """
        Get per-variable effect scaling of (X[:, i] - np.mean(X[:, i])) -> (y - np.mean(y)).
        See https://win-vector.com/2022/09/08/y-aware-pca/

        :param X: explanatory values
        :param y: dependent values
        :return: self for method chaining
        """
        assert sample_weight is None  # TODO: implement non-None case
        assert len(X.shape) == 2
        assert X.shape[0] > 0
        assert X.shape[1] > 0
        y = np.array(y, float)
        y = y - np.mean(y)
        y_sq = np.dot(y, y)
        assert len(y.shape) == 1
        assert y.shape[0] == X.shape[0]
        self._clear()
        self._n_columns = X.shape[1]
        self._x_means = np.zeros(self._n_columns)
        self._x_scales = np.zeros(self._n_columns)

        def calc_mean_and_scale(i: int, *, xi: np.ndarray) -> None:
            self._x_means[i] = np.mean(xi)
            if y_sq > 0:
                xi = xi - self._x_means[i]
                xi_sq = np.dot(xi, xi)
                if xi_sq > 0:
                    self._x_scales[i] = np.dot(xi, y) / np.dot(xi, xi)

        if isinstance(X, pd.DataFrame):
            self._colnames = list(X.columns)
            for i in range(self._n_columns):
                xi = np.array(X.iloc[:, i], float)
                calc_mean_and_scale(i, xi=xi)
        else:
            for i in range(self._n_columns):
                xi = np.array(X[:, i], float)
                calc_mean_and_scale(i, xi=xi)
        return self

    # noinspection PyPep8Naming
    def transform(self, X) -> pd.DataFrame:
        """
        Transform date based on previous fit.

        :param X: explanatory values
        :return: transformed data
        """
        assert self._n_columns is not None  # make sure we are fit
        assert len(X.shape) == 2
        assert X.shape[1] == self._n_columns

        def transform_col(i: int, *, xi: np.ndarray) -> np.ndarray:
            xi = (xi - self._x_means[i]) * self._x_scales[i]
            return xi

        if isinstance(X, pd.DataFrame):
            if self._colnames is not None:
                assert list(X.columns) == self._colnames
            return pd.DataFrame(
                {
                    c: transform_col(i, xi=np.array(X.loc[:, c], float))
                    for i, c in zip(range(X.shape[1]), X.columns)
                }
            )
        else:
            return pd.DataFrame(
                {
                    i: transform_col(i, xi=np.array(X[:, i], float))
                    for i in range(X.shape[1])
                }
            )

    # noinspection PyPep8Naming
    def predict(self, X) -> pd.DataFrame:
        return self.transform(X)

    # noinspection PyPep8Naming
    def fit_transform(self, X, y, sample_weight=None) -> pd.DataFrame:
        """
        Fit and transform combined. Not computed out of sample.

        :param X: explanatory values
        :param y: dependent values
        :return: transformed data
        """

        self.fit(X, y, sample_weight=sample_weight)
        return self.transform(X)

    # noinspection PyPep8Naming
    def fit_predict(self, X, y, sample_weight=None) -> pd.DataFrame:
        return self.fit_transform(X, y, sample_weight=sample_weight)
