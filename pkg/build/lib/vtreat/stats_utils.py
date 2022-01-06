"""util for basic statistical steps"""

from typing import Tuple

import numpy

import scipy.stats
import scipy.optimize
import sklearn.linear_model


# methods to avoid calling statsmodels which seems to be incompatible with many
# versions of other packages we need:
#  https://github.com/WinVector/pyvtreat/issues/14


def our_corr_score(*, y_true, y_pred) -> Tuple[float, float]:
    """
    Compute Pearson correlation. Case-out some corner cases.

    :param y_true: truth values
    :param y_pred: predictions
    :return: (pearson r, significance)
    """

    if not isinstance(y_true, numpy.ndarray):
        y_true = numpy.asarray(y_true)
    if not isinstance(y_pred, numpy.ndarray):
        y_pred = numpy.asarray(y_pred)
    n = len(y_true)
    if n < 2:
        return 1, 1
    if numpy.min(y_true) >= numpy.max(y_true):
        return 1, 1
    if numpy.min(y_pred) >= numpy.max(y_pred):
        return 0, 1
    r, sig = scipy.stats.pearsonr(y_true, y_pred)
    if n < 3:
        sig = 1
    return r, sig


def est_deviance(*, y, est, epsilon: float = 1.0e-5) -> float:
    """
    Estimate the deviance

    :param y: truth values
    :param est: predictions
    :param epsilon: how close to get to 0 and 1
    :return: deviance estimate
    """

    if not isinstance(y, numpy.ndarray):
        y = numpy.asarray(y)
    if not isinstance(est, numpy.ndarray):
        est = numpy.asarray(est)
    est = numpy.minimum(est, 1 - epsilon)
    est = numpy.maximum(est, epsilon)
    deviance = -2 * numpy.sum(y * numpy.log(est) + (1 - y) * numpy.log(1 - est))
    return deviance


def sklearn_solve_logistic(*, y, x, regularization: float = 1.0e-6):
    """
    Single variable logistic regression.
    Assumes special cases of solve_logistic_regression already eliminated.

    :param y: dependent variable
    :param x: explanatory variable
    :param regularization:
    :return: model predictions
    """

    if not isinstance(y, numpy.ndarray):
        y = numpy.asarray(y)
    if not isinstance(x, numpy.ndarray):
        x = numpy.asarray(x)
    fitter = sklearn.linear_model.LogisticRegression(
        penalty="l2", solver="lbfgs", fit_intercept=True, C=1 / regularization
    )
    dependent_vars = x.reshape((len(y), 1))
    fitter.fit(X=dependent_vars, y=y)
    preds = fitter.predict_proba(X=dependent_vars)[:, 1]
    return preds


# x, y - numpy numeric vectors, y 0/1.  solve for y- return predictions
def solve_logistic_regression(*, y, x):
    """
    Single variable logistic regression. Returns predictions, corner
    cases removed.

    :param y: dependent variable
    :param x: explanatory variable
    :return: predictions
    """

    if not isinstance(y, numpy.ndarray):
        y = numpy.asarray(y)
    if not isinstance(x, numpy.ndarray):
        x = numpy.asarray(x)
    # catch some corner cases
    n = len(y)
    if (n < 2) or (numpy.min(y) >= numpy.max(y)):
        return y.copy()
    if numpy.min(x) >= numpy.max(x):
        return numpy.asarray([numpy.mean(y)] * n)
    # check for fully separable cases
    big_y_indices = y > 0
    x_b = x[big_y_indices]
    x_s = x[numpy.logical_not(big_y_indices)]
    if (numpy.min(x_b) > numpy.max(x_s)) or (numpy.max(x_b) < numpy.min(x_s)):
        r = numpy.zeros(n)
        r[big_y_indices] = 1
        return r
    # run a full logistic regression
    preds = sklearn_solve_logistic(y=y, x=x)
    return numpy.asarray(preds)


# noinspection PyPep8Naming
def our_pseudo_R2(*, y_true, y_pred) -> Tuple[float, float]:
    """
    Return the logistic pseudo-R2

    :param y_true: dependent variable
    :param y_pred: explanatory variable
    :return: (pseudo-R2, significance)
    """

    if not isinstance(y_true, numpy.ndarray):
        y_true = numpy.asarray(y_true)
    if not isinstance(y_pred, numpy.ndarray):
        y_pred = numpy.asarray(y_pred)
    n = len(y_true)
    if n < 2:
        return 1, 1
    if numpy.min(y_true) >= numpy.max(y_true):
        return 1, 1
    if numpy.min(y_pred) >= numpy.max(y_pred):
        return 0, 1
    preds = solve_logistic_regression(y=y_true, x=y_pred)
    deviance = est_deviance(y=y_true, est=preds)
    null_deviance = est_deviance(y=y_true, est=numpy.zeros(n) + numpy.mean(y_true))
    r2 = 1 - deviance / null_deviance
    sig = 1
    if n >= 3:
        # https://github.com/WinVector/sigr/blob/master/R/ChiSqTest.R
        df_null = n - 1
        df_residual = n - 2
        delta_deviance = null_deviance - deviance
        delta_df = df_null - df_residual
        sig = 1 - scipy.stats.chi2.cdf(x=delta_deviance, df=delta_df)
    return r2, sig
