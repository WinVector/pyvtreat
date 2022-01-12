"""util for basic statistical steps"""

from typing import Tuple

import numpy
import pandas

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


def xicor(xvec, yvec, *, n_reps: int = 5) -> Tuple[float, float]:
    """
    xicor calculation built to match from R::caclulateXI() and published article.

    :param xvec: numeric vector with explanatory variable to compute xicor for.
    :param yvec: numeric dependent variable to relate to.
    :param n_reps: number of times to repeat calculation.
    :return: mean and standard error of estimate (under x-tie breaking)
    """
    yvec = numpy.asarray(yvec)
    n = len(yvec)
    assert n > 1
    assert isinstance(n_reps, int)
    assert n_reps > 0
    xvec = numpy.asarray(xvec)
    assert n == len(xvec)
    PI = numpy.zeros(n)
    fr_orig = scipy.stats.rankdata(yvec, method="max") / n
    gr = scipy.stats.rankdata(-yvec, method="max") / n
    CU = numpy.mean(gr * (1 - gr))
    xi_s = numpy.zeros(n_reps)
    for rep_i in range(n_reps):
        perm = numpy.random.permutation(n)  # get random ranking by ranking permuted
        PI_inv = scipy.stats.rankdata(xvec[perm], method="ordinal")
        PI[perm] = PI_inv  # invert permutation, self assignment fails
        ord = numpy.argsort(PI)
        fr = fr_orig[ord]
        A1 = numpy.sum(numpy.abs(fr[0:(n - 1)] - fr[1:n])) / (2 * n)
        xi = 1 - A1 / CU
        xi_s[rep_i] = xi
    return numpy.mean(xi_s), numpy.std(xi_s) / numpy.sqrt(n_reps)


def xicor_for_frame(d: pandas.DataFrame, y, *, n_reps=5):
    """
    Calculate xicor for all columns of data frame d with respect to dependent column y.

    :param d: data frame of proposed explanatory variables.
    :param y: vector of the dependent variable values.
    :param n_reps: number of times to repeat experiment (positive integer)
    :return: data frame with: variable (name of column), xicor (estimated xicor statistic),
             xicor_se (standard error of xicor estimate, goes to zero as n_reps grows),
             xicor_perm_mean (mean value of xicor with y scrambled, goes to zero as n_reps grows),
             xicor_perm_stddev (sample standard deviation of y scrambled xicor estimates,
             used to form z or t style estimates).
    """
    n = d.shape[0]
    assert n > 1
    y = numpy.asarray(y)
    assert len(y) == n
    assert isinstance(n_reps, int)
    assert n_reps > 0
    res = pandas.DataFrame({
        'variable': d.columns,
        'xicor': 0.0,
        'xicor_se': 0.0,
        'xicor_perm_mean': 0.0,
        'xicor_perm_stddev': 0.0,
        'xicor_perm_sum': 0.0,
        'xicor_perm_sum_sq': 0.0,
    })
    # get the xicor estimates
    for col_i in range(len(d.columns)):
        xvec = d[d.columns[col_i]]
        xi_est, xi_est_dev = xicor(xvec, y, n_reps=n_reps)
        res.loc[col_i, 'xicor'] = xi_est
        res.loc[col_i, 'xicor_se'] = xi_est_dev
    # score all x-columns with the same y-permutation
    # estimate stddev with expanding squares to cut down storage
    for rep_j in range(n_reps):
        y_perm = y[numpy.random.permutation(n)]
        for col_i in range(len(d.columns)):
            xvec = d[d.columns[col_i]]
            xi_perm, _ = xicor(xvec, y_perm, n_reps=1)
            res.loc[col_i, 'xicor_perm_sum'] = res.loc[col_i, 'xicor_perm_sum'] + xi_perm
            res.loc[col_i, 'xicor_perm_sum_sq'] = res.loc[col_i, 'xicor_perm_sum_sq'] + xi_perm * xi_perm
    res['xicor_perm_mean'] = res['xicor_perm_sum'] / n_reps
    res['xicor_perm_stddev'] = numpy.sqrt((1 / (n_reps - 1)) * (
            res['xicor_perm_sum_sq'] - (1 / n_reps) * res['xicor_perm_sum']**2))
    del res['xicor_perm_sum']
    del res['xicor_perm_sum_sq']
    return res
