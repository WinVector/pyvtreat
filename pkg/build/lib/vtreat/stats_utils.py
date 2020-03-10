import numpy

import scipy.stats
import scipy.optimize

have_sklearn = False
# noinspection PyBroadException
try:
    import sklearn.linear_model

    have_sklearn = True
except Exception:
    pass


# methods to avoid calling statsmodels which seems to be incompatible with many
# versions of other packages we need:
#  https://github.com/WinVector/pyvtreat/issues/14


def our_corr_score(*, y_true, y_pred):
    # compute Pearson correlation
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


def est_deviance(*, y, est, epsilon=1.0e-5):
    if not isinstance(y, numpy.ndarray):
        y = numpy.asarray(y)
    if not isinstance(est, numpy.ndarray):
        x = numpy.asarray(est)
    est = numpy.minimum(est, 1 - epsilon)
    est = numpy.maximum(est, epsilon)
    deviance = -2 * numpy.sum(
        y * numpy.log(est) +
        (1 - y) * numpy.log(1 - est))
    return deviance


# assumes special cases of solve_logistic_regression already eliminated
def brute_force_solve_logistic(*, y, x, regularization=1.e-6):
    if not isinstance(y, numpy.ndarray):
        y = numpy.asarray(y)
    if not isinstance(x, numpy.ndarray):
        x = numpy.asarray(x)

    def predict(beta):
        return 1 / (1 + numpy.exp(-(beta[0] + beta[1] * x)))

    def loss(beta):
        preds = predict(beta)
        return (est_deviance(y=y, est=preds)
                + regularization * (beta[0] * beta[0] + beta[1] * beta[1])  # regularization
                )

    soln = scipy.optimize.fmin_bfgs(
        loss,
        x0=[numpy.log(numpy.mean(y) / (1 - numpy.mean(y))), 0.001],
        gtol=1.0e-3,
        disp=0)
    return predict(soln)


# assumes special cases of solve_logistic_regression already eliminated
def sklearn_solve_logistic(*, y, x, regularization=1.e-6):
    if not isinstance(y, numpy.ndarray):
        y = numpy.asarray(y)
    if not isinstance(x, numpy.ndarray):
        x = numpy.asarray(x)
    fitter = sklearn.linear_model.LogisticRegression(
        penalty='l2',
        solver='lbfgs',
        fit_intercept=True,
        C=1/regularization)
    dependent_vars = x.reshape((len(y), 1))
    fitter.fit(X=dependent_vars, y=y)
    preds = fitter.predict_proba(X=dependent_vars)[:, 1]
    return preds


# x, y - numpy numeric vectors, y 0/1.  solve for y- return predictions
def solve_logistic_regression(*, y, x):
    # catch some corner cases
    if not isinstance(y, numpy.ndarray):
        y = numpy.asarray(y)
    if not isinstance(x, numpy.ndarray):
        x = numpy.asarray(x)
    n = len(y)
    if (n < 2) or (numpy.min(y) >= numpy.max(y)):
        return y.copy()
    if numpy.min(x) >= numpy.max(x):
        return numpy.asarray([numpy.mean(y)] * n)
    # check for fully seperable cases
    big_y_indices = y > 0
    x_b = x[big_y_indices]
    x_s = x[numpy.logical_not(big_y_indices)]
    if (min(x_b) > max(x_s)) or (max(x_b) < min(x_s)):
        r = numpy.zeros(n)
        r[big_y_indices] = 1
        return r
    # run a full logistic regression
    if have_sklearn:
        preds = sklearn_solve_logistic(y=y, x=x)
    else:
        preds = brute_force_solve_logistic(y=y, x=x)
    return numpy.asarray(preds)


# noinspection PyPep8Naming
def our_pseudo_R2(*, y_true, y_pred):
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
