import numpy

import scipy.stats

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
    y_true = numpy.asarray(y_true)
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


# x, y - numpy numeric vectors, y 0/1.  solve for y- return predictions
def solve_logistic_regression(*, y, x):
    fitter = sklearn.linear_model.LogisticRegression(
        penalty='l2',
        solver='lbfgs',
        fit_intercept=True,
        C=1000)
    dependent_vars = x.reshape((len(x), 1))
    fitter.fit(X=dependent_vars, y=y)
    preds = fitter.predict_proba(X=dependent_vars)[:, 1]
    return preds


# noinspection PyPep8Naming
def our_pseudo_R2(*, y_true, y_pred):
    if not have_sklearn:
        cor, sig = our_corr_score(y_true=y_true, y_pred=y_pred)
        return cor**2, sig
    # compute Pearson correlation
    y_true = numpy.asarray(y_true)
    y_pred = numpy.asarray(y_pred)
    n = len(y_true)
    if n < 2:
        return 1, 1
    if numpy.min(y_true) >= numpy.max(y_true):
        return 1, 1
    if numpy.min(y_pred) >= numpy.max(y_pred):
        return 0, 1

    preds = solve_logistic_regression(y=y_true, x=y_pred)
    eps = 1e-5
    preds = numpy.minimum(preds, 1-eps)
    preds = numpy.maximum(preds, eps)
    deviance = -2 * numpy.sum(y_true * numpy.log(preds) +\
                              (1 - y_true) * numpy.log(1 - preds))
    null_pred = numpy.zeros(n) + numpy.mean(y_true)
    null_deviance = -2 * numpy.sum(y_true * numpy.log(null_pred) +\
                                   (1 - y_true) * numpy.log(1 - null_pred))
    r2 = 1 - deviance/null_deviance
    sig = 1

    if n >= 3:
        # https://github.com/WinVector/sigr/blob/master/R/ChiSqTest.R
        df_null = n - 1
        df_residual = n - 2
        delta_deviance = null_deviance - deviance
        delta_df = df_null - df_residual
        sig = 1 - scipy.stats.chi2.cdf(x=delta_deviance, df=delta_df)

    return r2, sig
