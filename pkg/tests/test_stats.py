
import numpy

import vtreat.stats_utils

def test_linear_cor():
    y_true = [1, 1, 0, 1, 0, 1, 1, 0, 1, 0]
    y_pred = [0.8, 1, 0.2, 0.5, 0.5, 0.8, 1, 0.2, 0.5, 0.5]
    cor, sig = vtreat.stats_utils.our_corr_score(y_true=y_true, y_pred=y_pred)
    # R:
    # y_true = c(1, 1, 0, 1, 0, 1, 1, 0, 1, 0)
    # y_pred = c(0.8, 1, 0.2, 0.5, 0.5, 0.8, 1, 0.2, 0.5, 0.5)
    # cor.test(y_true, y_pred)
    # # t = 3.1159, df = 8, p-value = 0.01432
    # #       cor
    # # 0.7404361
    # summary(lm(y_true ~ y_pred))
    # Multiple R-squared:  0.5482,	Adjusted R-squared:  0.4918
    # F-statistic: 9.709 on 1 and 8 DF,  p-value: 0.01432
    assert numpy.abs(cor - 0.7404361) < 1.0e-2
    assert numpy.abs(cor*cor - 0.5482) < 1.0e-2
    assert numpy.abs(sig - 0.01432)  < 1.0e-2

    cor, sig = vtreat.stats_utils.our_corr_score(y_true=[1], y_pred=[0])
    expect_cor, expect_sig = (1, 1)
    assert numpy.abs(cor - expect_cor) < 1.0e-3
    assert numpy.abs(sig - expect_sig) < 1.0e-3

    cor, sig = vtreat.stats_utils.our_corr_score(y_true=[0, 0, 0], y_pred=[0, 1, 0])
    expect_cor, expect_sig = (1, 1)
    assert numpy.abs(cor - expect_cor) < 1.0e-3
    assert numpy.abs(sig - expect_sig) < 1.0e-3

    cor, sig = vtreat.stats_utils.our_corr_score(y_true=[1, 1, 1], y_pred=[0, 1, 0])
    expect_cor, expect_sig = (1, 1)
    assert numpy.abs(cor - expect_cor) < 1.0e-3
    assert numpy.abs(sig - expect_sig) < 1.0e-3

    cor, sig = vtreat.stats_utils.our_corr_score(y_true=[0, 1, 0], y_pred=[1, 1, 1])
    expect_cor, expect_sig = (0, 1)
    assert numpy.abs(cor - expect_cor) < 1.0e-3
    assert numpy.abs(sig - expect_sig) < 1.0e-3

    cor, sig = vtreat.stats_utils.our_corr_score(y_true=[0, 1, 0], y_pred=[0, 0, 0])
    expect_cor, expect_sig = (0, 1)
    assert numpy.abs(cor - expect_cor) < 1.0e-3
    assert numpy.abs(sig - expect_sig) < 1.0e-3


def test_solve_logistic():
    soln = vtreat.stats_utils.solve_logistic_regression(y=[1], x=[1])
    expect = [1]
    assert numpy.max(numpy.abs(numpy.asarray(expect) - soln)) < 1.0e-3

    soln = vtreat.stats_utils.solve_logistic_regression(y=[0], x=[1])
    expect = [0]
    assert numpy.max(numpy.abs(numpy.asarray(expect) - soln)) < 1.0e-3

    soln = vtreat.stats_utils.solve_logistic_regression(y=[1, 0, 1, 0, 0], x=[1, 1, 1, 1, 1])
    expect = [0.4, 0.4, 0.4, 0.4, 0.4]
    assert numpy.max(numpy.abs(numpy.asarray(expect) - soln)) < 1.0e-3

    soln = vtreat.stats_utils.solve_logistic_regression(y=[1, 0, 1, 0, 0], x=[0, 0, 0, 0, 0])
    expect = [0.4, 0.4, 0.4, 0.4, 0.4]
    assert numpy.max(numpy.abs(numpy.asarray(expect) - soln)) < 1.0e-3

    soln = vtreat.stats_utils.solve_logistic_regression(y=[1, 1, 1, 1, 1], x=[1, 0, 1, 0, 0])
    expect = [1, 1, 1, 1, 1]
    assert numpy.max(numpy.abs(numpy.asarray(expect) - soln)) < 1.0e-3

    soln = vtreat.stats_utils.solve_logistic_regression(y=[0, 0, 0, 0, 0], x=[1, 0, 1, 0, 0])
    expect = [0, 0, 0, 0, 0]
    assert numpy.max(numpy.abs(numpy.asarray(expect) - soln)) < 1.0e-3

    soln = vtreat.stats_utils.solve_logistic_regression(y=[1, 0, 1, 0, 0], x=[1, 0, 1, 0, 0])
    expect = [1, 0, 1, 0, 0]
    assert numpy.max(numpy.abs(numpy.asarray(expect) - soln)) < 1.0e-3

    soln = vtreat.stats_utils.solve_logistic_regression(y=[1, 0, 1, 0, 0], x=[1, 0, 1, 0, 0])
    expect = [1, 0, 1, 0, 0]
    assert numpy.max(numpy.abs(numpy.asarray(expect) - soln)) < 1.0e-3

    soln = vtreat.stats_utils.solve_logistic_regression(y=[1, 0, 1, 0, 0], x=[1, 0, 1, 0, 1])
    expect = [6.66662683e-01, 1.64791483e-05, 6.66662683e-01, 1.64791483e-05,
       6.66662683e-01]
    assert numpy.max(numpy.abs(numpy.asarray(expect) - soln)) < 1.0e-2


def test_est_dev():
    est = vtreat.stats_utils.est_deviance(y=[1, 0, 1, 1, 0, 0], est=[1, 0.2, 0, 0.5, 0.2, 0.3])
    expect = 26.018089384294647
    assert numpy.abs(expect - est) < 1.0e-2


def test_brute_logistic():
    y_true = [1, 1, 0, 0, 0, 1, 1, 0, 1, 1]
    y_pred = [0.8, 1, 1, 0.5, 0.5, 0.8, 1, 0.2, 0.5, 0.5]
    soln = vtreat.stats_utils.brute_force_solve_logistic(y=y_true, x=y_pred)
    expect = [0.70953711, 0.8298342 , 0.8298342 , 0.46410274, 0.46410274,
       0.70953711, 0.8298342 , 0.23490658, 0.46410274, 0.46410274]
    assert numpy.max(numpy.abs(numpy.asarray(expect) - soln)) < 1.0e-2

    if vtreat.stats_utils.have_sklearn:
        soln2 = vtreat.stats_utils.sklearn_solve_logistic(y=y_true, x=y_pred)
        assert numpy.max(numpy.abs(numpy.asarray(expect) - soln2)) < 1.0e-2


def test_logistic_r2():
    cor, sig = vtreat.stats_utils.our_pseudo_R2(y_true=[1], y_pred=[0])
    expect_cor, expect_sig = (1, 1)
    assert numpy.abs(cor - expect_cor) < 1.0e-3
    assert numpy.abs(sig - expect_sig) < 1.0e-3

    cor, sig = vtreat.stats_utils.our_pseudo_R2(y_true=[1, 0, 1, 0, 0], y_pred=[1, 1, 1, 1, 1])
    expect_cor, expect_sig = (0, 1)
    assert numpy.abs(cor - expect_cor) < 1.0e-3
    assert numpy.abs(sig - expect_sig) < 1.0e-3

    cor, sig = vtreat.stats_utils.our_pseudo_R2(y_true=[1, 1, 1, 1, 1], y_pred=[1, 0, 1, 0, 0])
    expect_cor, expect_sig = (1, 1)
    assert numpy.abs(cor - expect_cor) < 1.0e-3
    assert numpy.abs(sig - expect_sig) < 1.0e-3

    y_true = [1, 1, 0, 0, 0, 1, 1, 0, 1, 1]
    y_pred = [0.8, 1, 1, 0.5, 0.5, 0.8, 1, 0.2, 0.5, 0.5]
    # R:
    # y_true = c(1, 1, 0, 0, 0, 1, 1, 0, 1, 1)
    # y_pred = c(0.8, 1, 1, 0.5, 0.5, 0.8, 1, 0.2, 0.5, 0.5)
    # (s <- summary(glm(y_true ~ y_pred, family = binomial())))
    #     Null deviance: 13.460  on 9  degrees of freedom
    # Residual deviance: 11.762  on 8  degrees of freedom
    # (w <- sigr::wrapChiSqTest(s))
    # Chi-Square Test summary: pseudo-R2=0.1262 (X2(1,N=10)=1.698, p=n.s.).
    # w$pValue
    # [1] 0.1925211
    check_r2 = 1 - 11.762/13.460
    r2, sig = vtreat.stats_utils.our_pseudo_R2(y_true=y_true, y_pred=y_pred)
    assert numpy.abs(r2 - check_r2) < 1.0e-2
    assert numpy.abs(r2 - 0.1262) < 1.0e-2
    assert numpy.abs(sig - 0.1925211) < 1.0e-2

