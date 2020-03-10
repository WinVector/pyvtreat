
import numpy

import vtreat.stats_utils

def test_linear_cor():
    y_true = [1, 1, 0, 1, 0, 1, 1, 0, 1, 0]
    y_pred = [0.8, 1, 0.2, 0.5, 0.5, 0.8, 1, 0.2, 0.5, 0.5]
    cor, sig = vtreat.stats_utils.our_corr_score(y_true=y_true, y_pred=y_pred)
    # R:
    # y_true = c(1, 1, 0, 1, 0, 1, 1, 0, 1, 0)
    # y_pred = c(0.8, 1, 0.2, 0.5, 0.5, 0.8, 1, 0.2, 0.5, 0.5)
    # summary(lm(y_true ~ y_pred))
    # Multiple R-squared:  0.5482,	Adjusted R-squared:  0.4918
    # F-statistic: 9.709 on 1 and 8 DF,  p-value: 0.01432
    assert numpy.abs(cor*cor - 0.5482) < 1.0e-2
    assert numpy.abs(sig - 0.01432)  < 1.0e-2



def test_logistic_r2():
    if not vtreat.stats_utils.have_sklearn:
        return
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
    pass
