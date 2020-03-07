
# From:
# https://github.com/WinVector/pyvtreat/blob/master/Examples/UserCoders/UserCoders.ipynb


import pandas
import numpy
import numpy.random
# import seaborn

import vtreat
import vtreat.util
import vtreat.transform

have_sklearn = True
try:
    import sklearn.linear_model
    import sklearn
except Exception:
    have_sklean = False


def test_user_coders():
    sklearn.warnings.filterwarnings('ignore')

    # avoid depending on sklearn.metrics.r2_score
    def r_squared(*, y_true, y_pred):
        y_true = numpy.asarray(y_true)
        y_pred = numpy.asarray(y_pred)
        return 1 - numpy.sum((y_true - y_pred)**2)/numpy.sum((y_true - numpy.mean(y_true))**2)

    # %%

    class PolyTransform(vtreat.transform.UserTransform):
        """a polynomial model"""

        def __init__(self, *, deg=5, alpha=0.1):
            vtreat.transform.UserTransform.__init__(self, treatment='poly')
            self.models_ = None
            self.deg = deg
            self.alpha = alpha

        def poly_terms(self, vname, vec):
            vec = numpy.asarray(vec)
            r = pandas.DataFrame({'x': vec})
            for d in range(1, self.deg + 1):
                r[vname + '_' + str(d)] = vec ** d
            return r

        def fit(self, X, y):
            self.models_ = {}
            self.incoming_vars_ = []
            self.derived_vars_ = []
            for v in X.columns:
                if vtreat.util.can_convert_v_to_numeric(X[v]):
                    X_v = self.poly_terms(v, X[v])
                    model_v = sklearn.linear_model.Ridge(alpha=self.alpha).fit(X_v, y)
                    new_var = v + "_poly"
                    self.models_[v] = (model_v, [c for c in X_v.columns], new_var)
                    self.incoming_vars_.append(v)
                    self.derived_vars_.append(new_var)
            return self

        def transform(self, X):
            r = pandas.DataFrame()
            for k, v in self.models_.items():
                model_k = v[0]
                cols_k = v[1]
                new_var = v[2]
                X_k = self.poly_terms(k, X[k])
                xform_k = model_k.predict(X_k)
                r[new_var] = xform_k
            return r

    # %%

    d = pandas.DataFrame({'x': [i for i in range(100)]})
    d['y'] = numpy.sin(0.2 * d['x']) + 0.2 * numpy.random.normal(size=d.shape[0])
    d.head()

    # %%

    step = PolyTransform(deg=10)

    # %%

    fit = step.fit_transform(d[['x']], d['y'])
    fit['x'] = d['x']
    fit.head()

    # %%

    # seaborn.scatterplot(x='x', y='y', data=d)
    # seaborn.lineplot(x='x', y='x_poly', data=fit, color='red', alpha=0.5)

    # %%

    transform = vtreat.NumericOutcomeTreatment(
        outcome_name='y',
        params=vtreat.vtreat_parameters({
            'filter_to_recommended': False,
            'user_transforms': [PolyTransform(deg=10)]
        }))

    # %%

    transform.fit(d, d['y'])

    # %%

    transform.score_frame_

    # %%

    x2_overfit = transform.transform(d)

    # %%
    # seaborn.scatterplot(x='x', y='y', data=x2_overfit)
    # seaborn.lineplot(x='x', y='x_poly', data=x2_overfit, color='red', alpha=0.5)

    # %%

    x2 = transform.fit_transform(d, d['y'])

    # %%

    transform.score_frame_

    # %%

    x2.head()

    # %%

    # seaborn.scatterplot(x='x', y='y', data=x2)
    # seaborn.lineplot(x='x', y='x_poly', data=x2, color='red', alpha=0.5)

    # %%


