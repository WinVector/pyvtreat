From [pyvtreat issue 12](https://github.com/WinVector/pyvtreat/issues/12)


```python
import pandas as pd
import numpy as np
import numpy.random
import vtreat
import vtreat.util
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

numpy.random.seed(2019)

def make_data(nrows):
    d = pd.DataFrame({'x': 5*numpy.random.normal(size=nrows)})
    d['y'] = numpy.sin(d['x']) + 0.1*numpy.random.normal(size=nrows)
    d.loc[numpy.arange(3, 10), 'x'] = numpy.nan                           # introduce a nan level
    d['xc'] = ['level_' + str(5*numpy.round(yi/5, 1)) for yi in d['y']]
    d['x2'] = np.random.normal(size=nrows)
    d.loc[d['xc']=='level_-1.0', 'xc'] = numpy.nan  # introduce a nan level
    d['yc'] = d['y']>0.5
    return d

df = make_data(500)

df = df.drop(columns=['y'])

transform = vtreat.BinomialOutcomeTreatment(outcome_target=True)

clf = Pipeline(steps=[
    ('preprocessor', transform),
    ('classifier', LogisticRegression(solver = 'lbfgs'))]
)

X, y = df, df.pop('yc')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

clf.fit(X_train, y_train)

print("model score: %.3f" % clf.score(X_test, y_test))
```

    model score: 0.880



```python
clf.score(X_train, y_train)
```

    /Users/johnmount/opt/anaconda3/envs/ai_academy_3_7/lib/python3.7/site-packages/vtreat/vtreat_api.py:369: UserWarning: called transform on same data used to fit (this causes over-fit, please use fit_transform() instead)
      "called transform on same data used to fit (this causes over-fit, please use fit_transform() instead)")





    0.93



The above fit is an over-fit (not achievable without data leakage). Notice vtreat gave as a warning.


```python
print(clf)
```

    Pipeline(memory=None,
             steps=[('preprocessor',
                     vtreat.vtreat_api.BinomialOutcomeTreatment(outcome_target=True,
    params={'coders': {'clean_copy',
                'deviation_code',
                'impact_code',
                'indicator_code',
                'logit_code',
                'missing_indicator',
                'prevalence_code'},
     'cross_validation_k': 5,
     'cross_validation_plan': <vtreat.cross_plan.KWayCrossPlanYStratified object at 0x10fa81b50>,
     '...
     'missingness_imputation': <function mean at 0x11093bb90>,
     'sparse_indicators': True,
     'use_hierarchical_estimate': True,
     'user_transforms': []},
    )),
                    ('classifier',
                     LogisticRegression(C=1.0, class_weight=None, dual=False,
                                        fit_intercept=True, intercept_scaling=1,
                                        l1_ratio=None, max_iter=100,
                                        multi_class='warn', n_jobs=None,
                                        penalty='l2', random_state=None,
                                        solver='lbfgs', tol=0.0001, verbose=0,
                                        warm_start=False))],
             verbose=False)



```python
print(transform.get_feature_names())
```

    ['x_is_bad', 'xc_is_bad', 'x', 'x2', 'xc_logit_code', 'xc_prevalence_code', 'xc_lev_level_1_0', 'xc_lev__NA_', 'xc_lev_level_-0_5', 'xc_lev_level_0_5']



```python
print(transform.get_params())

```

    {'use_hierarchical_estimate': True, 'coders': {'prevalence_code', 'logit_code', 'indicator_code', 'deviation_code', 'impact_code', 'missing_indicator', 'clean_copy'}, 'filter_to_recommended': True, 'indicator_min_fraction': 0.1, 'cross_validation_plan': <vtreat.cross_plan.KWayCrossPlanYStratified object at 0x10fa81b50>, 'cross_validation_k': 5, 'user_transforms': [], 'sparse_indicators': True, 'missingness_imputation': <function mean at 0x11093bb90>, 'outcome_target': True}



```python
print(clf.get_params())

```

    {'memory': None, 'steps': [('preprocessor', vtreat.vtreat_api.BinomialOutcomeTreatment(outcome_target=True,
    params={'coders': {'clean_copy',
                'deviation_code',
                'impact_code',
                'indicator_code',
                'logit_code',
                'missing_indicator',
                'prevalence_code'},
     'cross_validation_k': 5,
     'cross_validation_plan': <vtreat.cross_plan.KWayCrossPlanYStratified object at 0x10fa81b50>,
     'filter_to_recommended': True,
     'indicator_min_fraction': 0.1,
     'missingness_imputation': <function mean at 0x11093bb90>,
     'sparse_indicators': True,
     'use_hierarchical_estimate': True,
     'user_transforms': []},
    )), ('classifier', LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
                       intercept_scaling=1, l1_ratio=None, max_iter=100,
                       multi_class='warn', n_jobs=None, penalty='l2',
                       random_state=None, solver='lbfgs', tol=0.0001, verbose=0,
                       warm_start=False))], 'verbose': False, 'preprocessor': vtreat.vtreat_api.BinomialOutcomeTreatment(outcome_target=True,
    params={'coders': {'clean_copy',
                'deviation_code',
                'impact_code',
                'indicator_code',
                'logit_code',
                'missing_indicator',
                'prevalence_code'},
     'cross_validation_k': 5,
     'cross_validation_plan': <vtreat.cross_plan.KWayCrossPlanYStratified object at 0x10fa81b50>,
     'filter_to_recommended': True,
     'indicator_min_fraction': 0.1,
     'missingness_imputation': <function mean at 0x11093bb90>,
     'sparse_indicators': True,
     'use_hierarchical_estimate': True,
     'user_transforms': []},
    ), 'classifier': LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
                       intercept_scaling=1, l1_ratio=None, max_iter=100,
                       multi_class='warn', n_jobs=None, penalty='l2',
                       random_state=None, solver='lbfgs', tol=0.0001, verbose=0,
                       warm_start=False), 'preprocessor__use_hierarchical_estimate': True, 'preprocessor__coders': {'prevalence_code', 'logit_code', 'indicator_code', 'deviation_code', 'impact_code', 'missing_indicator', 'clean_copy'}, 'preprocessor__filter_to_recommended': True, 'preprocessor__indicator_min_fraction': 0.1, 'preprocessor__cross_validation_plan': <vtreat.cross_plan.KWayCrossPlanYStratified object at 0x10fa81b50>, 'preprocessor__cross_validation_k': 5, 'preprocessor__user_transforms': [], 'preprocessor__sparse_indicators': True, 'preprocessor__missingness_imputation': <function mean at 0x11093bb90>, 'preprocessor__outcome_target': True, 'classifier__C': 1.0, 'classifier__class_weight': None, 'classifier__dual': False, 'classifier__fit_intercept': True, 'classifier__intercept_scaling': 1, 'classifier__l1_ratio': None, 'classifier__max_iter': 100, 'classifier__multi_class': 'warn', 'classifier__n_jobs': None, 'classifier__penalty': 'l2', 'classifier__random_state': None, 'classifier__solver': 'lbfgs', 'classifier__tol': 0.0001, 'classifier__verbose': 0, 'classifier__warm_start': False}

