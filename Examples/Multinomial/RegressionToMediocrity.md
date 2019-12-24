Let's take another look at the concept of "regression to mediocrity" as described in Nina Zumel's great article [*Why Do We Plot Predictions on the x-axis?*](http://www.win-vector.com/blog/2019/09/why-do-we-plot-predictions-on-the-x-axis/).

This time let's consider the issue from the point of view of multinomial classification (a concept discussed [here](https://github.com/WinVector/pyvtreat/blob/master/Examples/Multinomial/MultinomialExample.md)).

First we load our packages and generate some synthetic data.


```python
import numpy
import numpy.random
import pandas
import sklearn.linear_model
import sklearn.metrics

numpy.random.seed(2019)
```


```python
numpy.random.seed(34524)

N = 1000

df = pandas.DataFrame({
    'x1': numpy.random.normal(size=N),
    'x2': numpy.random.normal(size=N),
    })
noise = numpy.random.normal(size=N)
y = df.x1 + df.x2 + noise
df['y'] = numpy.where(
    y < -3, 
    'short_opportunity', 
    numpy.where(
        y > 3, 
        'long_opportunity', 
        'indeterminate'))

df.head()
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>x1</th>
      <th>x2</th>
      <th>y</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.389409</td>
      <td>-2.115627</td>
      <td>indeterminate</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-0.354096</td>
      <td>-0.195495</td>
      <td>indeterminate</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-0.057603</td>
      <td>0.928929</td>
      <td>indeterminate</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-0.400339</td>
      <td>-0.936919</td>
      <td>indeterminate</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-0.125245</td>
      <td>-0.220789</td>
      <td>indeterminate</td>
    </tr>
  </tbody>
</table>
</div>




```python
df['y'].value_counts()
```




    indeterminate        925
    short_opportunity     41
    long_opportunity      34
    Name: y, dtype: int64



Please pretend this data is a record of stock market trading situations where we have determined (by peaking into the future, something quite easy to do with historic data) there is a large opportunity to make money buying security (called `long_opportunity`) or a larger opportunity to make money selling a security (called `short_opportunity`).

Let's build a model using the two observable dependent variables `x1` and `x2`.  These are measurements that are available at the time of the proposed trade that we hope correlate with or "predict" the future trading result.  For our model we will use a simple multinomial logistic regression.


```python
model_vars = ['x1', 'x2']

fitter = sklearn.linear_model.LogisticRegression(
    solver = 'saga',
    penalty = 'l2',
    C = 1,
    max_iter = 1e+5,
    multi_class = 'multinomial')
fitter.fit(df[model_vars], df['y'])

```




    LogisticRegression(C=1, class_weight=None, dual=False, fit_intercept=True,
                       intercept_scaling=1, l1_ratio=None, max_iter=100000.0,
                       multi_class='multinomial', n_jobs=None, penalty='l2',
                       random_state=None, solver='saga', tol=0.0001, verbose=0,
                       warm_start=False)



We can then examining the model predictions on the training data itself (a *much* lower standard than evaluating the model on held out data!!).


```python
# convenience functions for predicting and adding predictions to original data frame

def add_predictions(d_prepared, model_vars, fitter):
    pred = fitter.predict_proba(d_prepared[model_vars])
    classes = fitter.classes_
    d_prepared['prob_on_predicted_class'] = 0
    d_prepared['prediction'] = None
    for i in range(len(classes)):
        cl = classes[i]
        d_prepared[cl] = pred[:, i]
        improved = d_prepared[cl] > d_prepared['prob_on_predicted_class']
        d_prepared.loc[improved, 'prediction'] = cl
        d_prepared.loc[improved, 'prob_on_predicted_class'] = d_prepared.loc[improved, cl]
    return d_prepared

def add_value_by_column(d_prepared, name_column, new_column):
    vals = d_prepared[name_column].unique()
    d_prepared[new_column] = None
    for v in vals:
        matches = d_prepared[name_column]==v
        d_prepared.loc[matches, new_column] = d_prepared.loc[matches, v]
    return d_prepared
```


```python
# df['prediction'] = fitter.predict(df[model_vars])
df = add_predictions(df, model_vars, fitter)
df = add_value_by_column(df, 'y', 'prob_on_correct_class')
```


```python
result_columns = ['y', 'prob_on_predicted_class', 'prediction', 
                  'indeterminate', 'long_opportunity', 
                  'short_opportunity', 'prob_on_correct_class']
df[result_columns].head()
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>y</th>
      <th>prob_on_predicted_class</th>
      <th>prediction</th>
      <th>indeterminate</th>
      <th>long_opportunity</th>
      <th>short_opportunity</th>
      <th>prob_on_correct_class</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>indeterminate</td>
      <td>0.949149</td>
      <td>indeterminate</td>
      <td>0.949149</td>
      <td>0.000175</td>
      <td>0.050676</td>
      <td>0.949149</td>
    </tr>
    <tr>
      <th>1</th>
      <td>indeterminate</td>
      <td>0.989852</td>
      <td>indeterminate</td>
      <td>0.989852</td>
      <td>0.001375</td>
      <td>0.008773</td>
      <td>0.989852</td>
    </tr>
    <tr>
      <th>2</th>
      <td>indeterminate</td>
      <td>0.982227</td>
      <td>indeterminate</td>
      <td>0.982227</td>
      <td>0.017159</td>
      <td>0.000614</td>
      <td>0.982227</td>
    </tr>
    <tr>
      <th>3</th>
      <td>indeterminate</td>
      <td>0.964236</td>
      <td>indeterminate</td>
      <td>0.964236</td>
      <td>0.000332</td>
      <td>0.035432</td>
      <td>0.964236</td>
    </tr>
    <tr>
      <th>4</th>
      <td>indeterminate</td>
      <td>0.992411</td>
      <td>indeterminate</td>
      <td>0.992411</td>
      <td>0.002010</td>
      <td>0.005579</td>
      <td>0.992411</td>
    </tr>
  </tbody>
</table>
</div>



Notice, as described in [*The Simpler Derivation of Logistic Regression*](http://www.win-vector.com/blog/2011/09/the-simpler-derivation-of-logistic-regression/) that the sums of the prediction probabilities essentially equal the counts of each category on the training data (differences due to numeric issues and regularization).


```python
df[['short_opportunity', 'indeterminate', 'long_opportunity']].sum(axis=0)
```




    short_opportunity     41.007198
    indeterminate        924.988576
    long_opportunity      34.004226
    dtype: float64




```python
df['y'].value_counts()
```




    indeterminate        925
    short_opportunity     41
    long_opportunity      34
    Name: y, dtype: int64



A common way to examine the relation of the model predictions to outcomes is a graphical table called a *confusion matrix*.  The [scikit learn documentation](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html) has states:

> By definition a confusion matrix `C` is such that `C[i,j]` is equal to the number of observations known to be in group `i` but predicted to be in group `j`.

and

> Wikipedia and other references may use a different convention for axes.

This means in the scikit learn convention the column-id is determined by the prediction.  This further means: as a visual point the horizontal position of cells in the scikit learn confusion matrix is determined by the prediction because matrices have the odd convention that the first index is row which specifies what vertical level one is referring to.

Frankly we think scikit learn has the right rendering choice: consistency and legibility over convention. As Nina Zumel [demonstrated](http://www.win-vector.com/blog/2019/09/why-do-we-plot-predictions-on-the-x-axis/): there are good reasons to have predictions on the x-axis for plots, and the same holds for diagrams or matrices.

So let's look at this confusion matrix.


```python
sklearn.metrics.confusion_matrix(
    y_true=df.y, 
    y_pred=df.prediction, 
    labels=['short_opportunity', 'indeterminate', 'long_opportunity'])
```




    array([[ 14,  27,   0],
           [  3, 918,   4],
           [  0,  23,  11]])



Our claim is: the prediction is controlling left/right in this matrix and the actual value to be predicted is determining up/down.

What we have noticed often in practice is: for unbalanced classification problems, there is more vertical than horizontal dispersion in such confusion matrices.  This means: the predictions tend to have less range than seen in the training data.  Though this is not always the case (especially when classes are closer to balanced), some counter examples please see [here](https://github.com/WinVector/pyvtreat/blob/master/Examples/Multinomial/MultinomialExample.md) and [here](https://github.com/WinVector/vtreat/blob/master/Examples/Multinomial/MultinomialExample.md).

We can confirm this as we see there are 75 actual values of `y` that are not `intermediate` and only 32 values of `prediction` that are not intermediate.  As the rows of the confusion matrix match the `y`-totals and the columns of the confusion matrix match the `prediction` totals we can confirm the matrix is oriented as described.


```python
sum(df['y']!='indeterminate')
```




    75




```python
sum(df['prediction']!='indeterminate')
```




    32



Right or wrong, the model only identifies about one half the rate of possible extreme situations. This is not a pathology, but a typical conservative failure: good models tend to have less variation than their training data (or not more than, especially when using regularized methods).  I would try to liken this to the [regression to mediocrity](https://en.wikipedia.org/wiki/Regression_toward_the_mean) effects [Nina Zumel already described clearly](http://www.win-vector.com/blog/2019/09/why-do-we-plot-predictions-on-the-x-axis/).

Of course one can try to adjust the per-class thresholds to find more potential trading opportunities. However, in my experience the new opportunities found are often of lower quality than the ones initially identified.

