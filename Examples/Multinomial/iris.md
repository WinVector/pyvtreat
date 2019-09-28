

```python
import vtreat
import pandas as pd
from sklearn.datasets import load_iris

iris = load_iris()

X, y = pd.DataFrame(iris['data']), iris['target']

plan  = vtreat.MultinomialOutcomeTreatment()
X_new = plan.fit_transform(X, y)
score_frame = plan.score_frame_
```


```python
score_frame
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>variable</th>
      <th>orig_variable</th>
      <th>treatment</th>
      <th>y_aware</th>
      <th>has_range</th>
      <th>PearsonR</th>
      <th>significance</th>
      <th>vcount</th>
      <th>default_threshold</th>
      <th>recommended</th>
      <th>outcome_target</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
      <td>clean_copy</td>
      <td>False</td>
      <td>True</td>
      <td>-0.717416</td>
      <td>5.288768e-25</td>
      <td>4.0</td>
      <td>0.25</td>
      <td>True</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>1</td>
      <td>clean_copy</td>
      <td>False</td>
      <td>True</td>
      <td>0.603348</td>
      <td>3.054699e-16</td>
      <td>4.0</td>
      <td>0.25</td>
      <td>True</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>2</td>
      <td>clean_copy</td>
      <td>False</td>
      <td>True</td>
      <td>-0.922765</td>
      <td>3.623379e-63</td>
      <td>4.0</td>
      <td>0.25</td>
      <td>True</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>3</td>
      <td>clean_copy</td>
      <td>False</td>
      <td>True</td>
      <td>-0.887344</td>
      <td>1.288504e-51</td>
      <td>4.0</td>
      <td>0.25</td>
      <td>True</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>0</td>
      <td>clean_copy</td>
      <td>False</td>
      <td>True</td>
      <td>0.079396</td>
      <td>3.341524e-01</td>
      <td>4.0</td>
      <td>0.25</td>
      <td>False</td>
      <td>1</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1</td>
      <td>1</td>
      <td>clean_copy</td>
      <td>False</td>
      <td>True</td>
      <td>-0.467703</td>
      <td>1.595624e-09</td>
      <td>4.0</td>
      <td>0.25</td>
      <td>True</td>
      <td>1</td>
    </tr>
    <tr>
      <th>6</th>
      <td>2</td>
      <td>2</td>
      <td>clean_copy</td>
      <td>False</td>
      <td>True</td>
      <td>0.201754</td>
      <td>1.329302e-02</td>
      <td>4.0</td>
      <td>0.25</td>
      <td>True</td>
      <td>1</td>
    </tr>
    <tr>
      <th>7</th>
      <td>3</td>
      <td>3</td>
      <td>clean_copy</td>
      <td>False</td>
      <td>True</td>
      <td>0.117899</td>
      <td>1.507473e-01</td>
      <td>4.0</td>
      <td>0.25</td>
      <td>True</td>
      <td>1</td>
    </tr>
    <tr>
      <th>8</th>
      <td>0</td>
      <td>0</td>
      <td>clean_copy</td>
      <td>False</td>
      <td>True</td>
      <td>0.638020</td>
      <td>1.619533e-18</td>
      <td>4.0</td>
      <td>0.25</td>
      <td>True</td>
      <td>2</td>
    </tr>
    <tr>
      <th>9</th>
      <td>1</td>
      <td>1</td>
      <td>clean_copy</td>
      <td>False</td>
      <td>True</td>
      <td>-0.135645</td>
      <td>9.791170e-02</td>
      <td>4.0</td>
      <td>0.25</td>
      <td>True</td>
      <td>2</td>
    </tr>
    <tr>
      <th>10</th>
      <td>2</td>
      <td>2</td>
      <td>clean_copy</td>
      <td>False</td>
      <td>True</td>
      <td>0.721011</td>
      <td>2.381987e-25</td>
      <td>4.0</td>
      <td>0.25</td>
      <td>True</td>
      <td>2</td>
    </tr>
    <tr>
      <th>11</th>
      <td>3</td>
      <td>3</td>
      <td>clean_copy</td>
      <td>False</td>
      <td>True</td>
      <td>0.769445</td>
      <td>1.297773e-30</td>
      <td>4.0</td>
      <td>0.25</td>
      <td>True</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
</div>



Note in the multinomial case the score frame is keyed by `orig_variable` plus `outcome_target` (not just `orig_variable`).  This means to decide which variables to include in a model we must aggregate.

The recommended new variables are:


```python
score_frame.variable[score_frame.recommended].unique()
```




    array([0, 1, 2, 3])



And the recommended original variables are:


```python
score_frame.orig_variable[score_frame.recommended].unique()
```




    array([0, 1, 2, 3])



In this example all the names are the same as the only variable treatments were the `clean_copy`.

Let's take a look at the transformed frame.


```python
X_new.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>5.1</td>
      <td>3.5</td>
      <td>1.4</td>
      <td>0.2</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4.9</td>
      <td>3.0</td>
      <td>1.4</td>
      <td>0.2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4.7</td>
      <td>3.2</td>
      <td>1.3</td>
      <td>0.2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4.6</td>
      <td>3.1</td>
      <td>1.5</td>
      <td>0.2</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5.0</td>
      <td>3.6</td>
      <td>1.4</td>
      <td>0.2</td>
    </tr>
  </tbody>
</table>
</div>



It doesn't apply for `clean_copy` varaibles, but in general `.fit_transform` values are a function of the incoming variable *plus* the cross-validation fold (not always just a function of just the incoming value!).  This is how the cross-frame methodology helps fight nested model bias driven over-fit for complex variables such as the impact-codes. `.transform()` values are, as one would expect, functions of just the input values (indpendent of cross validation fold).


```python

```
