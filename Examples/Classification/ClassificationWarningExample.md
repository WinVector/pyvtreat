# [`vtreat`](https://github.com/WinVector/pyvtreat) Nested Model Bias Warning

For quite a while we have been teaching estimating variable re-encodings on the exact same data they
are later *naively* using to train a model on leads to an undesirable nested model bias.  The `vtreat`
package (both the [`R` version](https://github.com/WinVector/vtreat) and 
[`Python` version](https://github.com/WinVector/pyvtreat)) both incorporate a cross-frame method
that allows one to use all the training data both to build learn variable re-encodings and to correctly train a subsequent model (for an example please see our recent [PyData LA talk](http://www.win-vector.com/blog/2019/12/pydata-los-angeles-2019-talk-preparing-messy-real-world-data-for-supervised-machine-learning/)).

The next version of `vtreat` will warn the user if they have improperly used the same data for both `vtreat` impact code inference and downstream modeling.  So in addition to us warning you not to do this, the package now also checks and warns against this situation.


## Set up the Example


This example is copied from [some of our classification documentation](https://github.com/WinVector/pyvtreat/blob/master/Examples/Classification/Classification.md).


Load modules/packages.


```python
import pkg_resources
import pandas
import numpy
import numpy.random
import vtreat
import vtreat.util

numpy.random.seed(2019)
```

Generate example data. 

* `y` is a noisy sinusoidal function of the variable `x`
* `yc` is the output to be predicted: : whether `y` is > 0.5. 
* Input `xc` is a categorical variable that represents a discretization of `y`, along some `NaN`s
* Input `x2` is a pure noise variable with no relationship to the output


```python
def make_data(nrows):
    d = pandas.DataFrame({'x': 5*numpy.random.normal(size=nrows)})
    d['y'] = numpy.sin(d['x']) + 0.1*numpy.random.normal(size=nrows)
    d.loc[numpy.arange(3, 10), 'x'] = numpy.nan                           # introduce a nan level
    d['xc'] = ['level_' + str(5*numpy.round(yi/5, 1)) for yi in d['y']]
    d['x2'] = numpy.random.normal(size=nrows)
    d.loc[d['xc']=='level_-1.0', 'xc'] = numpy.nan  # introduce a nan level
    d['yc'] = d['y']>0.5
    return d

training_data = make_data(500)

training_data.head()
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>x</th>
      <th>y</th>
      <th>xc</th>
      <th>x2</th>
      <th>yc</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-1.088395</td>
      <td>-0.956311</td>
      <td>NaN</td>
      <td>-1.424184</td>
      <td>False</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4.107277</td>
      <td>-0.671564</td>
      <td>level_-0.5</td>
      <td>0.427360</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2</th>
      <td>7.406389</td>
      <td>0.906303</td>
      <td>level_1.0</td>
      <td>0.668849</td>
      <td>True</td>
    </tr>
    <tr>
      <th>3</th>
      <td>NaN</td>
      <td>0.222792</td>
      <td>level_0.0</td>
      <td>-0.015787</td>
      <td>False</td>
    </tr>
    <tr>
      <th>4</th>
      <td>NaN</td>
      <td>-0.975431</td>
      <td>NaN</td>
      <td>-0.491017</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
</div>




```python
outcome_name = 'yc'    # outcome variable / column
outcome_target = True  # value we consider positive
```

## Demonstrate the Warning

Now that we have the data, we want to treat it prior to modeling: we want training data where all the input variables are numeric and have no missing values or `NA`s.

First create the data treatment transform design object, in this case a treatment for a binomial classification problem.

We use the training data `training_data` to fit the transform and the return a treated training set: completely numeric, with no missing values.


```python
treatment = vtreat.BinomialOutcomeTreatment(
    outcome_name=outcome_name,      # outcome variable
    outcome_target=outcome_target,  # outcome of interest
    cols_to_copy=['y'],  # columns to "carry along" but not treat as input variables
)  
```


```python
train_prepared = treatment.fit_transform(training_data, training_data['yc'])
```

`train_prepared` is prepared in the correct way to use the same training data for inferring the impact-coded variables, using `.fit_transform()` instead of `.fit().transform()`.

We prepare new test or application data as follows.


```python
test_data = make_data(100)

test_prepared = treatment.transform(test_data)
```

The issue is: for training data we should not call `transform()`, but instead use the value returned by `.fit_transform()`.

The point is we should not do the following:


```python
train_prepared_wrong = treatment.transform(training_data)
```

    /Users/johnmount/opt/anaconda3/envs/ai_academy_3_7/lib/python3.7/site-packages/vtreat/vtreat_api.py:370: UserWarning: possibly called transform on same data used to fit (this causes over-fit, please use fit_transform() instead)
      "possibly called transform on same data used to fit (this causes over-fit, please use fit_transform() instead)")



Notice we now get a warning that we should not have done this, and in doing so we may have a nested model bias data leak.

And that is the new nested model bias warning feature.

The `R`-version of this document can be found [here](https://github.com/WinVector/vtreat/blob/master/Examples/Classification/ClassificationWarningExample.md).


```python

```

