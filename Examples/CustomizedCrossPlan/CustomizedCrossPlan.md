Nina Zumel, John Mount
March 2020

[These](https://github.com/WinVector/pyvtreat/blob/master/Examples/CustomizedCrossPlan/CustomizedCrossPlan.md) are notes on controlling the cross-validation plan in the [`Python` version of `vtreat`](https://github.com/WinVector/pyvtreat), for notes on the [`R` version of `vtreat`](https://github.com/WinVector/vtreat), please see [here](https://github.com/WinVector/vtreat/blob/master/Examples/CustomizedCrossPlan/CustomizedCrossPlan.md).

# Using Custom Cross-Validation Plans with `vtreat`

By default, `Python` `vtreat` uses a y-stratified randomized k-way cross validation when creating and evaluating complex synthetic variables. This will work well for the majority of applications. However, there may be times when you need a more specialized cross validation scheme for your modeling projects. In this document, we'll show how to replace the cross validation scheme in `vtreat`.


```python
import pandas
import numpy
import numpy.random

import vtreat
import vtreat.cross_plan
```

## Example: Highly Unbalanced Class Outcomes

As an example, suppose you have data where the target class of interest is relatively rare; in this case about 5%:


```python
n_row = 1000

numpy.random.seed(2020)

d = pandas.DataFrame({
    'x': numpy.random.normal(size=n_row),
    'y': numpy.random.binomial(size=n_row, p=0.05, n=1)
})

d.describe()
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>x</th>
      <th>y</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>1000.000000</td>
      <td>1000.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>-0.033441</td>
      <td>0.054000</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.974859</td>
      <td>0.226131</td>
    </tr>
    <tr>
      <th>min</th>
      <td>-2.870341</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>-0.693371</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>-0.033076</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>0.593758</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>3.099762</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>



First, try preparing this data using `vtreat`.

By default, `Python` `vtreat` uses a `y`-stratified randomized k-way cross validation when creating and evaluating complex synthetic variables. 

Here we start with the default `k`-way y-stratified cross validation plan. This will work well for the majority of applications. However, there may be times when you need a more specialized cross validation scheme for your modeling projects. In this document, we'll show how to replace the cross validation scheme in `vtreat`.


```python
#
# create the treatment plan
#

k = 5 # number of cross-val folds (actually, the default)
treatment_stratified = vtreat.BinomialOutcomeTreatment(
    var_list=['x'],
    outcome_name='y',
    outcome_target=1,
    params=vtreat.vtreat_parameters({
        'cross_validation_k': k,
        'retain_cross_plan': True,
    })
)

# prepare the training data
prepared_stratified = treatment_stratified.fit_transform(d, d['y'])
```

Let's look at the distribution  of the target outcome in each of the cross-validation groups:


```python
# convenience function to mark the cross-validation group of each row
def label_rows(df, cross_plan, *, label_column = 'group'):
    df[label_column] = 0
    for i in range(len(cross_plan)):
        app = cross_plan[i]['app']
        df.loc[app, label_column] = i
            
# label the rows            
label_rows(prepared_stratified, treatment_stratified.cross_plan_)
# print(prepared_stratified.head())

# get some summary statistics on the data
stratified_summary = prepared_stratified.groupby(['group']).agg({'y': ['sum', 'mean', 'count']})
stratified_summary.columns = stratified_summary.columns.get_level_values(1)
stratified_summary
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>sum</th>
      <th>mean</th>
      <th>count</th>
    </tr>
    <tr>
      <th>group</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>10</td>
      <td>0.050</td>
      <td>200</td>
    </tr>
    <tr>
      <th>1</th>
      <td>11</td>
      <td>0.055</td>
      <td>200</td>
    </tr>
    <tr>
      <th>2</th>
      <td>12</td>
      <td>0.060</td>
      <td>200</td>
    </tr>
    <tr>
      <th>3</th>
      <td>8</td>
      <td>0.040</td>
      <td>200</td>
    </tr>
    <tr>
      <th>4</th>
      <td>13</td>
      <td>0.065</td>
      <td>200</td>
    </tr>
  </tbody>
</table>
</div>




```python
# standard deviation of target prevalence per cross-val fold
std_stratified = numpy.std(stratified_summary['mean'])
std_stratified 
```




    0.008602325267042627



## Explicitly Controlling the Sampler

A user chosen cross validation plan generator can be passed in as follows.  Also to retain the plan for later
inspection, set the `'retain_cross_plan'` parameter.  The passed in class should be derived from
`vtreat.cross_plan.CrossValidationPlan`.


```python
class KWayCrossPlan(vtreat.cross_plan.CrossValidationPlan):
    """K-way cross validation plan"""

    def __init__(self):
        vtreat.cross_plan.CrossValidationPlan.__init__(self)

    # create a custom cross-plan generator
    # noinspection PyMethodMayBeStatic
    def _k_way_cross_plan(self, n_rows, k_folds):
        """randomly split range(n_rows) into k_folds disjoint groups"""
        # first assign groups modulo k (ensuring at least one in each group)
        grp = [i % k_folds for i in range(n_rows)]
        # now shuffle
        numpy.random.shuffle(grp)
        plan = [
            {
                "train": [i for i in range(n_rows) if grp[i] != j],
                "app": [i for i in range(n_rows) if grp[i] == j],
            }
            for j in range(k_folds)
        ]
        return plan
    
    def split_plan(self, *, n_rows=None, k_folds=None, data=None, y=None):
        if n_rows is None:
            raise ValueError("n_rows must not be None")
        if k_folds is None:
            raise ValueError("k_folds must not be None")
        return self._k_way_cross_plan(n_rows=n_rows, k_folds=k_folds)


# create the treatment plan
treatment_unstratified = vtreat.BinomialOutcomeTreatment(
    var_list=['x'],
    outcome_name='y',
    outcome_target=1,
    params=vtreat.vtreat_parameters({
        'cross_validation_plan': KWayCrossPlan(),
        'cross_validation_k': k,
        'retain_cross_plan': True,
    })
)

# prepare the training data
prepared_unstratified = treatment_unstratified.fit_transform(d, d['y'])
```


```python
# get some summary statistics on the data
label_rows(prepared_unstratified, treatment_unstratified.cross_plan_)
unstratified_summary = prepared_unstratified.groupby(['group']).agg({'y': ['sum', 'mean', 'count']})
unstratified_summary.columns = unstratified_summary.columns.get_level_values(1)
unstratified_summary
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>sum</th>
      <th>mean</th>
      <th>count</th>
    </tr>
    <tr>
      <th>group</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>8</td>
      <td>0.040</td>
      <td>200</td>
    </tr>
    <tr>
      <th>1</th>
      <td>9</td>
      <td>0.045</td>
      <td>200</td>
    </tr>
    <tr>
      <th>2</th>
      <td>15</td>
      <td>0.075</td>
      <td>200</td>
    </tr>
    <tr>
      <th>3</th>
      <td>8</td>
      <td>0.040</td>
      <td>200</td>
    </tr>
    <tr>
      <th>4</th>
      <td>14</td>
      <td>0.070</td>
      <td>200</td>
    </tr>
  </tbody>
</table>
</div>




```python
# standard deviation of target prevalence per cross-val fold
std_unstratified = numpy.std(unstratified_summary['mean'])
std_unstratified 
```




    0.015297058540778355



Notice the between group y-variances are about 70% larger in the unstratified sampling plan than in the stratified sampling plan.


```python
std_unstratified/std_stratified
```




    1.7782469350914576



## Other cross-validation schemes

If you want to cross-validate under another scheme--for example, stratifying on the prevalences on an input class--you can write your own custom cross-validation scheme and pass it into `vtreat` in a similar fashion as above. Your cross-validation scheme must extend `vtreat`'s [`CrossValidationPlan`](https://github.com/WinVector/pyvtreat/blob/master/pkg/vtreat/cross_plan.py#L14) class.

Another benefit of explicit cross-validation plans is that one can use the same cross-validation plan for both the variable design and later modeling steps. This can limit data leaks across the cross-validation folds.

### Other predefined cross-validation schemes

In addition to the y-stratified cross validation, `vtreat` also defines a time-oriented cross validation scheme ([`OrderedCrossPlan`](https://github.com/WinVector/pyvtreat/blob/master/pkg/vtreat/cross_plan.py#L161)). The ordered cross plan treats time as the grouping variable. For each fold, all the datums in the application set (the datums that the model will be applied to) come from the same time period. All the datums in the training set come from one side of the application set; that is all the training data will be either earlier or later than the data in the application set. Ordered cross plans are useful when modeling time-oriented data.

Note: it is important to *not* use leave-one-out cross-validation when using nested or stacked modeling concepts (such as seen in `vtreat`), we have some notes on this [here](https://github.com/WinVector/vtreat/blob/master/extras/ConstantLeak.md).

