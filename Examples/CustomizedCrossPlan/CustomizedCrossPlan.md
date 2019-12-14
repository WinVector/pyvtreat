
Nina Zumel, John Mount
October 2019

[These](https://github.com/WinVector/pyvtreat/blob/master/Examples/CustomizedCrossPlan/CustomizedCrossPlan.md) are notes on controlling the cross-validation plan in the [`Python` version of `vtreat`](https://github.com/WinVector/pyvtreat), for notes on the [`R` version of `vtreat`](https://github.com/WinVector/vtreat), please see [here](https://github.com/WinVector/vtreat/blob/master/Examples/CustomizedCrossPlan/CustomizedCrossPlan.md).

# Using Custom Cross-Validation Plans with `vtreat`

By default, `Python` `vtreat` uses simple randomized k-way cross validation when creating and evaluating complex synthetic variables. This will work well for the majority of applications. However, there may be times when you need a more specialized cross validation scheme for your modeling projects. In this document, we'll show how to replace the cross validation scheme in `vtreat`.


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

numpy.random.seed(2019)

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
      <td>1000.00000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>0.005766</td>
      <td>0.05800</td>
    </tr>
    <tr>
      <th>std</th>
      <td>1.024104</td>
      <td>0.23386</td>
    </tr>
    <tr>
      <th>min</th>
      <td>-3.205040</td>
      <td>0.00000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>-0.689752</td>
      <td>0.00000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>0.012250</td>
      <td>0.00000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>0.702009</td>
      <td>0.00000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>2.928164</td>
      <td>1.00000</td>
    </tr>
  </tbody>
</table>
</div>



First, try preparing this data using `vtreat`.

By default, `Python` `vtreat` uses a `y`-stratified randomized k-way cross validation when creating and evaluating complex synthetic variables. 

Here we start with a simple `k`-way cross validation plan. This will work well for the majority of applications. However, there may be times when you need a more specialized cross validation scheme for your modeling projects. In this document, we'll show how to replace the cross validation scheme in `vtreat`.


```python
#
# create the treatment plan
#

k = 5 # number of cross-val folds (actually, the default)
treatment_unstratified = vtreat.BinomialOutcomeTreatment(
    var_list=['x'],
    outcome_name='y',
    outcome_target=1,
    params=vtreat.vtreat_parameters({
        'cross_validation_plan': vtreat.cross_plan.KWayCrossPlan(),
        'cross_validation_k': k
    })
)

# prepare the training data
prepared_unstratified = treatment_unstratified.fit_transform(d, d['y'])
```

Let's look at the distribution  of the target outcome in each of the cross-validation groups:


```python
# convenience function to mark the cross-validation group of each row
def label_rows(d, cross_plan, *, label_column = 'group'):
    d[label_column] = 0
    for i in range(len(cross_plan)):
        app = cross_plan[i]['app']
        d.loc[app, label_column] = i
            
# label the rows            
label_rows(prepared_unstratified, treatment_unstratified.cross_plan_)
# print(prepared_unstratified.head())

# get some summary statistics on the data
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
      <td>14</td>
      <td>0.07</td>
      <td>200</td>
    </tr>
    <tr>
      <th>1</th>
      <td>18</td>
      <td>0.09</td>
      <td>200</td>
    </tr>
    <tr>
      <th>2</th>
      <td>8</td>
      <td>0.04</td>
      <td>200</td>
    </tr>
    <tr>
      <th>3</th>
      <td>12</td>
      <td>0.06</td>
      <td>200</td>
    </tr>
    <tr>
      <th>4</th>
      <td>6</td>
      <td>0.03</td>
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




    0.02135415650406262



The target prevalence in the cross validation groups can vary fairly widely with respect to the "true" prevalence of 0.05; this may adversely affect the resulting synthetic variables in the treated data. For situations like this where the target outcome is rare, you may want to stratify the cross-validation sampling to preserve the target prevalence as much as possible. 

## Passing in a Stratified Sampler

In this situation, `vtreat` has an alternative cross-validation sampler called `KWayCrossPlanYStratified` that can be passed in as follows:


```python

# create the treatment plan
treatment_stratified = vtreat.BinomialOutcomeTreatment(
    var_list=['x'],
    outcome_name='y',
    outcome_target=1,
    params=vtreat.vtreat_parameters({
        'cross_validation_plan': vtreat.cross_plan.KWayCrossPlanYStratified(),
        'cross_validation_k': k
    })
)

# prepare the training data
prepared_stratified = treatment_stratified.fit_transform(d, d['y'])

# examine the target prevalence
label_rows(prepared_stratified, treatment_stratified.cross_plan_)

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
      <td>13</td>
      <td>0.065</td>
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
      <td>12</td>
      <td>0.060</td>
      <td>200</td>
    </tr>
    <tr>
      <th>4</th>
      <td>10</td>
      <td>0.050</td>
      <td>200</td>
    </tr>
  </tbody>
</table>
</div>




```python
# standard deviation of target prevalence
std_stratified = numpy.std(stratified_summary['mean'])
std_stratified
```




    0.005099019513592784



The target prevalence in the stratified cross-validation groups are much closer to the true target prevalence, and the variation (standard deviation) of the target prevalence across groups has been substantially reduced.


```python
std_unstratified/std_stratified
```




    4.187894642712677



## Other cross-validation schemes

If you want to cross-validate under another scheme--for example, stratifying on the prevalences on an input class--you can write your own custom cross-validation scheme and pass it into `vtreat` in a similar fashion as above. Your cross-validation scheme must extend `vtreat`'s [`CrossValidationPlan`](https://github.com/WinVector/pyvtreat/blob/master/pkg/vtreat/cross_plan.py#L14) class.

Another benefit of explicit cross-validation plans is that one can use the same cross-validation plan for both the variable design and later modeling steps. This can limit data leaks across the cross-validation folds.

### Other predefined cross-validation schemes

In addition to the y-stratified cross validation, `vtreat` also defines a time-oriented cross validation scheme ([`OrderedCrossPlan`](https://github.com/WinVector/pyvtreat/blob/master/pkg/vtreat/cross_plan.py#L161)). The ordered cross plan treats time as the grouping variable. For each fold, all the datums in the application set (the datums that the model will be applied to) come from the same time period. All the datums in the training set come from one side of the application set; that is all the training data will be either earlier or later than the data in the application set. Ordered cross plans are useful when modeling time-oriented data.

Note: it is important to *not* use leave-one-out cross-validation when using nested or stacked modeling concepts (such as seen in `vtreat`), we have some notes on this [here](https://github.com/WinVector/vtreat/blob/master/extras/ConstantLeak.md).

