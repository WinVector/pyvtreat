```python
import numpy
import numpy.random
import pandas
import vtreat
```


```python
numpy.random.seed(2019)
```


```python
n_rows = 1000
y_levels = ['a', 'b', 'c']
d = pandas.DataFrame({'y': numpy.random.choice(y_levels, size=n_rows)})
# signal variables, correlated with y-levels
for i in range(2):
    vmap = {yl: numpy.random.normal(size=1)[0] for yl in y_levels}
    d['var_n_' + str(i)] = [vmap[li] + numpy.random.normal(size=1)[0] for li in d['y']]
for i in range(2):
    col = numpy.random.choice(y_levels, size=n_rows)
    col = [col[i] if numpy.random.uniform(size=1)[0]<=0.8 else d['y'][i] for i in range(n_rows)]
    d['var_c_' + str(i)] = col
# noise variables, uncorrelated with y-levels
for i in range(2):
    d['noise_n_' + str(i)] = [numpy.random.normal(size=1)[0] + numpy.random.normal(size=1)[0] for li in d['y']]
for i in range(2):
    d['noise_c_' + str(i)] = numpy.random.choice(y_levels, size=n_rows)
d.head()
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>y</th>
      <th>var_n_0</th>
      <th>var_n_1</th>
      <th>var_c_0</th>
      <th>var_c_1</th>
      <th>noise_n_0</th>
      <th>noise_n_1</th>
      <th>noise_c_0</th>
      <th>noise_c_1</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>a</td>
      <td>0.544891</td>
      <td>1.595448</td>
      <td>b</td>
      <td>a</td>
      <td>1.196747</td>
      <td>-0.714955</td>
      <td>a</td>
      <td>a</td>
    </tr>
    <tr>
      <th>1</th>
      <td>c</td>
      <td>-0.433273</td>
      <td>0.778452</td>
      <td>a</td>
      <td>b</td>
      <td>0.332786</td>
      <td>-1.175452</td>
      <td>a</td>
      <td>b</td>
    </tr>
    <tr>
      <th>2</th>
      <td>b</td>
      <td>-1.230834</td>
      <td>0.859162</td>
      <td>b</td>
      <td>a</td>
      <td>-1.454101</td>
      <td>-3.079244</td>
      <td>a</td>
      <td>b</td>
    </tr>
    <tr>
      <th>3</th>
      <td>a</td>
      <td>1.158161</td>
      <td>-0.344363</td>
      <td>c</td>
      <td>b</td>
      <td>-1.175055</td>
      <td>-1.314302</td>
      <td>a</td>
      <td>b</td>
    </tr>
    <tr>
      <th>4</th>
      <td>c</td>
      <td>-1.029690</td>
      <td>0.789506</td>
      <td>c</td>
      <td>c</td>
      <td>0.809634</td>
      <td>0.951092</td>
      <td>b</td>
      <td>c</td>
    </tr>
  </tbody>
</table>
</div>




```python
treatment = vtreat.MultinomialOutcomeTreatment(
    outcome_name='y')
cross_frame = treatment.fit_transform(d, d['y'])
score_frame = treatment.score_frame_
```

To select variables we either make our selection in terms of new variables as follows.


```python
good_new_variables = score_frame.variable[score_frame.recommended].unique()
good_new_variables
```




    array(['var_n_0', 'var_n_1', 'var_c_0_logit_code_a',
           'var_c_0_prevalence_code', 'var_c_0_lev_b', 'var_c_0_lev_a',
           'var_c_0_lev_c', 'var_c_1_logit_code_a', 'var_c_1_prevalence_code',
           'var_c_1_lev_a', 'var_c_0_logit_code_b', 'var_c_1_logit_code_b',
           'var_c_1_lev_b', 'var_c_1_lev_c', 'var_c_0_logit_code_c',
           'var_c_1_logit_code_c'], dtype=object)



Or in terms of original variables as follows.


```python
good_original_variables = score_frame.orig_variable[score_frame.recommended].unique()
good_original_variables
```




    array(['var_n_0', 'var_n_1', 'var_c_0', 'var_c_1'], dtype=object)



Notice, in each case we must call unique as each variable (derived or original) is potentially qualified against each possible outcome.

The cross frame and score frame look like the following.


```python
cross_frame.head()
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>y</th>
      <th>var_n_0</th>
      <th>var_n_1</th>
      <th>var_c_0_logit_code_a</th>
      <th>var_c_0_logit_code_c</th>
      <th>var_c_0_logit_code_b</th>
      <th>var_c_0_prevalence_code</th>
      <th>var_c_0_lev_b</th>
      <th>var_c_0_lev_a</th>
      <th>var_c_0_lev_c</th>
      <th>var_c_1_logit_code_a</th>
      <th>var_c_1_logit_code_c</th>
      <th>var_c_1_logit_code_b</th>
      <th>var_c_1_prevalence_code</th>
      <th>var_c_1_lev_a</th>
      <th>var_c_1_lev_b</th>
      <th>var_c_1_lev_c</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>a</td>
      <td>0.544891</td>
      <td>1.595448</td>
      <td>-0.183425</td>
      <td>-0.246066</td>
      <td>0.333199</td>
      <td>0.352</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.164827</td>
      <td>-0.258790</td>
      <td>0.046310</td>
      <td>0.346</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>c</td>
      <td>-0.433273</td>
      <td>0.778452</td>
      <td>0.320872</td>
      <td>-0.144157</td>
      <td>-0.254121</td>
      <td>0.328</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>-0.149856</td>
      <td>-0.257395</td>
      <td>0.322305</td>
      <td>0.334</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>b</td>
      <td>-1.230834</td>
      <td>0.859162</td>
      <td>-0.233913</td>
      <td>-0.186684</td>
      <td>0.326693</td>
      <td>0.352</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.209085</td>
      <td>-0.287449</td>
      <td>0.023177</td>
      <td>0.346</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>a</td>
      <td>1.158161</td>
      <td>-0.344363</td>
      <td>-0.220148</td>
      <td>0.290051</td>
      <td>-0.199207</td>
      <td>0.320</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>-0.149856</td>
      <td>-0.257395</td>
      <td>0.322305</td>
      <td>0.334</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>c</td>
      <td>-1.029690</td>
      <td>0.789506</td>
      <td>-0.193397</td>
      <td>0.283625</td>
      <td>-0.212830</td>
      <td>0.320</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>-0.117681</td>
      <td>0.397098</td>
      <td>-0.522505</td>
      <td>0.320</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
treatment.score_frame_
```




<div>

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
      <td>var_n_0</td>
      <td>var_n_0</td>
      <td>clean_copy</td>
      <td>False</td>
      <td>True</td>
      <td>0.460713</td>
      <td>1.079644e-53</td>
      <td>4.0</td>
      <td>0.062500</td>
      <td>True</td>
      <td>a</td>
    </tr>
    <tr>
      <th>1</th>
      <td>var_n_1</td>
      <td>var_n_1</td>
      <td>clean_copy</td>
      <td>False</td>
      <td>True</td>
      <td>-0.253237</td>
      <td>4.259663e-16</td>
      <td>4.0</td>
      <td>0.062500</td>
      <td>True</td>
      <td>a</td>
    </tr>
    <tr>
      <th>2</th>
      <td>noise_n_0</td>
      <td>noise_n_0</td>
      <td>clean_copy</td>
      <td>False</td>
      <td>True</td>
      <td>0.003113</td>
      <td>9.216881e-01</td>
      <td>4.0</td>
      <td>0.062500</td>
      <td>False</td>
      <td>a</td>
    </tr>
    <tr>
      <th>3</th>
      <td>noise_n_1</td>
      <td>noise_n_1</td>
      <td>clean_copy</td>
      <td>False</td>
      <td>True</td>
      <td>-0.030264</td>
      <td>3.390479e-01</td>
      <td>4.0</td>
      <td>0.062500</td>
      <td>False</td>
      <td>a</td>
    </tr>
    <tr>
      <th>4</th>
      <td>var_c_0_logit_code_a</td>
      <td>var_c_0</td>
      <td>logit_code</td>
      <td>True</td>
      <td>True</td>
      <td>0.180222</td>
      <td>9.520214e-09</td>
      <td>12.0</td>
      <td>0.020833</td>
      <td>True</td>
      <td>a</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>91</th>
      <td>noise_c_1_logit_code_b</td>
      <td>noise_c_1</td>
      <td>logit_code</td>
      <td>True</td>
      <td>True</td>
      <td>0.015519</td>
      <td>6.240063e-01</td>
      <td>12.0</td>
      <td>0.020833</td>
      <td>False</td>
      <td>c</td>
    </tr>
    <tr>
      <th>92</th>
      <td>noise_c_1_prevalence_code</td>
      <td>noise_c_1</td>
      <td>prevalence_code</td>
      <td>False</td>
      <td>True</td>
      <td>-0.028448</td>
      <td>3.688353e-01</td>
      <td>4.0</td>
      <td>0.062500</td>
      <td>False</td>
      <td>c</td>
    </tr>
    <tr>
      <th>93</th>
      <td>noise_c_1_lev_c</td>
      <td>noise_c_1</td>
      <td>indicator_code</td>
      <td>False</td>
      <td>True</td>
      <td>-0.012951</td>
      <td>6.825051e-01</td>
      <td>12.0</td>
      <td>0.020833</td>
      <td>False</td>
      <td>c</td>
    </tr>
    <tr>
      <th>94</th>
      <td>noise_c_1_lev_b</td>
      <td>noise_c_1</td>
      <td>indicator_code</td>
      <td>False</td>
      <td>True</td>
      <td>-0.036427</td>
      <td>2.497826e-01</td>
      <td>12.0</td>
      <td>0.020833</td>
      <td>False</td>
      <td>c</td>
    </tr>
    <tr>
      <th>95</th>
      <td>noise_c_1_lev_a</td>
      <td>noise_c_1</td>
      <td>indicator_code</td>
      <td>False</td>
      <td>True</td>
      <td>0.049435</td>
      <td>1.182245e-01</td>
      <td>12.0</td>
      <td>0.020833</td>
      <td>False</td>
      <td>c</td>
    </tr>
  </tbody>
</table>
<p>96 rows × 11 columns</p>
</div>



## Conclusion

In all cases (classification, regression, unsupervised, and multinomial classification) the intent is that `vtreat` transforms are essentially one liners.

The preparation commands are organized as follows:

 * **Regression**: [`R` regression example](https://github.com/WinVector/vtreat/blob/master/Examples/Regression/Regression.md), [`Python` regression example](https://github.com/WinVector/pyvtreat/blob/master/Examples/Regression/Regression.md).
 * **Classification**: [`R` classification example](https://github.com/WinVector/vtreat/blob/master/Examples/Classification/Classification.md), [`Python` classification  example](https://github.com/WinVector/pyvtreat/blob/master/Examples/Classification/Classification.md).
 * **Unsupervised tasks**: [`R` unsupervised example](https://github.com/WinVector/vtreat/blob/master/Examples/Unsupervised/Unsupervised.md), [`Python` unsupervised example](https://github.com/WinVector/pyvtreat/blob/master/Examples/Unsupervised/Unsupervised.md).
 * **Multinomial classification**: [`R` multinomial classification example](https://winvector.github.io/vtreat/articles/MultiClassVtreat.html), [`Python` multinomial classification example](https://github.com/WinVector/pyvtreat/blob/master/Examples/Multinomial/MultinomialExample.ipynb).

These current revisions of the examples are designed to be small, yet complete.  So as a set they have some overlap, but the user can rely mostly on a single example for a single task type.


