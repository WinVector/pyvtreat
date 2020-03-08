# Using [vtreat](https://github.com/WinVector/pyvtreat) with Multinomial Classification Problems

Nina Zumel and John Mount
November 2019

Note: this is a description of the [`Python` version of `vtreat`](https://github.com/WinVector/pyvtreat), the same example for the [`R` version of `vtreat`](https://github.com/WinVector/vtreat) can be found [here](https://github.com/WinVector/pyvtreat/blob/master/Examples/Classification/Classification.md).


## Preliminaries

Load modules/packages.


```python
import pkg_resources
import pandas
import numpy
import numpy.random
import seaborn
import matplotlib.pyplot as plt
import vtreat
import vtreat.util
import wvpy.util

numpy.random.seed(2019)
```

Generate example data. 

* `y` is a noisy sinusoidal function of the variable `x`
* `yc` is the multiple class output to be predicted: : `y`'s quantized value as 'large', 'liminal', or 'small'.
* Input `xc` is a categorical variable that represents a discretization of `y`, along with some `NaN`s
* Input `x2` is a pure noise variable with no relationship to the output


```python
def make_data(nrows):
    d = pandas.DataFrame({'x': 5*numpy.random.normal(size=nrows)})
    d['y'] = numpy.sin(d['x']) + 0.1*numpy.random.normal(size=nrows)
    d.loc[numpy.arange(3, 10), 'x'] = numpy.nan                           # introduce a nan level
    d['xc'] = ['level_' + str(5*numpy.round(yi/5, 1)) for yi in d['y']]
    d['x2'] = numpy.random.normal(size=nrows)
    d.loc[d['xc']=='level_-1.0', 'xc'] = numpy.nan  # introduce a nan level
    d['yc'] = numpy.where(d['y']>0.5, 'large', numpy.where(d['y']<-0.5, 'small', 'liminal'))
    return d

d = make_data(500)

d.head()
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
      <td>small</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4.107277</td>
      <td>-0.671564</td>
      <td>level_-0.5</td>
      <td>0.427360</td>
      <td>small</td>
    </tr>
    <tr>
      <th>2</th>
      <td>7.406389</td>
      <td>0.906303</td>
      <td>level_1.0</td>
      <td>0.668849</td>
      <td>large</td>
    </tr>
    <tr>
      <th>3</th>
      <td>NaN</td>
      <td>0.222792</td>
      <td>level_0.0</td>
      <td>-0.015787</td>
      <td>liminal</td>
    </tr>
    <tr>
      <th>4</th>
      <td>NaN</td>
      <td>-0.975431</td>
      <td>NaN</td>
      <td>-0.491017</td>
      <td>small</td>
    </tr>
  </tbody>
</table>
</div>



### Some quick data exploration

Check how many levels `xc` has, and their distribution (including `NaN`)


```python
d['xc'].unique()
```




    array([nan, 'level_-0.5', 'level_1.0', 'level_0.0', 'level_-0.0',
           'level_0.5'], dtype=object)




```python
d['xc'].value_counts(dropna=False)
```




    level_1.0     140
    NaN           109
    level_-0.5    103
    level_0.5      75
    level_0.0      37
    level_-0.0     36
    Name: xc, dtype: int64



Show the distribution of `yc`


```python
d['yc'].value_counts(dropna=False)
```




    large      175
    small      166
    liminal    159
    Name: yc, dtype: int64



## Build a transform appropriate for classification problems.

Now that we have the data, we want to treat it prior to modeling: we want training data where all the input variables are numeric and have no missing values or `NaN`s.

First create the data treatment transform object, in this case a treatment for a multinomial classification problem.


```python
transform = vtreat.MultinomialOutcomeTreatment(
    outcome_name='yc',    # outcome variable
    cols_to_copy=['y'],   # columns to "carry along" but not treat as input variables
)  
```

Use the training data `d` to fit the transform and return a treated training set: completely numeric, with no missing values.
Note that for the training data `d`, `transform.fit_transform()` is **not** the same as `transform.fit().transform()`; the second call can lead to nested model bias in some situations, and is **not** recommended. 
For other, later data, not seen during transform design `transform.transform(o)` is an appropriate step.


```python
d_prepared = transform.fit_transform(d, d['yc'])
```

Now examine the score frame, which gives information about each new variable, including its type, which original variable it is  derived from, its (cross-validated) correlation with the outcome, and its (cross-validated) significance as a one-variable linear model for the outcome. 


```python
transform.score_frame_
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
      <td>x_is_bad</td>
      <td>x</td>
      <td>missing_indicator</td>
      <td>False</td>
      <td>True</td>
      <td>-0.051749</td>
      <td>2.480789e-01</td>
      <td>2.0</td>
      <td>0.100000</td>
      <td>False</td>
      <td>large</td>
    </tr>
    <tr>
      <th>1</th>
      <td>xc_is_bad</td>
      <td>xc</td>
      <td>missing_indicator</td>
      <td>False</td>
      <td>True</td>
      <td>-0.387438</td>
      <td>2.353230e-19</td>
      <td>2.0</td>
      <td>0.100000</td>
      <td>True</td>
      <td>large</td>
    </tr>
    <tr>
      <th>2</th>
      <td>x</td>
      <td>x</td>
      <td>clean_copy</td>
      <td>False</td>
      <td>True</td>
      <td>0.052826</td>
      <td>2.383607e-01</td>
      <td>2.0</td>
      <td>0.100000</td>
      <td>False</td>
      <td>large</td>
    </tr>
    <tr>
      <th>3</th>
      <td>x2</td>
      <td>x2</td>
      <td>clean_copy</td>
      <td>False</td>
      <td>True</td>
      <td>0.069126</td>
      <td>1.226672e-01</td>
      <td>2.0</td>
      <td>0.100000</td>
      <td>False</td>
      <td>large</td>
    </tr>
    <tr>
      <th>4</th>
      <td>xc_logit_code_liminal</td>
      <td>xc</td>
      <td>logit_code</td>
      <td>True</td>
      <td>True</td>
      <td>-0.447451</td>
      <td>5.490853e-26</td>
      <td>3.0</td>
      <td>0.066667</td>
      <td>False</td>
      <td>large</td>
    </tr>
    <tr>
      <th>5</th>
      <td>xc_logit_code_large</td>
      <td>xc</td>
      <td>logit_code</td>
      <td>True</td>
      <td>True</td>
      <td>0.867882</td>
      <td>1.996377e-153</td>
      <td>3.0</td>
      <td>0.066667</td>
      <td>True</td>
      <td>large</td>
    </tr>
    <tr>
      <th>6</th>
      <td>xc_logit_code_small</td>
      <td>xc</td>
      <td>logit_code</td>
      <td>True</td>
      <td>True</td>
      <td>-0.631421</td>
      <td>5.580035e-57</td>
      <td>3.0</td>
      <td>0.066667</td>
      <td>False</td>
      <td>large</td>
    </tr>
    <tr>
      <th>7</th>
      <td>xc_prevalence_code</td>
      <td>xc</td>
      <td>prevalence_code</td>
      <td>False</td>
      <td>True</td>
      <td>0.567968</td>
      <td>4.779330e-44</td>
      <td>1.0</td>
      <td>0.200000</td>
      <td>True</td>
      <td>large</td>
    </tr>
    <tr>
      <th>8</th>
      <td>xc_lev_level_1_0</td>
      <td>xc</td>
      <td>indicator_code</td>
      <td>False</td>
      <td>True</td>
      <td>0.849837</td>
      <td>1.270913e-140</td>
      <td>4.0</td>
      <td>0.050000</td>
      <td>True</td>
      <td>large</td>
    </tr>
    <tr>
      <th>9</th>
      <td>xc_lev__NA_</td>
      <td>xc</td>
      <td>indicator_code</td>
      <td>False</td>
      <td>True</td>
      <td>-0.387438</td>
      <td>2.353230e-19</td>
      <td>4.0</td>
      <td>0.050000</td>
      <td>True</td>
      <td>large</td>
    </tr>
    <tr>
      <th>10</th>
      <td>xc_lev_level_-0_5</td>
      <td>xc</td>
      <td>indicator_code</td>
      <td>False</td>
      <td>True</td>
      <td>-0.373767</td>
      <td>5.045930e-18</td>
      <td>4.0</td>
      <td>0.050000</td>
      <td>True</td>
      <td>large</td>
    </tr>
    <tr>
      <th>11</th>
      <td>xc_lev_level_0_5</td>
      <td>xc</td>
      <td>indicator_code</td>
      <td>False</td>
      <td>True</td>
      <td>0.102752</td>
      <td>2.156466e-02</td>
      <td>4.0</td>
      <td>0.050000</td>
      <td>True</td>
      <td>large</td>
    </tr>
    <tr>
      <th>12</th>
      <td>x_is_bad</td>
      <td>x</td>
      <td>missing_indicator</td>
      <td>False</td>
      <td>True</td>
      <td>0.028292</td>
      <td>5.279294e-01</td>
      <td>2.0</td>
      <td>0.100000</td>
      <td>False</td>
      <td>liminal</td>
    </tr>
    <tr>
      <th>13</th>
      <td>xc_is_bad</td>
      <td>xc</td>
      <td>missing_indicator</td>
      <td>False</td>
      <td>True</td>
      <td>-0.360534</td>
      <td>8.564773e-17</td>
      <td>2.0</td>
      <td>0.100000</td>
      <td>True</td>
      <td>liminal</td>
    </tr>
    <tr>
      <th>14</th>
      <td>x</td>
      <td>x</td>
      <td>clean_copy</td>
      <td>False</td>
      <td>True</td>
      <td>-0.060936</td>
      <td>1.736986e-01</td>
      <td>2.0</td>
      <td>0.100000</td>
      <td>False</td>
      <td>liminal</td>
    </tr>
    <tr>
      <th>15</th>
      <td>x2</td>
      <td>x2</td>
      <td>clean_copy</td>
      <td>False</td>
      <td>True</td>
      <td>0.001912</td>
      <td>9.659861e-01</td>
      <td>2.0</td>
      <td>0.100000</td>
      <td>False</td>
      <td>liminal</td>
    </tr>
    <tr>
      <th>16</th>
      <td>xc_logit_code_liminal</td>
      <td>xc</td>
      <td>logit_code</td>
      <td>True</td>
      <td>True</td>
      <td>0.705192</td>
      <td>2.141044e-76</td>
      <td>3.0</td>
      <td>0.066667</td>
      <td>True</td>
      <td>liminal</td>
    </tr>
    <tr>
      <th>17</th>
      <td>xc_logit_code_large</td>
      <td>xc</td>
      <td>logit_code</td>
      <td>True</td>
      <td>True</td>
      <td>-0.268632</td>
      <td>1.032402e-09</td>
      <td>3.0</td>
      <td>0.066667</td>
      <td>False</td>
      <td>liminal</td>
    </tr>
    <tr>
      <th>18</th>
      <td>xc_logit_code_small</td>
      <td>xc</td>
      <td>logit_code</td>
      <td>True</td>
      <td>True</td>
      <td>-0.201252</td>
      <td>5.750716e-06</td>
      <td>3.0</td>
      <td>0.066667</td>
      <td>False</td>
      <td>liminal</td>
    </tr>
    <tr>
      <th>19</th>
      <td>xc_prevalence_code</td>
      <td>xc</td>
      <td>prevalence_code</td>
      <td>False</td>
      <td>True</td>
      <td>-0.711666</td>
      <td>2.164070e-78</td>
      <td>1.0</td>
      <td>0.200000</td>
      <td>True</td>
      <td>liminal</td>
    </tr>
    <tr>
      <th>20</th>
      <td>xc_lev_level_1_0</td>
      <td>xc</td>
      <td>indicator_code</td>
      <td>False</td>
      <td>True</td>
      <td>-0.425828</td>
      <td>1.924077e-23</td>
      <td>4.0</td>
      <td>0.050000</td>
      <td>True</td>
      <td>liminal</td>
    </tr>
    <tr>
      <th>21</th>
      <td>xc_lev__NA_</td>
      <td>xc</td>
      <td>indicator_code</td>
      <td>False</td>
      <td>True</td>
      <td>-0.360534</td>
      <td>8.564773e-17</td>
      <td>4.0</td>
      <td>0.050000</td>
      <td>True</td>
      <td>liminal</td>
    </tr>
    <tr>
      <th>22</th>
      <td>xc_lev_level_-0_5</td>
      <td>xc</td>
      <td>indicator_code</td>
      <td>False</td>
      <td>True</td>
      <td>0.140658</td>
      <td>1.615849e-03</td>
      <td>4.0</td>
      <td>0.050000</td>
      <td>True</td>
      <td>liminal</td>
    </tr>
    <tr>
      <th>23</th>
      <td>xc_lev_level_0_5</td>
      <td>xc</td>
      <td>indicator_code</td>
      <td>False</td>
      <td>True</td>
      <td>0.194241</td>
      <td>1.218632e-05</td>
      <td>4.0</td>
      <td>0.050000</td>
      <td>True</td>
      <td>liminal</td>
    </tr>
    <tr>
      <th>24</th>
      <td>x_is_bad</td>
      <td>x</td>
      <td>missing_indicator</td>
      <td>False</td>
      <td>True</td>
      <td>0.024435</td>
      <td>5.856828e-01</td>
      <td>2.0</td>
      <td>0.100000</td>
      <td>False</td>
      <td>small</td>
    </tr>
    <tr>
      <th>25</th>
      <td>xc_is_bad</td>
      <td>xc</td>
      <td>missing_indicator</td>
      <td>False</td>
      <td>True</td>
      <td>0.748935</td>
      <td>4.736195e-91</td>
      <td>2.0</td>
      <td>0.100000</td>
      <td>True</td>
      <td>small</td>
    </tr>
    <tr>
      <th>26</th>
      <td>x</td>
      <td>x</td>
      <td>clean_copy</td>
      <td>False</td>
      <td>True</td>
      <td>0.006755</td>
      <td>8.802327e-01</td>
      <td>2.0</td>
      <td>0.100000</td>
      <td>False</td>
      <td>small</td>
    </tr>
    <tr>
      <th>27</th>
      <td>x2</td>
      <td>x2</td>
      <td>clean_copy</td>
      <td>False</td>
      <td>True</td>
      <td>-0.071903</td>
      <td>1.083077e-01</td>
      <td>2.0</td>
      <td>0.100000</td>
      <td>False</td>
      <td>small</td>
    </tr>
    <tr>
      <th>28</th>
      <td>xc_logit_code_liminal</td>
      <td>xc</td>
      <td>logit_code</td>
      <td>True</td>
      <td>True</td>
      <td>-0.244170</td>
      <td>3.201241e-08</td>
      <td>3.0</td>
      <td>0.066667</td>
      <td>False</td>
      <td>small</td>
    </tr>
    <tr>
      <th>29</th>
      <td>xc_logit_code_large</td>
      <td>xc</td>
      <td>logit_code</td>
      <td>True</td>
      <td>True</td>
      <td>-0.613362</td>
      <td>5.345060e-53</td>
      <td>3.0</td>
      <td>0.066667</td>
      <td>False</td>
      <td>small</td>
    </tr>
    <tr>
      <th>30</th>
      <td>xc_logit_code_small</td>
      <td>xc</td>
      <td>logit_code</td>
      <td>True</td>
      <td>True</td>
      <td>0.838533</td>
      <td>1.973352e-133</td>
      <td>3.0</td>
      <td>0.066667</td>
      <td>True</td>
      <td>small</td>
    </tr>
    <tr>
      <th>31</th>
      <td>xc_prevalence_code</td>
      <td>xc</td>
      <td>prevalence_code</td>
      <td>False</td>
      <td>True</td>
      <td>0.128509</td>
      <td>3.998112e-03</td>
      <td>1.0</td>
      <td>0.200000</td>
      <td>True</td>
      <td>small</td>
    </tr>
    <tr>
      <th>32</th>
      <td>xc_lev_level_1_0</td>
      <td>xc</td>
      <td>indicator_code</td>
      <td>False</td>
      <td>True</td>
      <td>-0.439636</td>
      <td>4.791039e-25</td>
      <td>4.0</td>
      <td>0.050000</td>
      <td>True</td>
      <td>small</td>
    </tr>
    <tr>
      <th>33</th>
      <td>xc_lev__NA_</td>
      <td>xc</td>
      <td>indicator_code</td>
      <td>False</td>
      <td>True</td>
      <td>0.748935</td>
      <td>4.736195e-91</td>
      <td>4.0</td>
      <td>0.050000</td>
      <td>True</td>
      <td>small</td>
    </tr>
    <tr>
      <th>34</th>
      <td>xc_lev_level_-0_5</td>
      <td>xc</td>
      <td>indicator_code</td>
      <td>False</td>
      <td>True</td>
      <td>0.239464</td>
      <td>5.952274e-08</td>
      <td>4.0</td>
      <td>0.050000</td>
      <td>True</td>
      <td>small</td>
    </tr>
    <tr>
      <th>35</th>
      <td>xc_lev_level_0_5</td>
      <td>xc</td>
      <td>indicator_code</td>
      <td>False</td>
      <td>True</td>
      <td>-0.296154</td>
      <td>1.401007e-11</td>
      <td>4.0</td>
      <td>0.050000</td>
      <td>True</td>
      <td>small</td>
    </tr>
  </tbody>
</table>
</div>



Note that the variable `xc` has been converted to multiple variables: 

* an indicator variable for each possible level (`xc_lev_level_*`)
* the value of a (cross-validated) one-variable "one versus rest" model for `yc` as a function of `xc`; one per possible outcome class (`xc_logit_code_*`)
* a variable that returns how prevalent this particular value of `xc` is in the training data (`xc_prevalence_code`)
* a variable indicating when `xc` was `NaN` in the original data (`xc_is_bad`)

Any or all of these new variables are available for downstream modeling. 

Variables of type `logit_code_*` are useful when dealing with categorical variables with a very large number of possible levels. For example, a categorical variable with 10,000 possible values potentially converts to 10,000 indicator variables, which may be unwieldy for some modeling methods. Using one numerical variable of type `logit_code_*` per outcome target may be a preferable alternative.

Unlike the other `vtreat` treatments (Numeric, Binomial, Unsupervised), the score frame here has *more* rows than created variables, because the significance of each variable is evaluated against each possible outcome target.

The `recommended` column indicates which variables are non constant (`has_range` == True) and have a significance value smaller than `default_threshold` with respect to a particular outcome target.  See the section *Deriving the Default Thresholds* below for the reasoning behind the default thresholds. Recommended columns are intended as advice about which variables appear to be most likely to be useful in a downstream model. This advice attempts to be conservative, to reduce the possibility of mistakenly eliminating variables that may in fact be useful (although, obviously, it can still mistakenly eliminate variables that have a real but non-linear relationship to the output, as is the case with `x`, in  our example). Since each variable has multiple recommendations, one can consider a variable to be recommended if it is recommended for any of the outcome targets: an OR of all the recommendations.

## Examining variables

To select variables we either make our selection in terms of new variables as follows.


```python
score_frame = transform.score_frame_
good_new_variables = score_frame.variable[score_frame.recommended].unique()
good_new_variables
```




    array(['xc_is_bad', 'xc_logit_code_large', 'xc_prevalence_code',
           'xc_lev_level_1_0', 'xc_lev__NA_', 'xc_lev_level_-0_5',
           'xc_lev_level_0_5', 'xc_logit_code_liminal', 'xc_logit_code_small'],
          dtype=object)



Or in terms of original variables as follows.


```python
good_original_variables = score_frame.orig_variable[score_frame.recommended].unique()
good_original_variables
```




    array(['xc'], dtype=object)



Notice, in each case we must call unique as each variable (derived or original) is potentially qualified against each possible outcome.

Notice that, by default, `d_prepared` only includes recommended variables (along with `y` and `yc`):


```python
d_prepared.head()
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>y</th>
      <th>yc</th>
      <th>xc_is_bad</th>
      <th>xc_logit_code_liminal</th>
      <th>xc_logit_code_large</th>
      <th>xc_logit_code_small</th>
      <th>xc_prevalence_code</th>
      <th>xc_lev_level_1_0</th>
      <th>xc_lev__NA_</th>
      <th>xc_lev_level_-0_5</th>
      <th>xc_lev_level_0_5</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-0.956311</td>
      <td>small</td>
      <td>1.0</td>
      <td>-5.745320</td>
      <td>-5.837138</td>
      <td>1.099069</td>
      <td>0.218</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-0.671564</td>
      <td>small</td>
      <td>0.0</td>
      <td>0.315510</td>
      <td>-5.835271</td>
      <td>0.517553</td>
      <td>0.206</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.906303</td>
      <td>large</td>
      <td>0.0</td>
      <td>-5.749526</td>
      <td>1.047935</td>
      <td>-5.793186</td>
      <td>0.280</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.222792</td>
      <td>liminal</td>
      <td>0.0</td>
      <td>1.137167</td>
      <td>-5.776219</td>
      <td>-5.726406</td>
      <td>0.074</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-0.975431</td>
      <td>small</td>
      <td>1.0</td>
      <td>-5.745742</td>
      <td>-5.837590</td>
      <td>1.099070</td>
      <td>0.218</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>



This is `vtreat`s default behavior; to include all variables in the prepared data, set the parameter `filter_to_recommended` to False, as we show later, in the *Parameters for `MultinomialOutcomeTreatment`* section below.


## Using the Prepared Data in a Model

Of course, what we really want to do with the prepared training data is to fit a model jointly with all the (recommended) variables. 
Let's try fitting a logistic regression model to `d_prepared`.


```python
import sklearn.linear_model
import seaborn

not_variables = ['y', 'yc', 'prediction', 'prob_on_predicted_class', 'predict', 'large', 'liminal', 'small', 'prob_on_correct_class']
model_vars = [v for v in d_prepared.columns if v not in set(not_variables)]

fitter = sklearn.linear_model.LogisticRegression(
    solver = 'saga',
    penalty = 'l2',
    C = 1,
    max_iter = 1000,
    multi_class = 'multinomial')
fitter.fit(d_prepared[model_vars], d_prepared['yc'])
```




    LogisticRegression(C=1, class_weight=None, dual=False, fit_intercept=True,
                       intercept_scaling=1, l1_ratio=None, max_iter=1000,
                       multi_class='multinomial', n_jobs=None, penalty='l2',
                       random_state=None, solver='saga', tol=0.0001, verbose=0,
                       warm_start=False)




```python
# convenience functions for predicting and adding predictions to original data frame

def add_predictions(d_prepared, model_vars, fitter):
    pred = fitter.predict_proba(d_prepared[model_vars])
    classes = fitter.classes_
    d_prepared['prob_on_predicted_class'] = 0
    d_prepared['predict'] = None
    for i in range(len(classes)):
        cl = classes[i]
        d_prepared[cl] = pred[:, i]
        improved = d_prepared[cl] > d_prepared['prob_on_predicted_class']
        d_prepared.loc[improved, 'predict'] = cl
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
# now predict
d_prepared = add_predictions(d_prepared, model_vars, fitter)
d_prepared = add_value_by_column(d_prepared, 'yc', 'prob_on_correct_class')
to_print=['yc', 'predict','large','liminal','small', 'prob_on_predicted_class','prob_on_correct_class']
d_prepared[to_print].head()
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>yc</th>
      <th>predict</th>
      <th>large</th>
      <th>liminal</th>
      <th>small</th>
      <th>prob_on_predicted_class</th>
      <th>prob_on_correct_class</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>small</td>
      <td>small</td>
      <td>0.000344</td>
      <td>0.000630</td>
      <td>0.999026</td>
      <td>0.999026</td>
      <td>0.999026</td>
    </tr>
    <tr>
      <th>1</th>
      <td>small</td>
      <td>small</td>
      <td>0.000370</td>
      <td>0.437370</td>
      <td>0.562260</td>
      <td>0.562260</td>
      <td>0.56226</td>
    </tr>
    <tr>
      <th>2</th>
      <td>large</td>
      <td>large</td>
      <td>0.999188</td>
      <td>0.000550</td>
      <td>0.000261</td>
      <td>0.999188</td>
      <td>0.999188</td>
    </tr>
    <tr>
      <th>3</th>
      <td>liminal</td>
      <td>liminal</td>
      <td>0.000794</td>
      <td>0.998388</td>
      <td>0.000818</td>
      <td>0.998388</td>
      <td>0.998388</td>
    </tr>
    <tr>
      <th>4</th>
      <td>small</td>
      <td>small</td>
      <td>0.000344</td>
      <td>0.000630</td>
      <td>0.999026</td>
      <td>0.999026</td>
      <td>0.999026</td>
    </tr>
  </tbody>
</table>
</div>



Here, the columns `large`, `liminal` and `small` give the predicted probability of each target outcome and `predict` gives the predicted (most probable) class. The column `prob_on_predicted_class` returns the predicted probability of the predicted class, and `prob_on_correct_class` returns the predicted probability of the actual class.

We can compare the predictions to actual outcomes with a confusion matrix:


```python
import sklearn.metrics

print(fitter.classes_)    
sklearn.metrics.confusion_matrix(d_prepared.yc, d_prepared.predict, labels=fitter.classes_)
```

    ['large' 'liminal' 'small']





    array([[140,  35,   0],
           [  0, 113,  46],
           [  0,   0, 166]])



In the above confusion matrix, the entry `[row, column]` gives the number of true items of `class[row]` that also have prediction of `class[column]`. In other words, the entry `[1,2]` gives the number of 'large' items predicted to be 'liminal'.

Now apply the model to new data.


```python
# create the new data
dtest = make_data(450)

# prepare the new data with vtreat
dtest_prepared = transform.transform(dtest)

# apply the model to the prepared data
dtest_prepared = add_predictions(dtest_prepared, model_vars, fitter)
dtest_prepared = add_value_by_column(dtest_prepared, 'yc', 'prob_on_correct_class')

dtest_prepared[to_print].head()
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>yc</th>
      <th>predict</th>
      <th>large</th>
      <th>liminal</th>
      <th>small</th>
      <th>prob_on_predicted_class</th>
      <th>prob_on_correct_class</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>large</td>
      <td>large</td>
      <td>0.999192</td>
      <td>0.000548</td>
      <td>0.000261</td>
      <td>0.999192</td>
      <td>0.999192</td>
    </tr>
    <tr>
      <th>1</th>
      <td>liminal</td>
      <td>liminal</td>
      <td>0.465065</td>
      <td>0.534503</td>
      <td>0.000432</td>
      <td>0.534503</td>
      <td>0.534503</td>
    </tr>
    <tr>
      <th>2</th>
      <td>large</td>
      <td>liminal</td>
      <td>0.465065</td>
      <td>0.534503</td>
      <td>0.000432</td>
      <td>0.534503</td>
      <td>0.465065</td>
    </tr>
    <tr>
      <th>3</th>
      <td>large</td>
      <td>large</td>
      <td>0.999192</td>
      <td>0.000548</td>
      <td>0.000261</td>
      <td>0.999192</td>
      <td>0.999192</td>
    </tr>
    <tr>
      <th>4</th>
      <td>liminal</td>
      <td>small</td>
      <td>0.000367</td>
      <td>0.445570</td>
      <td>0.554063</td>
      <td>0.554063</td>
      <td>0.44557</td>
    </tr>
  </tbody>
</table>
</div>




```python
print(fitter.classes_)    
sklearn.metrics.confusion_matrix(dtest_prepared.yc, dtest_prepared.predict, labels=fitter.classes_)
```

    ['large' 'liminal' 'small']





    array([[ 90,  52,   0],
           [  0, 112,  41],
           [  0,   0, 155]])



## Parameters for `MultinomialOutcomeTreatment`

We've tried to set the defaults for all parameters so that `vtreat` is usable out of the box for most applications.



```python
vtreat.vtreat_parameters()
```




    {'use_hierarchical_estimate': True,
     'coders': {'clean_copy',
      'deviation_code',
      'impact_code',
      'indicator_code',
      'logit_code',
      'missing_indicator',
      'prevalence_code'},
     'filter_to_recommended': True,
     'indicator_min_fraction': 0.1,
     'cross_validation_plan': <vtreat.cross_plan.KWayCrossPlanYStratified at 0x1a219dd810>,
     'cross_validation_k': 5,
     'user_transforms': [],
     'sparse_indicators': True,
     'missingness_imputation': <function numpy.mean(a, axis=None, dtype=None, out=None, keepdims=<no value>)>}



**use_hierarchical_estimate:**: When True, uses hierarchical smoothing when estimating `logit_code` variables; when False, uses unsmoothed logistic regression.

**coders**: The types of synthetic variables that `vtreat` will (potentially) produce. See *Types of prepared variables* below.

**filter_to_recommended**: When True, prepared data only includes variables marked as "recommended" in score frame. When False, prepared data includes all variables. See the Example below.

**indicator_min_fraction**: For categorical variables, indicator variables (type `indicator_code`) are only produced for levels that are present at least `indicator_min_fraction` of the time. A consequence of this is that 1/`indicator_min_fraction` is the maximum number of indicators that will be produced for a given categorical variable. To make sure that *all* possible indicator variables are produced, set `indicator_min_fraction = 0`

**cross_validation_plan**: The cross validation method used by `vtreat`. Most people won't have to change this.

**cross_validation_k**: The number of folds to use for cross-validation

**user_transforms**: For passing in user-defined transforms for custom data preparation. Won't be needed in most situations, but see [here](https://github.com/WinVector/pyvtreat/blob/master/Examples/UserCoders/UserCoders.ipynb) for an example of applying a GAM transform to input variables.

**sparse_indicators**: When True, use a (Pandas) sparse representation for indicator variables. This representation is compatible with `sklearn`; however, it may not be compatible with other modeling packages. When False, use a dense representation.

**missingness_imputation** The function or value that `vtreat` uses to impute or "fill in" missing numerical values. The default is `numpy.mean()`. To change the imputation function or use different functions/values for different columns, see the [Imputation example](https://github.com/WinVector/pyvtreat/blob/master/Examples/Imputation/Imputation.ipynb).

### Example: Use all variables to model, not just recommended


```python
transform_all = vtreat.MultinomialOutcomeTreatment(
    outcome_name='yc',    # outcome variable
    cols_to_copy=['y'],   # columns to "carry along" but not treat as input variables
    params = vtreat.vtreat_parameters({
        'filter_to_recommended': False
    })
)  

# the variable columns in the transformed data
omit = ['x', 'y','yc']
columns = transform_all.fit_transform(d, d['yc']).columns
the_vars = list(set(columns)-set(omit))
the_vars.sort()
the_vars
```




    ['x2',
     'x_is_bad',
     'xc_is_bad',
     'xc_lev__NA_',
     'xc_lev_level_-0_5',
     'xc_lev_level_0_5',
     'xc_lev_level_1_0',
     'xc_logit_code_large',
     'xc_logit_code_liminal',
     'xc_logit_code_small',
     'xc_prevalence_code']




```python
# the variables marked "recommended" by the transform
score_frame = transform_all.score_frame_
recommended = list(score_frame.variable[score_frame.recommended].unique())
recommended.sort()
recommended
```




    ['xc_is_bad',
     'xc_lev__NA_',
     'xc_lev_level_-0_5',
     'xc_lev_level_0_5',
     'xc_lev_level_1_0',
     'xc_logit_code_large',
     'xc_logit_code_liminal',
     'xc_logit_code_small',
     'xc_prevalence_code']



Note that the prepared data produced by `fit_transform()` includes all the variables, including those that were not marked as "recommended" (if any). 

## Types of prepared variables

**clean_copy**: Produced from numerical variables: a clean numerical variable with no `NaNs` or missing values

**indicator_code**: Produced from categorical variables, one for each (common) level: for each level of the variable, indicates if that level was "on"

**prevalence_code**: Produced from categorical variables: indicates how often each level of the variable was "on"

**logit_code**: Produced from categorical variables: score from a one-dimensional "one versus rest" model of the centered output as a function of the variable. One `logit_code` variable is produced for each target class.

**missing_indicator**: Produced for both numerical and categorical variables: an indicator variable that marks when the original variable was missing or  `NaN`

**deviation_code**: not used by `MultinomialOutcomeTreatment`

**impact_code**: not used by `MultinomialOutcomeTreatment`

### Example: Produce only a subset of variable types

In this example, suppose you only want to use indicators and continuous variables in your model; 
in other words, you only want to use variables of types (`clean_copy`, `missing_indicator`, and `indicator_code`), and no `logit_code` or `prevalence_code` variables.


```python
transform_thin = vtreat.MultinomialOutcomeTreatment(
    outcome_name='yc',    # outcome variable
    cols_to_copy=['y'],   # columns to "carry along" but not treat as input variables
    params = vtreat.vtreat_parameters({
        'filter_to_recommended': False,
        'coders': {'clean_copy',
                   'missing_indicator',
                   'indicator_code',
                  }
    })
)

transform_thin.fit_transform(d, d['yc']).head()
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>y</th>
      <th>yc</th>
      <th>x_is_bad</th>
      <th>xc_is_bad</th>
      <th>x</th>
      <th>x2</th>
      <th>xc_lev_level_1_0</th>
      <th>xc_lev__NA_</th>
      <th>xc_lev_level_-0_5</th>
      <th>xc_lev_level_0_5</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-0.956311</td>
      <td>small</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>-1.088395</td>
      <td>-1.424184</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-0.671564</td>
      <td>small</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>4.107277</td>
      <td>0.427360</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.906303</td>
      <td>large</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>7.406389</td>
      <td>0.668849</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.222792</td>
      <td>liminal</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>-0.057044</td>
      <td>-0.015787</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-0.975431</td>
      <td>small</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>-0.057044</td>
      <td>-0.491017</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
transform_thin.score_frame_
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
      <td>x_is_bad</td>
      <td>x</td>
      <td>missing_indicator</td>
      <td>False</td>
      <td>True</td>
      <td>-0.051749</td>
      <td>2.480789e-01</td>
      <td>2.0</td>
      <td>0.166667</td>
      <td>False</td>
      <td>large</td>
    </tr>
    <tr>
      <th>1</th>
      <td>xc_is_bad</td>
      <td>xc</td>
      <td>missing_indicator</td>
      <td>False</td>
      <td>True</td>
      <td>-0.387438</td>
      <td>2.353230e-19</td>
      <td>2.0</td>
      <td>0.166667</td>
      <td>True</td>
      <td>large</td>
    </tr>
    <tr>
      <th>2</th>
      <td>x</td>
      <td>x</td>
      <td>clean_copy</td>
      <td>False</td>
      <td>True</td>
      <td>0.052826</td>
      <td>2.383607e-01</td>
      <td>2.0</td>
      <td>0.166667</td>
      <td>False</td>
      <td>large</td>
    </tr>
    <tr>
      <th>3</th>
      <td>x2</td>
      <td>x2</td>
      <td>clean_copy</td>
      <td>False</td>
      <td>True</td>
      <td>0.069126</td>
      <td>1.226672e-01</td>
      <td>2.0</td>
      <td>0.166667</td>
      <td>True</td>
      <td>large</td>
    </tr>
    <tr>
      <th>4</th>
      <td>xc_lev_level_1_0</td>
      <td>xc</td>
      <td>indicator_code</td>
      <td>False</td>
      <td>True</td>
      <td>0.849837</td>
      <td>1.270913e-140</td>
      <td>4.0</td>
      <td>0.083333</td>
      <td>True</td>
      <td>large</td>
    </tr>
    <tr>
      <th>5</th>
      <td>xc_lev__NA_</td>
      <td>xc</td>
      <td>indicator_code</td>
      <td>False</td>
      <td>True</td>
      <td>-0.387438</td>
      <td>2.353230e-19</td>
      <td>4.0</td>
      <td>0.083333</td>
      <td>True</td>
      <td>large</td>
    </tr>
    <tr>
      <th>6</th>
      <td>xc_lev_level_-0_5</td>
      <td>xc</td>
      <td>indicator_code</td>
      <td>False</td>
      <td>True</td>
      <td>-0.373767</td>
      <td>5.045930e-18</td>
      <td>4.0</td>
      <td>0.083333</td>
      <td>True</td>
      <td>large</td>
    </tr>
    <tr>
      <th>7</th>
      <td>xc_lev_level_0_5</td>
      <td>xc</td>
      <td>indicator_code</td>
      <td>False</td>
      <td>True</td>
      <td>0.102752</td>
      <td>2.156466e-02</td>
      <td>4.0</td>
      <td>0.083333</td>
      <td>True</td>
      <td>large</td>
    </tr>
    <tr>
      <th>8</th>
      <td>x_is_bad</td>
      <td>x</td>
      <td>missing_indicator</td>
      <td>False</td>
      <td>True</td>
      <td>0.028292</td>
      <td>5.279294e-01</td>
      <td>2.0</td>
      <td>0.166667</td>
      <td>False</td>
      <td>liminal</td>
    </tr>
    <tr>
      <th>9</th>
      <td>xc_is_bad</td>
      <td>xc</td>
      <td>missing_indicator</td>
      <td>False</td>
      <td>True</td>
      <td>-0.360534</td>
      <td>8.564773e-17</td>
      <td>2.0</td>
      <td>0.166667</td>
      <td>True</td>
      <td>liminal</td>
    </tr>
    <tr>
      <th>10</th>
      <td>x</td>
      <td>x</td>
      <td>clean_copy</td>
      <td>False</td>
      <td>True</td>
      <td>-0.060936</td>
      <td>1.736986e-01</td>
      <td>2.0</td>
      <td>0.166667</td>
      <td>False</td>
      <td>liminal</td>
    </tr>
    <tr>
      <th>11</th>
      <td>x2</td>
      <td>x2</td>
      <td>clean_copy</td>
      <td>False</td>
      <td>True</td>
      <td>0.001912</td>
      <td>9.659861e-01</td>
      <td>2.0</td>
      <td>0.166667</td>
      <td>False</td>
      <td>liminal</td>
    </tr>
    <tr>
      <th>12</th>
      <td>xc_lev_level_1_0</td>
      <td>xc</td>
      <td>indicator_code</td>
      <td>False</td>
      <td>True</td>
      <td>-0.425828</td>
      <td>1.924077e-23</td>
      <td>4.0</td>
      <td>0.083333</td>
      <td>True</td>
      <td>liminal</td>
    </tr>
    <tr>
      <th>13</th>
      <td>xc_lev__NA_</td>
      <td>xc</td>
      <td>indicator_code</td>
      <td>False</td>
      <td>True</td>
      <td>-0.360534</td>
      <td>8.564773e-17</td>
      <td>4.0</td>
      <td>0.083333</td>
      <td>True</td>
      <td>liminal</td>
    </tr>
    <tr>
      <th>14</th>
      <td>xc_lev_level_-0_5</td>
      <td>xc</td>
      <td>indicator_code</td>
      <td>False</td>
      <td>True</td>
      <td>0.140658</td>
      <td>1.615849e-03</td>
      <td>4.0</td>
      <td>0.083333</td>
      <td>True</td>
      <td>liminal</td>
    </tr>
    <tr>
      <th>15</th>
      <td>xc_lev_level_0_5</td>
      <td>xc</td>
      <td>indicator_code</td>
      <td>False</td>
      <td>True</td>
      <td>0.194241</td>
      <td>1.218632e-05</td>
      <td>4.0</td>
      <td>0.083333</td>
      <td>True</td>
      <td>liminal</td>
    </tr>
    <tr>
      <th>16</th>
      <td>x_is_bad</td>
      <td>x</td>
      <td>missing_indicator</td>
      <td>False</td>
      <td>True</td>
      <td>0.024435</td>
      <td>5.856828e-01</td>
      <td>2.0</td>
      <td>0.166667</td>
      <td>False</td>
      <td>small</td>
    </tr>
    <tr>
      <th>17</th>
      <td>xc_is_bad</td>
      <td>xc</td>
      <td>missing_indicator</td>
      <td>False</td>
      <td>True</td>
      <td>0.748935</td>
      <td>4.736195e-91</td>
      <td>2.0</td>
      <td>0.166667</td>
      <td>True</td>
      <td>small</td>
    </tr>
    <tr>
      <th>18</th>
      <td>x</td>
      <td>x</td>
      <td>clean_copy</td>
      <td>False</td>
      <td>True</td>
      <td>0.006755</td>
      <td>8.802327e-01</td>
      <td>2.0</td>
      <td>0.166667</td>
      <td>False</td>
      <td>small</td>
    </tr>
    <tr>
      <th>19</th>
      <td>x2</td>
      <td>x2</td>
      <td>clean_copy</td>
      <td>False</td>
      <td>True</td>
      <td>-0.071903</td>
      <td>1.083077e-01</td>
      <td>2.0</td>
      <td>0.166667</td>
      <td>True</td>
      <td>small</td>
    </tr>
    <tr>
      <th>20</th>
      <td>xc_lev_level_1_0</td>
      <td>xc</td>
      <td>indicator_code</td>
      <td>False</td>
      <td>True</td>
      <td>-0.439636</td>
      <td>4.791039e-25</td>
      <td>4.0</td>
      <td>0.083333</td>
      <td>True</td>
      <td>small</td>
    </tr>
    <tr>
      <th>21</th>
      <td>xc_lev__NA_</td>
      <td>xc</td>
      <td>indicator_code</td>
      <td>False</td>
      <td>True</td>
      <td>0.748935</td>
      <td>4.736195e-91</td>
      <td>4.0</td>
      <td>0.083333</td>
      <td>True</td>
      <td>small</td>
    </tr>
    <tr>
      <th>22</th>
      <td>xc_lev_level_-0_5</td>
      <td>xc</td>
      <td>indicator_code</td>
      <td>False</td>
      <td>True</td>
      <td>0.239464</td>
      <td>5.952274e-08</td>
      <td>4.0</td>
      <td>0.083333</td>
      <td>True</td>
      <td>small</td>
    </tr>
    <tr>
      <th>23</th>
      <td>xc_lev_level_0_5</td>
      <td>xc</td>
      <td>indicator_code</td>
      <td>False</td>
      <td>True</td>
      <td>-0.296154</td>
      <td>1.401007e-11</td>
      <td>4.0</td>
      <td>0.083333</td>
      <td>True</td>
      <td>small</td>
    </tr>
  </tbody>
</table>
</div>



## Deriving the Default Thresholds

While machine learning algorithms are generally tolerant to a reasonable number of irrelevant or noise variables, too many irrelevant variables can lead to serious overfit; see [this article](http://www.win-vector.com/blog/2014/02/bad-bayes-an-example-of-why-you-need-hold-out-testing/) for an extreme example, one we call "Bad Bayes". The default threshold is an attempt to eliminate obviously irrelevant variables early.

Imagine that you have a pure noise dataset, where none of the *n* inputs are related to the output. If you treat each variable as a one-variable model for the output, and look at the significances of each model, these significance-values will be uniformly distributed in the range [0:1]. You want to pick a weakest possible significance threshold that eliminates as many noise variables as possible. A moment's thought should convince you that a threshold of *1/n* allows only one variable through, in expectation. 

This leads to the general-case heuristic that a significance threshold of *1/n* on your variables should allow only one irrelevant variable through, in expectation (along with all the relevant variables). Hence, *1/n* used to be our recommended threshold, when we developed the R version of `vtreat`.

We noticed, however, that this biases the filtering against numerical variables, since there are at most two derived variables (of types *clean_copy* and *missing_indicator* for every numerical variable in the original data. Categorical variables, on the other hand, are expanded to many derived variables: several indicators (one for every common level), plus a *logit_code* and a *prevalence_code*. So we now reweight the thresholds. 

Suppose you have a (treated) data set with *ntreat* different types of `vtreat` variables (`clean_copy`, `indicator_code`, etc).
There are *nT* variables of type *T*. Then the default threshold for all the variables of type *T* is *1/(ntreat nT)*. This reweighting  helps to reduce the bias against any particular type of variable. The heuristic is still that the set of recommended variables will allow at most one noise variable into the set of candidate variables.

As noted above, because `vtreat` estimates variable significances using linear methods by default, some variables with a non-linear relationship  to the output may fail to pass the threshold. Setting the `filter_to_recommended` parameter to False will keep all derived variables in the treated frame, for the data scientist to filter (or not) as they will.



## Conclusion

In all cases (classification, regression, unsupervised, and multinomial classification) the intent is that `vtreat` transforms are essentially one liners.

The preparation commands are organized as follows:

 * **Regression**: [`R` regression example](https://github.com/WinVector/vtreat/blob/master/Examples/Regression/Regression.md), [`Python` regression example](https://github.com/WinVector/pyvtreat/blob/master/Examples/Regression/Regression.md).
 * **Classification**: [`R` classification example](https://github.com/WinVector/vtreat/blob/master/Examples/Classification/Classification.md), [`Python` classification  example](https://github.com/WinVector/pyvtreat/blob/master/Examples/Classification/Classification.md).
 * **Unsupervised tasks**: [`R` unsupervised example](https://github.com/WinVector/vtreat/blob/master/Examples/Unsupervised/Unsupervised.md), [`Python` unsupervised example](https://github.com/WinVector/pyvtreat/blob/master/Examples/Unsupervised/Unsupervised.md).
 * **Multinomial classification**: [`R` multinomial classification example](https://github.com/WinVector/vtreat/blob/master/Examples/Multinomial/MultinomialExample.md), [`Python` multinomial classification example](https://github.com/WinVector/pyvtreat/blob/master/Examples/Multinomial/MultinomialExample.md).
 
The shared structure of the `score_frame` is discussed here:

  * **Score Frame** [score_frame_](https://github.com/WinVector/pyvtreat/blob/master/Examples/ScoreFrame/ScoreFrame.md).

These current revisions of the examples are designed to be small, yet complete.  So as a set they have some overlap, but the user can rely mostly on a single example for a single task type.

