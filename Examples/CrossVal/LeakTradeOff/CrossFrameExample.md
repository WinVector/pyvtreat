```python

```

# Cross-Methods are a Leak/Variance Trade-Off

  * John Mount, [Win Vector LLC](http://www.win-vector.com/)
  * Nina Zumel, [Win Vector LLC](http://www.win-vector.com/)
  * March 10, 2020
  * [https://github.com/WinVector/pyvtreat/blob/master/Examples/CrossVal/LeakTradeOff/](https://github.com/WinVector/pyvtreat/blob/master/Examples/CrossVal/LeakTradeOff/)

## Introduction

Cross-methods such as [cross-validation](https://en.wikipedia.org/wiki/Cross-validation_%28statistics%29), and [cross-prediction](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_val_predict.html#sklearn.model_selection.cross_val_predict) are effective tools for many machine learning, statisitics, and data science related applications. They are useful for parameter selection, model selection, impact/target encoding of high cardinality variables, stacking models, and super learning. They are more statistically efficient than partitioning training data into calibration/training/holdout sets, but do not satisfy the full exchangeability conditions that full hold-out methods have. This introduces some additional statistical trade-offs when using cross-methods, beyond the obvious increases in computational cost.

Specifically, cross-methods can introduce an information leak into the modeling process. This information leak will be the subject of this post.

To show the information leak, we will use a simple artificial problem where there is no relation between the proposed explanatory variable(s) and the output variable. This example is in the spirit of our previous article, [Bad Bayes: an example of why you need hold-out testing](http://www.win-vector.com/blog/2014/02/bad-bayes-an-example-of-why-you-need-hold-out-testing/), as well as the paper by Claudia Perlich Grzegorz Swirszcz, "On Cross-Validation and Stacking: Building Seemingly Predictive Models On Random Data", SIGKDD Explorations, volume 12, number 2, 2010.

We will demonstrate that even in this situation, target-encoding (or conditionally re-encoding) categorical variables prior to the model-fitting step leaks information about the dependent variable. *This is true even when using cross-methods*. This leaked information may cause the downstream modeling step to treat noise variables as informative ones, leading to overfit.

Finally, we will conclude with a more realistic case: a combination of useless and useful explanatory variables.  For our last example we will use our recommended package [`vtreat`](https://github.com/WinVector/pyvtreat) ([available for `Python`](https://github.com/WinVector/pyvtreat) and [available for `R`](https://github.com/WinVector/vtreat)). The `vtreat` package manages impact coding, cross-validation, and reporting in a convenient unit.

## Preliminaries


We will work some regression examples using `Python`/`Pandas`.  In addition to calling out what to look for in each result, we will add "`assert`" statements to doublecheck the results as we present them.

First we import our packages and modules, and set our pseudo-random state to make the result more easily reproducible.


```python
import re

# https://numpy.org
import numpy

# https://pandas.pydata.org
import pandas

# https://seaborn.pydata.org
import seaborn

# https://matplotlib.org
import matplotlib.pyplot

# https://scikit-learn.org/
import sklearn.metrics
import sklearn.linear_model
import sklearn.model_selection
# https://contrib.scikit-learn.org/categorical-encoding/targetencoder.html
import category_encoders

# https://www.statsmodels.org/
import statsmodels.api

# https://github.com/WinVector/pyvtreat/blob/master/Examples/CrossVal/LeakTradeOff/break_cross_val.py
# https://github.com/WinVector/data_algebra
from break_cross_val import mk_data, TransformerAdapter, Container, solve_for_partition, collect_relations

# https://github.com/WinVector/pyvtreat
import vtreat
```


```python
numpy.random.seed(2020)
prng = numpy.random.RandomState(numpy.random.randint(2**32))
```

### Initial Example Data: The Pure Noise Case

Now we create some example data. The data has 100 rows of 10 categorical variables, with 50 levels each, and one constant variable.


```python
d_example, y_example = mk_data(
    nrow=100,
    n_noise_var=10,
    n_noise_level=50,
    n_signal_var=0)
```


```python
d_example
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>const_col</th>
      <th>noise_0</th>
      <th>noise_1</th>
      <th>noise_2</th>
      <th>noise_3</th>
      <th>noise_4</th>
      <th>noise_5</th>
      <th>noise_6</th>
      <th>noise_7</th>
      <th>noise_8</th>
      <th>noise_9</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>a</td>
      <td>nl_14</td>
      <td>nl_48</td>
      <td>nl_0</td>
      <td>nl_36</td>
      <td>nl_11</td>
      <td>nl_37</td>
      <td>nl_14</td>
      <td>nl_0</td>
      <td>nl_28</td>
      <td>nl_41</td>
    </tr>
    <tr>
      <th>1</th>
      <td>a</td>
      <td>nl_4</td>
      <td>nl_45</td>
      <td>nl_45</td>
      <td>nl_6</td>
      <td>nl_25</td>
      <td>nl_10</td>
      <td>nl_18</td>
      <td>nl_37</td>
      <td>nl_19</td>
      <td>nl_38</td>
    </tr>
    <tr>
      <th>2</th>
      <td>a</td>
      <td>nl_36</td>
      <td>nl_38</td>
      <td>nl_9</td>
      <td>nl_34</td>
      <td>nl_29</td>
      <td>nl_49</td>
      <td>nl_18</td>
      <td>nl_14</td>
      <td>nl_0</td>
      <td>nl_7</td>
    </tr>
    <tr>
      <th>3</th>
      <td>a</td>
      <td>nl_25</td>
      <td>nl_31</td>
      <td>nl_18</td>
      <td>nl_36</td>
      <td>nl_41</td>
      <td>nl_30</td>
      <td>nl_5</td>
      <td>nl_31</td>
      <td>nl_21</td>
      <td>nl_21</td>
    </tr>
    <tr>
      <th>4</th>
      <td>a</td>
      <td>nl_14</td>
      <td>nl_24</td>
      <td>nl_5</td>
      <td>nl_5</td>
      <td>nl_1</td>
      <td>nl_23</td>
      <td>nl_27</td>
      <td>nl_42</td>
      <td>nl_34</td>
      <td>nl_6</td>
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
      <th>95</th>
      <td>a</td>
      <td>nl_17</td>
      <td>nl_13</td>
      <td>nl_33</td>
      <td>nl_13</td>
      <td>nl_49</td>
      <td>nl_31</td>
      <td>nl_35</td>
      <td>nl_32</td>
      <td>nl_17</td>
      <td>nl_35</td>
    </tr>
    <tr>
      <th>96</th>
      <td>a</td>
      <td>nl_49</td>
      <td>nl_43</td>
      <td>nl_14</td>
      <td>nl_3</td>
      <td>nl_32</td>
      <td>nl_47</td>
      <td>nl_27</td>
      <td>nl_23</td>
      <td>nl_30</td>
      <td>nl_36</td>
    </tr>
    <tr>
      <th>97</th>
      <td>a</td>
      <td>nl_33</td>
      <td>nl_47</td>
      <td>nl_42</td>
      <td>nl_17</td>
      <td>nl_37</td>
      <td>nl_48</td>
      <td>nl_24</td>
      <td>nl_29</td>
      <td>nl_44</td>
      <td>nl_37</td>
    </tr>
    <tr>
      <th>98</th>
      <td>a</td>
      <td>nl_49</td>
      <td>nl_28</td>
      <td>nl_32</td>
      <td>nl_40</td>
      <td>nl_32</td>
      <td>nl_26</td>
      <td>nl_33</td>
      <td>nl_4</td>
      <td>nl_27</td>
      <td>nl_41</td>
    </tr>
    <tr>
      <th>99</th>
      <td>a</td>
      <td>nl_21</td>
      <td>nl_8</td>
      <td>nl_46</td>
      <td>nl_13</td>
      <td>nl_30</td>
      <td>nl_8</td>
      <td>nl_28</td>
      <td>nl_0</td>
      <td>nl_1</td>
      <td>nl_4</td>
    </tr>
  </tbody>
</table>
<p>100 rows × 11 columns</p>
</div>




```python
y_example
```




    array([ 0.35640831, -1.02966661,  0.10937507, -1.22461964,  1.98078615,
           -0.08133883, -0.8407814 ,  0.17419664, -0.26695962,  1.81730992,
           -0.40031323, -0.01591122,  0.06183108, -2.63659715, -1.23370528,
           -1.5720175 , -0.30802748, -1.12459106,  0.80999499, -0.58387749,
            0.56838667,  0.90359773,  0.98001598,  0.94958149, -0.15583399,
            1.3929476 ,  0.4957971 , -0.18760574, -0.97232061, -0.1383119 ,
           -1.98764929,  0.42246929, -0.25438059,  0.64496689, -0.12015076,
           -0.48352493,  0.53825049,  1.23793055,  0.14021035,  1.38925737,
            0.18708701,  0.45131922,  1.80806884, -0.51693141,  0.87514908,
            0.36805093, -0.36548753, -1.56253055, -0.88706849,  0.57927198,
           -0.2806769 , -0.07133204,  0.74667248, -0.81331984,  0.66688814,
            1.03676875,  1.00533415,  0.83378592, -0.81403847, -2.26635425,
           -0.99387029, -0.48577153, -0.51869578,  0.17533136, -0.79042072,
           -0.88466057,  0.21123103,  1.68172973,  0.41984886, -2.41421325,
            3.09279159,  0.35416444, -0.235704  , -1.06207208, -0.65270214,
            1.27772083,  0.06024998,  0.76973631,  0.66210325,  0.45382781,
            1.35590224,  0.84735606,  1.26620742,  1.24446983,  0.80523392,
            1.08266673,  0.66324872,  0.61433843, -0.49462885,  1.62266163,
            0.53028485,  0.48833726,  0.31907274, -0.04827782, -1.02430346,
           -0.4880873 , -0.23963282, -1.74395751,  3.14710139, -2.43246331])



In this data all "noise variables" are generated independently of the outcome or dependent variable `y_example`.  `const_col` is a variable that does not vary: it always has the value "`a`". We want to fit a linear regression model for `y_example` to the data. Of course, such a model should predict nothing, since there is no relationship between the inputs and the output.

For our examples, we will re-encode each categorical variable into a single numerical encoding (here called a *target encoding*) where each level of the categorical variable is encoded as the (possibly smoothed) conditional mean of the *y* variable in the training set. Target encoding (and the similar [*impact coding*](https://www.r-bloggers.com/modeling-trick-impact-coding-of-categorical-variables-with-many-levels/)) can be useful when modeling with very high cardinality categorical variables (that is, categorical variables with a very large number of levels), or when modeling with many moderate cardinality categorical variables. In either case, target encoding, when used properly, is good for managing the "variable blowup" caused by encoding a single categorical variable into multiple indicator variables.

For more discussion, see [this article](http://www.win-vector.com/blog/2012/07/modeling-trick-impact-coding-of-categorical-variables-with-many-levels/) and [this one](http://www.win-vector.com/blog/2019/11/when-cross-validation-is-more-powerful-than-regularization/). 

## The Case of No Cross-Method

For this first example we will use [`category_encoders.target_encoder.TargetEncoder`](https://contrib.scikit-learn.org/categorical-encoding/targetencoder.html) to re-encode our categorical variables prior to a linear regression.  This target encoder re-encodes categorical variables as smoothed conditional estimates.

On its own, the encoder is not cross-validated. This means high complexity explanatory variables hide their true number of degrees of freedom and leak information.  This causes the variables to over-fit on training data even when they are useless on test data, as we will demonstrate below.


```python
te0 = category_encoders.target_encoder.TargetEncoder()
d_coded_0 = te0. \
    fit_transform(d_example, y_example)
d_coded_0
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>const_col</th>
      <th>noise_0</th>
      <th>noise_1</th>
      <th>noise_2</th>
      <th>noise_3</th>
      <th>noise_4</th>
      <th>noise_5</th>
      <th>noise_6</th>
      <th>noise_7</th>
      <th>noise_8</th>
      <th>noise_9</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.089719</td>
      <td>0.878442</td>
      <td>-0.279115</td>
      <td>-0.134412</td>
      <td>-0.293227</td>
      <td>0.296913</td>
      <td>-1.324166</td>
      <td>0.448744</td>
      <td>-0.547689</td>
      <td>0.089719</td>
      <td>1.090799</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.089719</td>
      <td>-0.864603</td>
      <td>0.074604</td>
      <td>0.372476</td>
      <td>-0.707656</td>
      <td>-0.025692</td>
      <td>-0.310375</td>
      <td>-0.312264</td>
      <td>0.089719</td>
      <td>-0.509022</td>
      <td>0.035622</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.089719</td>
      <td>0.089719</td>
      <td>0.122710</td>
      <td>0.147765</td>
      <td>-0.474352</td>
      <td>1.145241</td>
      <td>0.089719</td>
      <td>-0.312264</td>
      <td>0.491677</td>
      <td>0.554715</td>
      <td>0.127783</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.089719</td>
      <td>-0.179738</td>
      <td>0.039331</td>
      <td>-0.347868</td>
      <td>-0.293227</td>
      <td>-0.964601</td>
      <td>-0.178340</td>
      <td>0.034212</td>
      <td>-0.236233</td>
      <td>-0.637285</td>
      <td>0.420169</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.089719</td>
      <td>0.878442</td>
      <td>0.536216</td>
      <td>-0.276830</td>
      <td>0.770188</td>
      <td>1.212039</td>
      <td>0.640521</td>
      <td>0.250698</td>
      <td>0.384876</td>
      <td>1.255121</td>
      <td>-0.119833</td>
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
      <th>95</th>
      <td>0.089719</td>
      <td>0.073256</td>
      <td>0.481162</td>
      <td>-0.509693</td>
      <td>-1.043418</td>
      <td>0.129405</td>
      <td>-0.038776</td>
      <td>-0.180355</td>
      <td>0.089719</td>
      <td>0.092802</td>
      <td>0.089719</td>
    </tr>
    <tr>
      <th>96</th>
      <td>0.089719</td>
      <td>1.086894</td>
      <td>0.038942</td>
      <td>-0.099632</td>
      <td>0.053167</td>
      <td>0.847704</td>
      <td>0.089719</td>
      <td>0.250698</td>
      <td>0.232613</td>
      <td>-0.200089</td>
      <td>-0.125004</td>
    </tr>
    <tr>
      <th>97</th>
      <td>0.089719</td>
      <td>-0.113104</td>
      <td>0.089719</td>
      <td>0.089719</td>
      <td>-0.128602</td>
      <td>-1.213415</td>
      <td>-0.569695</td>
      <td>-0.185133</td>
      <td>0.089719</td>
      <td>0.089719</td>
      <td>-0.264294</td>
    </tr>
    <tr>
      <th>98</th>
      <td>0.089719</td>
      <td>1.086894</td>
      <td>0.885041</td>
      <td>1.494380</td>
      <td>0.182377</td>
      <td>0.847704</td>
      <td>0.933494</td>
      <td>1.099114</td>
      <td>1.340374</td>
      <td>0.737929</td>
      <td>1.090799</td>
    </tr>
    <tr>
      <th>99</th>
      <td>0.089719</td>
      <td>-1.162300</td>
      <td>-1.210592</td>
      <td>0.089719</td>
      <td>-1.043418</td>
      <td>0.089719</td>
      <td>-0.288454</td>
      <td>-1.162300</td>
      <td>-0.547689</td>
      <td>-0.466596</td>
      <td>-0.963628</td>
    </tr>
  </tbody>
</table>
<p>100 rows × 11 columns</p>
</div>




```python
overfit_model = statsmodels.api.OLS(y_example, d_coded_0)
overfit_result = overfit_model.fit()

r2 = sklearn.metrics.r2_score(
    y_true=y_example, 
    y_pred=overfit_result.predict(d_coded_0))
assert r2 > 0.7
r2

overfit_result.summary()
```




<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>            <td>y</td>        <th>  R-squared:         </th> <td>   0.830</td>
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th> <td>   0.811</td>
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th> <td>   43.52</td>
</tr>
<tr>
  <th>Date:</th>             <td>Tue, 10 Mar 2020</td> <th>  Prob (F-statistic):</th> <td>5.29e-30</td>
</tr>
<tr>
  <th>Time:</th>                 <td>18:01:32</td>     <th>  Log-Likelihood:    </th> <td> -60.214</td>
</tr>
<tr>
  <th>No. Observations:</th>      <td>   100</td>      <th>  AIC:               </th> <td>   142.4</td>
</tr>
<tr>
  <th>Df Residuals:</th>          <td>    89</td>      <th>  BIC:               </th> <td>   171.1</td>
</tr>
<tr>
  <th>Df Model:</th>              <td>    10</td>      <th>                     </th>     <td> </td>   
</tr>
<tr>
  <th>Covariance Type:</th>      <td>nonrobust</td>    <th>                     </th>     <td> </td>   
</tr>
</table>
<table class="simpletable">
<tr>
      <td></td>         <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>const_col</th> <td>   -2.1941</td> <td>    0.558</td> <td>   -3.930</td> <td> 0.000</td> <td>   -3.303</td> <td>   -1.085</td>
</tr>
<tr>
  <th>noise_0</th>   <td>    0.3103</td> <td>    0.116</td> <td>    2.665</td> <td> 0.009</td> <td>    0.079</td> <td>    0.542</td>
</tr>
<tr>
  <th>noise_1</th>   <td>    0.2227</td> <td>    0.114</td> <td>    1.954</td> <td> 0.054</td> <td>   -0.004</td> <td>    0.449</td>
</tr>
<tr>
  <th>noise_2</th>   <td>    0.6015</td> <td>    0.127</td> <td>    4.754</td> <td> 0.000</td> <td>    0.350</td> <td>    0.853</td>
</tr>
<tr>
  <th>noise_3</th>   <td>    0.4393</td> <td>    0.126</td> <td>    3.499</td> <td> 0.001</td> <td>    0.190</td> <td>    0.689</td>
</tr>
<tr>
  <th>noise_4</th>   <td>    0.4852</td> <td>    0.102</td> <td>    4.760</td> <td> 0.000</td> <td>    0.283</td> <td>    0.688</td>
</tr>
<tr>
  <th>noise_5</th>   <td>    0.0554</td> <td>    0.121</td> <td>    0.459</td> <td> 0.647</td> <td>   -0.184</td> <td>    0.295</td>
</tr>
<tr>
  <th>noise_6</th>   <td>    0.1714</td> <td>    0.115</td> <td>    1.490</td> <td> 0.140</td> <td>   -0.057</td> <td>    0.400</td>
</tr>
<tr>
  <th>noise_7</th>   <td>    0.2075</td> <td>    0.127</td> <td>    1.635</td> <td> 0.106</td> <td>   -0.045</td> <td>    0.460</td>
</tr>
<tr>
  <th>noise_8</th>   <td>    0.4540</td> <td>    0.109</td> <td>    4.151</td> <td> 0.000</td> <td>    0.237</td> <td>    0.671</td>
</tr>
<tr>
  <th>noise_9</th>   <td>    0.4056</td> <td>    0.107</td> <td>    3.792</td> <td> 0.000</td> <td>    0.193</td> <td>    0.618</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td> 5.411</td> <th>  Durbin-Watson:     </th> <td>   1.728</td>
</tr>
<tr>
  <th>Prob(Omnibus):</th> <td> 0.067</td> <th>  Jarque-Bera (JB):  </th> <td>   5.454</td>
</tr>
<tr>
  <th>Skew:</th>          <td> 0.346</td> <th>  Prob(JB):          </th> <td>  0.0654</td>
</tr>
<tr>
  <th>Kurtosis:</th>      <td> 3.910</td> <th>  Cond. No.          </th> <td>    12.1</td>
</tr>
</table><br/><br/>Warnings:<br/>[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.



Notice the summary estimates a good adjusted R-squared (around 0.8), and finds many of the noise coefficients to be significant.  This is because the re-encoding step over-fit the data before we even got to the ordinary least squares regression.  We can confirm this by showing the model doesn't work on identically generated fresh data.


```python
d_test, y_test = mk_data(
    nrow=100,
    n_noise_var=10,
    n_noise_level=50,
    n_signal_var=0)

d_test_coded = te0.transform(d_test)

r2 = sklearn.metrics.r2_score(
    y_true=y_test, 
    y_pred=overfit_result.predict(d_test_coded))
assert r2 < 0.2

r2
```




    -0.25111316272369755



This can be an insidious issue: over-estimating model performance, and also allowing complex noise variables to outcompete low-complexity but actually useful explanatory variables.

Our advice to avoid this issue is to either use separate data for encoding and modeling, or use a cross-method when re-encoding the categorical variables.  This is what we are *very* careful to do correctly in [R vtreat](https://github.com/WinVector/vtreat) and [Python vtreat](https://github.com/WinVector/pyvtreat).

## Leave-One-Out Cross-Methods Data Leak

Let's take a quick look at how problems can arise even when using cross-methods.

In our opinion, to minimize data leaks one should avoid using a deterministic cross method plan, which can often pass through undesirable incidental structure in the data.  As such, we advise against using a leave-one-out cross-plan in production.

Leave-one-out leaks information in many places, including even in a constant column (a column that does not vary). To see this, let's try to fit a model for `y_example` using only `const_col`. First, we target-code `const_col`. We don't *need* to cross-validate a constant, but it is a problem that it doesn't work.


```python
cv_one_out = sklearn.model_selection.LeaveOneOut()

# TransformAdapter adapts the TargetEncoder object for cross-methods
te2 = TransformerAdapter(
    category_encoders.target_encoder.TargetEncoder())

# Build the cross-validated encoding of the training data
# For use in training the model
cross_frame_0 = sklearn.model_selection.cross_val_predict(
    te2, 
    d_example[['const_col']], # just look at the constant column
    y_example, 
    cv=cv_one_out)
cross_frame_0 = pandas.DataFrame(cross_frame_0)
cross_frame_0.columns = ['const_col']

# This is the "transformed" training data
cross_frame_0
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>const_col</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.087026</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.101026</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.089521</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.102996</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.070618</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
    </tr>
    <tr>
      <th>95</th>
      <td>0.095556</td>
    </tr>
    <tr>
      <th>96</th>
      <td>0.093046</td>
    </tr>
    <tr>
      <th>97</th>
      <td>0.108241</td>
    </tr>
    <tr>
      <th>98</th>
      <td>0.058837</td>
    </tr>
    <tr>
      <th>99</th>
      <td>0.115196</td>
    </tr>
  </tbody>
</table>
<p>100 rows × 1 columns</p>
</div>



Notice that the re-encoding of the constant column varies per row. This is because in cross-methods the prediction is a function of the input *plus* the fold-id; it is *not* a function of the input alone.  For leave-one-out encoding of a constant, the encoding is `code[i] = (m * mean(y) - y[i])/(m - 1)`: the grand mean with the `i`-th row held out.  But this is equal to `m * mean(y) / (m - 1) - y[i] / (m - 1)`.  This means during training it is trivial to read off the `y`-values from the re-encoded constant column. To see this, we fit a linear regression model for `y_example` as a function of `cross_frame_0`, the re-encoded training data.


```python
# fit a linear regression model as a function of the 
# target encoded training data
overfit_model_2 = statsmodels.api.OLS(
    y_example, 
    statsmodels.api.add_constant(   # add the DC intercept term
        cross_frame_0.values, 
        has_constant='add'))
overfit_result_2 = overfit_model_2.fit()

# calculate R-squared
r2 = sklearn.metrics.r2_score(
    y_true=y_example, 
    y_pred=overfit_result_2.predict(
        statsmodels.api.add_constant(
            cross_frame_0.values,
            has_constant='add')))
assert r2 > 0.9

overfit_result_2.summary()
```




<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>            <td>y</td>        <th>  R-squared:         </th> <td>   1.000</td> 
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th> <td>   1.000</td> 
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th> <td>1.305e+31</td>
</tr>
<tr>
  <th>Date:</th>             <td>Tue, 10 Mar 2020</td> <th>  Prob (F-statistic):</th>  <td>  0.00</td>  
</tr>
<tr>
  <th>Time:</th>                 <td>18:01:34</td>     <th>  Log-Likelihood:    </th> <td>  3204.2</td> 
</tr>
<tr>
  <th>No. Observations:</th>      <td>   100</td>      <th>  AIC:               </th> <td>  -6404.</td> 
</tr>
<tr>
  <th>Df Residuals:</th>          <td>    98</td>      <th>  BIC:               </th> <td>  -6399.</td> 
</tr>
<tr>
  <th>Df Model:</th>              <td>     1</td>      <th>                     </th>     <td> </td>    
</tr>
<tr>
  <th>Covariance Type:</th>      <td>nonrobust</td>    <th>                     </th>     <td> </td>    
</tr>
</table>
<table class="simpletable">
<tr>
    <td></td>       <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>const</th> <td>    8.9719</td> <td> 2.48e-15</td> <td> 3.62e+15</td> <td> 0.000</td> <td>    8.972</td> <td>    8.972</td>
</tr>
<tr>
  <th>x1</th>    <td>  -99.0000</td> <td> 2.74e-14</td> <td>-3.61e+15</td> <td> 0.000</td> <td>  -99.000</td> <td>  -99.000</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td> 0.122</td> <th>  Durbin-Watson:     </th> <td>   0.324</td>
</tr>
<tr>
  <th>Prob(Omnibus):</th> <td> 0.941</td> <th>  Jarque-Bera (JB):  </th> <td>   0.266</td>
</tr>
<tr>
  <th>Skew:</th>          <td>-0.066</td> <th>  Prob(JB):          </th> <td>   0.876</td>
</tr>
<tr>
  <th>Kurtosis:</th>      <td> 2.784</td> <th>  Cond. No.          </th> <td>    93.1</td>
</tr>
</table><br/><br/>Warnings:<br/>[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.



What we see here is that we have apparently fit a perfect linear regression model for `y_example` using only a (re-encoded) constant input!

A crucial point to notice is that the regression used large magnitude (and negative) coefficients.  This is because there is a data leak, but it is low magnitude.  So to exploit the data leak we have to scale it up quite a bit. This is typical: many results in the literature that show the efficacy of cross-methods do so by proving that with high probability that method is very near a correct result *in norm*: that is, the norm of the difference between the cross-validated result and the true result is small. Our encoding was close to the true result, in this sense, but still represented a data leak, one that linear regression was able to exploit.

Again, this chimeric "well-fit model" will be useless on new data. Let's try it.

To try to work with new data, we want the target encoder to be fit on all the original training data. Note that the [`vtreat`](https://github.com/WinVector/pyvtreat) package, which we will discuss later, does not require this extra step.


```python
# refit the target encoder on original data for later use
te2.fit(d_example[['const_col']], y_example) 
```




    TransformerAdapter(model=TargetEncoder(cols=['const_col'], drop_invariant=False,
                                           handle_missing='value',
                                           handle_unknown='value',
                                           min_samples_leaf=1, return_df=True,
                                           smoothing=1.0, verbose=0))



Now we can apply the encoder to  new data.


```python
# now target encode the new test data
d_test_coded_2 = te2.transform(d_test[['const_col']])

overfit_test_pred_2 = overfit_result_2.predict(
    statsmodels.api.add_constant(
        d_test_coded_2.values, 
        has_constant='add'))

r2 = sklearn.metrics.r2_score(y_true=y_test, y_pred=overfit_test_pred_2)
assert r2 < 0.2

r2
```




    -0.012108850282850137



The leak we demonstrated above is one of the reasons `vtreat` uses impact codes (conditional difference from the mean) instead of target codes (conditional means). With cross-validated impact coding, constant variables will *always* code to zero, effectively identifying them as uninformative.

## Cross Method Done Correctly 

Now lets look at shuffled (pseudo-random) `k`-way cross method version of the target encoding.

For this example we look at all the input columns, and we will use 3-fold cross method. For a 3-fold cross-plan the bias from any one column is small, but many columns together will leak information.  We will demonstrate this next.


```python
# Build a shuffled cross-plan
# http://www.win-vector.com/blog/2020/03/python-data-science-tip-dont-use-default-cross-validation-settings/
cvstrat = sklearn.model_selection.KFold(
    shuffle=True, 
    n_splits=3,
    random_state=prng)

te = category_encoders.target_encoder.TargetEncoder()
cross_frame = sklearn.model_selection.cross_val_predict(
    TransformerAdapter(te), 
    d_example, 
    y_example, 
    cv=cvstrat)

# Build the transformed training data
cross_frame = pandas.DataFrame(cross_frame)
cross_frame.rename(
    columns={i: d_example.columns[i] for i in
             range(d_example.shape[1])}, inplace=True)

cross_frame
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>const_col</th>
      <th>noise_0</th>
      <th>noise_1</th>
      <th>noise_2</th>
      <th>noise_3</th>
      <th>noise_4</th>
      <th>noise_5</th>
      <th>noise_6</th>
      <th>noise_7</th>
      <th>noise_8</th>
      <th>noise_9</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.055096</td>
      <td>0.055096</td>
      <td>-0.572658</td>
      <td>0.034988</td>
      <td>0.055096</td>
      <td>0.037668</td>
      <td>0.055096</td>
      <td>0.055096</td>
      <td>0.055096</td>
      <td>0.055096</td>
      <td>0.055096</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.053041</td>
      <td>-0.762554</td>
      <td>0.687571</td>
      <td>0.053041</td>
      <td>0.053041</td>
      <td>0.053041</td>
      <td>0.054035</td>
      <td>0.053041</td>
      <td>0.053041</td>
      <td>-0.711652</td>
      <td>0.053041</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.055096</td>
      <td>0.055096</td>
      <td>0.055096</td>
      <td>0.055096</td>
      <td>0.055096</td>
      <td>0.055096</td>
      <td>0.055096</td>
      <td>0.055096</td>
      <td>0.573658</td>
      <td>0.055096</td>
      <td>0.055096</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.160505</td>
      <td>0.160505</td>
      <td>0.160505</td>
      <td>-0.055440</td>
      <td>0.160505</td>
      <td>-0.400298</td>
      <td>0.160505</td>
      <td>0.160505</td>
      <td>0.183377</td>
      <td>-0.246219</td>
      <td>0.468289</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.160505</td>
      <td>0.160505</td>
      <td>0.040946</td>
      <td>0.160505</td>
      <td>0.160505</td>
      <td>0.814798</td>
      <td>0.231655</td>
      <td>-0.258593</td>
      <td>0.160505</td>
      <td>0.160505</td>
      <td>-0.843376</td>
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
      <th>95</th>
      <td>0.160505</td>
      <td>0.160505</td>
      <td>0.160505</td>
      <td>0.160505</td>
      <td>0.160505</td>
      <td>0.369370</td>
      <td>0.109383</td>
      <td>0.160505</td>
      <td>0.160505</td>
      <td>-0.040279</td>
      <td>0.160505</td>
    </tr>
    <tr>
      <th>96</th>
      <td>0.053041</td>
      <td>0.053041</td>
      <td>0.053041</td>
      <td>0.215618</td>
      <td>0.053041</td>
      <td>1.096685</td>
      <td>0.053041</td>
      <td>0.669725</td>
      <td>0.053041</td>
      <td>-0.377680</td>
      <td>-0.067086</td>
    </tr>
    <tr>
      <th>97</th>
      <td>0.160505</td>
      <td>0.160505</td>
      <td>0.160505</td>
      <td>0.160505</td>
      <td>0.192124</td>
      <td>-0.843376</td>
      <td>0.160505</td>
      <td>0.160505</td>
      <td>0.160505</td>
      <td>0.160505</td>
      <td>0.100504</td>
    </tr>
    <tr>
      <th>98</th>
      <td>0.055096</td>
      <td>0.055096</td>
      <td>0.055096</td>
      <td>0.055096</td>
      <td>-0.355399</td>
      <td>0.098604</td>
      <td>0.055096</td>
      <td>0.055096</td>
      <td>0.055096</td>
      <td>0.055096</td>
      <td>0.055096</td>
    </tr>
    <tr>
      <th>99</th>
      <td>0.055096</td>
      <td>0.055096</td>
      <td>0.003025</td>
      <td>0.055096</td>
      <td>0.055096</td>
      <td>0.055096</td>
      <td>0.055096</td>
      <td>0.055096</td>
      <td>0.055096</td>
      <td>0.055096</td>
      <td>-0.309072</td>
    </tr>
  </tbody>
</table>
<p>100 rows × 11 columns</p>
</div>



Again the encoded columns vary as a function of both the cross-fold and the input, so even the constant column varies after the re-encoding.

However, due to the proper use of a randomized cross-method these columns will still appear to be useless noise.  Their correlation with the output was not substantially elevated during the encoding. Conversely useful columns, if there were any, would remain useful in the re-encoding.  Below, we show that the re-encoded noise variables remain uncorrelated with the explanatory variable when we fit a linear model.


```python
proper_fit_model = statsmodels.api.OLS(
    y_example, 
    statsmodels.api.add_constant(
        cross_frame.values, 
        has_constant='add'))
proper_fit_result = proper_fit_model.fit()

r2 = sklearn.metrics.r2_score(
    y_true=y_example, 
    y_pred=proper_fit_result.predict(
        statsmodels.api.add_constant(
            cross_frame.values,
            has_constant='add')))
assert r2 < 0.2

proper_fit_result.summary()
```




<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>            <td>y</td>        <th>  R-squared:         </th> <td>   0.137</td>
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th> <td>   0.029</td>
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th> <td>   1.265</td>
</tr>
<tr>
  <th>Date:</th>             <td>Tue, 10 Mar 2020</td> <th>  Prob (F-statistic):</th>  <td> 0.258</td> 
</tr>
<tr>
  <th>Time:</th>                 <td>18:01:35</td>     <th>  Log-Likelihood:    </th> <td> -141.53</td>
</tr>
<tr>
  <th>No. Observations:</th>      <td>   100</td>      <th>  AIC:               </th> <td>   307.1</td>
</tr>
<tr>
  <th>Df Residuals:</th>          <td>    88</td>      <th>  BIC:               </th> <td>   338.3</td>
</tr>
<tr>
  <th>Df Model:</th>              <td>    11</td>      <th>                     </th>     <td> </td>   
</tr>
<tr>
  <th>Covariance Type:</th>      <td>nonrobust</td>    <th>                     </th>     <td> </td>   
</tr>
</table>
<table class="simpletable">
<tr>
    <td></td>       <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>const</th> <td>    0.1989</td> <td>    0.231</td> <td>    0.861</td> <td> 0.392</td> <td>   -0.260</td> <td>    0.658</td>
</tr>
<tr>
  <th>x1</th>    <td>   -0.4560</td> <td>    2.452</td> <td>   -0.186</td> <td> 0.853</td> <td>   -5.328</td> <td>    4.416</td>
</tr>
<tr>
  <th>x2</th>    <td>   -0.0941</td> <td>    0.389</td> <td>   -0.242</td> <td> 0.809</td> <td>   -0.867</td> <td>    0.679</td>
</tr>
<tr>
  <th>x3</th>    <td>   -0.1700</td> <td>    0.290</td> <td>   -0.587</td> <td> 0.559</td> <td>   -0.746</td> <td>    0.406</td>
</tr>
<tr>
  <th>x4</th>    <td>   -0.4451</td> <td>    0.445</td> <td>   -1.001</td> <td> 0.320</td> <td>   -1.329</td> <td>    0.439</td>
</tr>
<tr>
  <th>x5</th>    <td>   -0.6703</td> <td>    0.382</td> <td>   -1.753</td> <td> 0.083</td> <td>   -1.430</td> <td>    0.090</td>
</tr>
<tr>
  <th>x6</th>    <td>    0.4572</td> <td>    0.273</td> <td>    1.674</td> <td> 0.098</td> <td>   -0.085</td> <td>    1.000</td>
</tr>
<tr>
  <th>x7</th>    <td>   -0.3084</td> <td>    0.405</td> <td>   -0.761</td> <td> 0.449</td> <td>   -1.114</td> <td>    0.497</td>
</tr>
<tr>
  <th>x8</th>    <td>    0.0349</td> <td>    0.275</td> <td>    0.127</td> <td> 0.899</td> <td>   -0.512</td> <td>    0.581</td>
</tr>
<tr>
  <th>x9</th>    <td>   -0.5848</td> <td>    0.431</td> <td>   -1.357</td> <td> 0.178</td> <td>   -1.441</td> <td>    0.272</td>
</tr>
<tr>
  <th>x10</th>   <td>    0.4283</td> <td>    0.294</td> <td>    1.457</td> <td> 0.149</td> <td>   -0.156</td> <td>    1.012</td>
</tr>
<tr>
  <th>x11</th>   <td>    0.3355</td> <td>    0.255</td> <td>    1.316</td> <td> 0.192</td> <td>   -0.171</td> <td>    0.842</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td> 0.851</td> <th>  Durbin-Watson:     </th> <td>   1.968</td>
</tr>
<tr>
  <th>Prob(Omnibus):</th> <td> 0.653</td> <th>  Jarque-Bera (JB):  </th> <td>   0.395</td>
</tr>
<tr>
  <th>Skew:</th>          <td> 0.045</td> <th>  Prob(JB):          </th> <td>   0.821</td>
</tr>
<tr>
  <th>Kurtosis:</th>      <td> 3.294</td> <th>  Cond. No.          </th> <td>    23.9</td>
</tr>
</table><br/><br/>Warnings:<br/>[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.



We can confirm from the summary that:

  * The overall model does not claim to be predictive
  * The individual coefficients appear to have non-significant p-values

This re-encoding failed to mess up the linear regression: uninformative variables correctly appear uninformative.  This appears to be good as we could hope for in this situation.

## Showing The Leak is Still There

However, there is still a data leak!

Code designed to look for the leak can find it. First we can identify from the encoded variables where the cross-folds are: wherever two rows have the same value for a variable, but see different encodings.  We demonstrate recovering the cross-plan here, by building a data leak machine that can recover `y_example` from the re-encoded noise variables, which should not be possible for variables unrelated to the output.

Here, we solve for complement sets: in this case, the complements of the folds of the cross-plan. Specifically, if row `i` is in fold `a`, the complement set for row `i` is the union of folds `b` and `c` (assuming a 3-fold cross-plan).


```python
partition_solution = solve_for_partition(d_example, cross_frame)

partition_solution
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>idx</th>
      <th>complement</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>[1, 3, 4, 5, 7, 10, 12, 14, 15, 16, 17, 19, 20...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>[0, 2, 3, 4, 5, 6, 7, 8, 9, 11, 13, 14, 15, 17...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>[1, 3, 4, 5, 7, 10, 12, 14, 15, 16, 17, 19, 20...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>[0, 1, 2, 6, 8, 9, 10, 11, 12, 13, 16, 18, 20,...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>[0, 1, 2, 6, 8, 9, 10, 11, 12, 13, 16, 18, 20,...</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>95</th>
      <td>95</td>
      <td>[0, 1, 2, 6, 8, 9, 10, 11, 12, 13, 16, 18, 20,...</td>
    </tr>
    <tr>
      <th>96</th>
      <td>96</td>
      <td>[0, 2, 3, 4, 5, 6, 7, 8, 9, 11, 13, 14, 15, 17...</td>
    </tr>
    <tr>
      <th>97</th>
      <td>97</td>
      <td>[0, 1, 2, 6, 8, 9, 10, 11, 12, 13, 16, 18, 20,...</td>
    </tr>
    <tr>
      <th>98</th>
      <td>98</td>
      <td>[1, 3, 4, 5, 7, 10, 12, 14, 15, 16, 17, 19, 20...</td>
    </tr>
    <tr>
      <th>99</th>
      <td>99</td>
      <td>[1, 3, 4, 5, 7, 10, 12, 14, 15, 16, 17, 19, 20...</td>
    </tr>
  </tbody>
</table>
<p>100 rows × 2 columns</p>
</div>



We know that the target encoder computes a conditional smoothed average of the dependent variable. We can use this knowledge to collect relations of the form `cross_frame[i, j] = dot(wts[i, j], y_example) + b`. 

`cross_frame[i, j]` is known. `wts[i, j]` is gotten by reading the [source code for the target encoder](https://github.com/scikit-learn-contrib/categorical-encoding/blob/master/category_encoders/target_encoder.py). We will solve for `y_example` and `b`.
 
We are proving there is a data leak by recovering `y_example` from the original input variable frame `d_example`, and the re-coded data frame `cross_frame`. There are ways to demonstrate the data leak without using `d_example`, but using `d_example` is simpler ot show.

We start with a new function `target_encoder_weight_rule` that is our adaptation of the original target encoder source code.


```python
def target_encoder_weight_rule(
    *, nrow, partition_indexes, value_indexes,
        min_samples_leaf=1, smoothing=1.0):
    if (partition_indexes is None) or (len(partition_indexes) < 1):
        return None
    res = numpy.zeros(nrow)
    prior_w = 1 / len(partition_indexes)
    if (value_indexes is None) or (len(value_indexes) <= 1):
        res[partition_indexes] = prior_w
        return res
    stats_count = len(value_indexes)
    stats_mean_w = 1 / stats_count
    smoove = 1 / (1 + numpy.exp(-(stats_count - min_samples_leaf) / smoothing))
    res[partition_indexes] = prior_w * (1 - smoove)
    res[value_indexes] = res[value_indexes] + stats_mean_w * smoove
    return res
```

Now use the function to capture the block of weights (`relns_x`) and the map from the original training data elements to their cross-frame encodings (`relns_y`).


```python
relns_x, relns_y = collect_relations(
    d_original=d_example,
    d_coded=cross_frame,
    d_partition=partition_solution,
    est_fn=target_encoder_weight_rule)

relns_x
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>9</th>
      <th>...</th>
      <th>90</th>
      <th>91</th>
      <th>92</th>
      <th>93</th>
      <th>94</th>
      <th>95</th>
      <th>96</th>
      <th>97</th>
      <th>98</th>
      <th>99</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.000000</td>
      <td>0.015152</td>
      <td>0.000000</td>
      <td>0.015152</td>
      <td>0.015152</td>
      <td>0.015152</td>
      <td>0.000000</td>
      <td>0.015152</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.015152</td>
      <td>0.000000</td>
      <td>0.015152</td>
      <td>0.015152</td>
      <td>0.015152</td>
      <td>0.015152</td>
      <td>0.015152</td>
      <td>0.015152</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.014925</td>
      <td>0.000000</td>
      <td>0.014925</td>
      <td>0.014925</td>
      <td>0.014925</td>
      <td>0.014925</td>
      <td>0.014925</td>
      <td>0.014925</td>
      <td>0.014925</td>
      <td>0.014925</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.014925</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.014925</td>
      <td>0.014925</td>
      <td>0.000000</td>
      <td>0.014925</td>
      <td>0.014925</td>
      <td>0.014925</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.000000</td>
      <td>0.015152</td>
      <td>0.000000</td>
      <td>0.015152</td>
      <td>0.015152</td>
      <td>0.015152</td>
      <td>0.000000</td>
      <td>0.015152</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.015152</td>
      <td>0.000000</td>
      <td>0.015152</td>
      <td>0.015152</td>
      <td>0.015152</td>
      <td>0.015152</td>
      <td>0.015152</td>
      <td>0.015152</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.014925</td>
      <td>0.014925</td>
      <td>0.014925</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.014925</td>
      <td>0.000000</td>
      <td>0.014925</td>
      <td>0.014925</td>
      <td>...</td>
      <td>0.014925</td>
      <td>0.014925</td>
      <td>0.014925</td>
      <td>0.014925</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.014925</td>
      <td>0.000000</td>
      <td>0.014925</td>
      <td>0.014925</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.014925</td>
      <td>0.014925</td>
      <td>0.014925</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.014925</td>
      <td>0.000000</td>
      <td>0.014925</td>
      <td>0.014925</td>
      <td>...</td>
      <td>0.014925</td>
      <td>0.014925</td>
      <td>0.014925</td>
      <td>0.014925</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.014925</td>
      <td>0.000000</td>
      <td>0.014925</td>
      <td>0.014925</td>
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
      <th>1095</th>
      <td>0.004014</td>
      <td>0.004014</td>
      <td>0.004014</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.004014</td>
      <td>0.000000</td>
      <td>0.004014</td>
      <td>0.004014</td>
      <td>...</td>
      <td>0.004014</td>
      <td>0.004014</td>
      <td>0.004014</td>
      <td>0.004014</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.004014</td>
      <td>0.000000</td>
      <td>0.004014</td>
      <td>0.004014</td>
    </tr>
    <tr>
      <th>1096</th>
      <td>0.000000</td>
      <td>0.015152</td>
      <td>0.000000</td>
      <td>0.015152</td>
      <td>0.015152</td>
      <td>0.015152</td>
      <td>0.000000</td>
      <td>0.015152</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.015152</td>
      <td>0.000000</td>
      <td>0.015152</td>
      <td>0.015152</td>
      <td>0.015152</td>
      <td>0.015152</td>
      <td>0.015152</td>
      <td>0.015152</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>1097</th>
      <td>0.000000</td>
      <td>0.015152</td>
      <td>0.000000</td>
      <td>0.015152</td>
      <td>0.015152</td>
      <td>0.015152</td>
      <td>0.000000</td>
      <td>0.015152</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.015152</td>
      <td>0.000000</td>
      <td>0.015152</td>
      <td>0.015152</td>
      <td>0.015152</td>
      <td>0.015152</td>
      <td>0.015152</td>
      <td>0.015152</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>1098</th>
      <td>0.004014</td>
      <td>0.000000</td>
      <td>0.004014</td>
      <td>0.004014</td>
      <td>0.004014</td>
      <td>0.004014</td>
      <td>0.369543</td>
      <td>0.004014</td>
      <td>0.004014</td>
      <td>0.004014</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.369543</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.004014</td>
      <td>0.004014</td>
      <td>0.000000</td>
      <td>0.004014</td>
      <td>0.004014</td>
      <td>0.004014</td>
    </tr>
    <tr>
      <th>1099</th>
      <td>0.000000</td>
      <td>0.015152</td>
      <td>0.000000</td>
      <td>0.015152</td>
      <td>0.015152</td>
      <td>0.015152</td>
      <td>0.000000</td>
      <td>0.015152</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.015152</td>
      <td>0.000000</td>
      <td>0.015152</td>
      <td>0.015152</td>
      <td>0.015152</td>
      <td>0.015152</td>
      <td>0.015152</td>
      <td>0.015152</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
  </tbody>
</table>
<p>1100 rows × 100 columns</p>
</div>




```python
relns_y
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>code</th>
      <th>i</th>
      <th>j</th>
      <th>level</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.055096</td>
      <td>0</td>
      <td>0</td>
      <td>a</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.053041</td>
      <td>1</td>
      <td>0</td>
      <td>a</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.055096</td>
      <td>2</td>
      <td>0</td>
      <td>a</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.160505</td>
      <td>3</td>
      <td>0</td>
      <td>a</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.160505</td>
      <td>4</td>
      <td>0</td>
      <td>a</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1095</th>
      <td>0.267566</td>
      <td>15</td>
      <td>10</td>
      <td>nl_31</td>
    </tr>
    <tr>
      <th>1096</th>
      <td>0.055096</td>
      <td>81</td>
      <td>10</td>
      <td>nl_46</td>
    </tr>
    <tr>
      <th>1097</th>
      <td>0.055096</td>
      <td>91</td>
      <td>10</td>
      <td>nl_18</td>
    </tr>
    <tr>
      <th>1098</th>
      <td>-0.114564</td>
      <td>43</td>
      <td>10</td>
      <td>nl_18</td>
    </tr>
    <tr>
      <th>1099</th>
      <td>0.055096</td>
      <td>6</td>
      <td>10</td>
      <td>nl_18</td>
    </tr>
  </tbody>
</table>
<p>1100 rows × 4 columns</p>
</div>



In the above, `relns_y$code[k]` is `cross_frame[relns_y$i[k], relns_y$j[k]]`.

We picked up around 1000 linear relations between the 100 `y_example` values (which we are treating as unknowns) and individual entries from the encoded data frame (which we are saving in `relns_y`).  Many of these are going to be redundant, but we have enough of them to solve for `y_example`:

Let's call the matrix `[1 relns_x]` "*A*" (denoting that we added an initial column of ones), and the column vector `[b y_example]`"*y*". Then we expect that in matrix terms, *A* *y* = `relns_y$code`. This means *y* is the solution to linear equations, or a linear regression in our known quantities.


```python
recover_model = sklearn.linear_model.Ridge(
    alpha = 1.0e-3, 
    normalize=True)
recover_model.fit(relns_x, relns_y['code'])
y_ests = recover_model.coef_

y_ests
```




    array([ 0.38286542, -1.00067254,  0.13751294, -1.19657209,  2.00542574,
           -0.05263463, -0.81119794,  0.20031859, -0.2395179 ,  1.8427718 ,
           -0.37208945,  0.01201268,  0.09141795, -2.6040474 , -1.20241303,
           -1.54167653, -0.27971238, -1.09557047,  0.83786987, -0.55641342,
            0.59636352,  0.92991006,  1.00825944,  0.97554531, -0.12696112,
            1.41833807,  0.52306021, -0.15926224, -0.94035348, -0.1096274 ,
           -1.95732204,  0.44977575, -0.22599382,  0.67050963, -0.09243166,
           -0.45687631,  0.5651105 ,  1.26497826,  0.16751519,  1.41478549,
            0.21518778,  0.47819745,  1.83465353, -0.48868329,  0.90191182,
            0.39574586, -0.33710389, -1.5322998 , -0.85735641,  0.60634676,
           -0.2537905 , -0.0442353 ,  0.77219068, -0.7843696 ,  0.69461054,
            1.06297566,  1.03153519,  0.86000347, -0.78408083, -2.23598836,
           -0.96249331, -0.45846078, -0.49024196,  0.20390794, -0.76087746,
           -0.85466415,  0.23548188,  1.70830328,  0.44751128, -2.38399337,
            3.11638852,  0.38140508, -0.2077327 , -1.03261288, -0.62524334,
            1.30351635,  0.08753775,  0.79674417,  0.68742663,  0.4801845 ,
            1.38062493,  0.87712229,  1.29332496,  1.26959284,  0.8321719 ,
            1.10976459,  0.69073894,  0.64109561, -0.46635664,  1.65171781,
            0.55929975,  0.51407236,  0.34689748, -0.0200642 , -0.99622881,
           -0.45757837, -0.21019318, -1.7123978 ,  3.17090358, -2.40283436])



And these recovered estimates are in fact the original values of `y_example`.


```python
r2 = sklearn.metrics.r2_score(y_true=y_example, y_pred=y_ests)
assert r2 > 0.9

r2
```




    0.9993249813376227




```python
plt_frame = pandas.DataFrame({
    'y_example': y_example,
    'y_estimated': y_ests
})
sbn = seaborn.scatterplot(
    'y_estimated', 'y_example', data=plt_frame)
sbn.set_title(
    "actual ys as a function of estimates from pure noise variables")
true_range = [numpy.min(plt_frame['y_example']), 
              numpy.max(plt_frame['y_example'])]
_ = matplotlib.pyplot.plot(
        true_range, true_range,
        color='orange', alpha=0.5)
```


![png](output_40_0.png)


Again: what we have shown is that even when we target encode a set of pure noise variables using cross-methods, we can still recover `y_example` from the encoded variables. This means that the encoded noise variables can still potentially appear informative about `y_example` when used in a downstream model: cross-methods still leak information about the output variable.

Does this mean we shouldn't use cross-methods? Not necessarily. We'll discuss this more, below.

## Correct Cross Methods, The General Case

Before we get to the question of whether we should use cross-methods at all, let's look at the general case, where the data has both noise variables and signal carrying variables. For this example, we'll use 100 rows of ten noise variables and five signal-carrying variables, all of fifty levels each. We'll throw a constant column in there, too.


```python
d_example_s, y_example_s = mk_data(
    nrow=100,
    n_noise_var=10,
    n_noise_level=50,
    n_signal_var=5)
```


```python
d_example_s
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>const_col</th>
      <th>noise_0</th>
      <th>noise_1</th>
      <th>noise_2</th>
      <th>noise_3</th>
      <th>noise_4</th>
      <th>noise_5</th>
      <th>noise_6</th>
      <th>noise_7</th>
      <th>noise_8</th>
      <th>noise_9</th>
      <th>signal_0</th>
      <th>signal_1</th>
      <th>signal_2</th>
      <th>signal_3</th>
      <th>signal_4</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>a</td>
      <td>nl_46</td>
      <td>nl_41</td>
      <td>nl_21</td>
      <td>nl_49</td>
      <td>nl_16</td>
      <td>nl_17</td>
      <td>nl_42</td>
      <td>nl_16</td>
      <td>nl_34</td>
      <td>nl_9</td>
      <td>a</td>
      <td>b</td>
      <td>a</td>
      <td>b</td>
      <td>a</td>
    </tr>
    <tr>
      <th>1</th>
      <td>a</td>
      <td>nl_16</td>
      <td>nl_27</td>
      <td>nl_0</td>
      <td>nl_5</td>
      <td>nl_41</td>
      <td>nl_5</td>
      <td>nl_40</td>
      <td>nl_23</td>
      <td>nl_27</td>
      <td>nl_42</td>
      <td>a</td>
      <td>a</td>
      <td>a</td>
      <td>b</td>
      <td>b</td>
    </tr>
    <tr>
      <th>2</th>
      <td>a</td>
      <td>nl_9</td>
      <td>nl_34</td>
      <td>nl_8</td>
      <td>nl_35</td>
      <td>nl_0</td>
      <td>nl_24</td>
      <td>nl_46</td>
      <td>nl_35</td>
      <td>nl_28</td>
      <td>nl_25</td>
      <td>a</td>
      <td>a</td>
      <td>b</td>
      <td>a</td>
      <td>a</td>
    </tr>
    <tr>
      <th>3</th>
      <td>a</td>
      <td>nl_40</td>
      <td>nl_31</td>
      <td>nl_25</td>
      <td>nl_35</td>
      <td>nl_37</td>
      <td>nl_19</td>
      <td>nl_20</td>
      <td>nl_46</td>
      <td>nl_38</td>
      <td>nl_9</td>
      <td>b</td>
      <td>a</td>
      <td>a</td>
      <td>a</td>
      <td>b</td>
    </tr>
    <tr>
      <th>4</th>
      <td>a</td>
      <td>nl_16</td>
      <td>nl_35</td>
      <td>nl_45</td>
      <td>nl_28</td>
      <td>nl_27</td>
      <td>nl_44</td>
      <td>nl_1</td>
      <td>nl_12</td>
      <td>nl_18</td>
      <td>nl_31</td>
      <td>b</td>
      <td>b</td>
      <td>a</td>
      <td>a</td>
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
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>95</th>
      <td>a</td>
      <td>nl_24</td>
      <td>nl_2</td>
      <td>nl_4</td>
      <td>nl_12</td>
      <td>nl_44</td>
      <td>nl_8</td>
      <td>nl_0</td>
      <td>nl_7</td>
      <td>nl_48</td>
      <td>nl_47</td>
      <td>b</td>
      <td>b</td>
      <td>b</td>
      <td>a</td>
      <td>a</td>
    </tr>
    <tr>
      <th>96</th>
      <td>a</td>
      <td>nl_9</td>
      <td>nl_10</td>
      <td>nl_17</td>
      <td>nl_39</td>
      <td>nl_2</td>
      <td>nl_31</td>
      <td>nl_4</td>
      <td>nl_24</td>
      <td>nl_12</td>
      <td>nl_6</td>
      <td>b</td>
      <td>a</td>
      <td>a</td>
      <td>a</td>
      <td>a</td>
    </tr>
    <tr>
      <th>97</th>
      <td>a</td>
      <td>nl_9</td>
      <td>nl_26</td>
      <td>nl_3</td>
      <td>nl_44</td>
      <td>nl_7</td>
      <td>nl_32</td>
      <td>nl_42</td>
      <td>nl_43</td>
      <td>nl_30</td>
      <td>nl_24</td>
      <td>a</td>
      <td>b</td>
      <td>a</td>
      <td>b</td>
      <td>a</td>
    </tr>
    <tr>
      <th>98</th>
      <td>a</td>
      <td>nl_12</td>
      <td>nl_1</td>
      <td>nl_16</td>
      <td>nl_18</td>
      <td>nl_26</td>
      <td>nl_28</td>
      <td>nl_20</td>
      <td>nl_10</td>
      <td>nl_39</td>
      <td>nl_7</td>
      <td>b</td>
      <td>b</td>
      <td>b</td>
      <td>a</td>
      <td>a</td>
    </tr>
    <tr>
      <th>99</th>
      <td>a</td>
      <td>nl_16</td>
      <td>nl_20</td>
      <td>nl_5</td>
      <td>nl_4</td>
      <td>nl_0</td>
      <td>nl_28</td>
      <td>nl_34</td>
      <td>nl_47</td>
      <td>nl_6</td>
      <td>nl_31</td>
      <td>a</td>
      <td>b</td>
      <td>b</td>
      <td>a</td>
      <td>b</td>
    </tr>
  </tbody>
</table>
<p>100 rows × 16 columns</p>
</div>




```python
y_example_s
```




    0     0.092855
    1     0.702124
    2     0.624716
    3    -0.057869
    4     2.984870
            ...   
    95   -0.903519
    96    2.625148
    97    1.778676
    98   -1.845954
    99   -1.480355
    Length: 100, dtype: float64



This time we will use `vtreat` to impact code all the variables with cross-methods, because we want to look at some extra information. Instructions for using `vtreat` for regression can be found [here](https://github.com/WinVector/pyvtreat/blob/master/Examples/Regression/Regression.md).


```python
vtreat_coder = vtreat.NumericOutcomeTreatment(
    outcome_name = None,
    params = vtreat.vtreat_parameters({
        'coders': {'impact_code'},
        'filter_to_recommended': False
    }))
vtreat_cross_frame = vtreat_coder.fit_transform(d_example_s, y_example_s)

# the frame of cross-validated encoded variables
vtreat_cross_frame
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>noise_0_impact_code</th>
      <th>noise_1_impact_code</th>
      <th>noise_2_impact_code</th>
      <th>noise_3_impact_code</th>
      <th>noise_4_impact_code</th>
      <th>noise_5_impact_code</th>
      <th>noise_6_impact_code</th>
      <th>noise_7_impact_code</th>
      <th>noise_8_impact_code</th>
      <th>noise_9_impact_code</th>
      <th>signal_0_impact_code</th>
      <th>signal_1_impact_code</th>
      <th>signal_2_impact_code</th>
      <th>signal_3_impact_code</th>
      <th>signal_4_impact_code</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-2.775558e-17</td>
      <td>-0.082937</td>
      <td>0.000000</td>
      <td>2.775558e-17</td>
      <td>8.957836e-01</td>
      <td>0.000000</td>
      <td>0.000000e+00</td>
      <td>-2.860668e+00</td>
      <td>1.227627e-01</td>
      <td>0.553686</td>
      <td>0.734911</td>
      <td>-1.195284</td>
      <td>0.518659</td>
      <td>-1.418944</td>
      <td>0.609915</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-2.126731e-01</td>
      <td>-0.479059</td>
      <td>-0.432493</td>
      <td>0.000000e+00</td>
      <td>-1.154976e-01</td>
      <td>-0.469978</td>
      <td>6.394952e-03</td>
      <td>2.775558e-17</td>
      <td>0.000000e+00</td>
      <td>0.000000</td>
      <td>0.724223</td>
      <td>1.426430</td>
      <td>0.502723</td>
      <td>-1.277764</td>
      <td>-0.478077</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2.076853e+00</td>
      <td>-1.252405</td>
      <td>-0.050814</td>
      <td>-5.230413e-01</td>
      <td>0.000000e+00</td>
      <td>0.061539</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000</td>
      <td>0.535319</td>
      <td>1.377291</td>
      <td>-0.862579</td>
      <td>1.095629</td>
      <td>0.542649</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.000000e+00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>-5.230413e-01</td>
      <td>0.000000e+00</td>
      <td>0.000000</td>
      <td>7.117627e-02</td>
      <td>0.000000e+00</td>
      <td>-1.431014e+00</td>
      <td>-0.486062</td>
      <td>-0.811250</td>
      <td>1.377291</td>
      <td>0.572315</td>
      <td>1.095629</td>
      <td>-0.511667</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-4.668268e-01</td>
      <td>0.000000</td>
      <td>1.328000</td>
      <td>0.000000e+00</td>
      <td>-1.083165e+00</td>
      <td>0.000000</td>
      <td>-5.551115e-17</td>
      <td>-5.551115e-17</td>
      <td>1.240756e+00</td>
      <td>0.000000</td>
      <td>-0.954283</td>
      <td>-1.351057</td>
      <td>0.401000</td>
      <td>1.220950</td>
      <td>0.691481</td>
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
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>95</th>
      <td>0.000000e+00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>4.031278e-01</td>
      <td>-3.880526e-01</td>
      <td>0.000000</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000</td>
      <td>-0.811250</td>
      <td>-1.233949</td>
      <td>-0.862579</td>
      <td>1.095629</td>
      <td>0.542649</td>
    </tr>
    <tr>
      <th>96</th>
      <td>7.325165e-01</td>
      <td>-0.085311</td>
      <td>0.000000</td>
      <td>2.775558e-17</td>
      <td>2.775558e-17</td>
      <td>1.637958</td>
      <td>-1.032072e+00</td>
      <td>2.775558e-17</td>
      <td>-2.775558e-17</td>
      <td>0.000000</td>
      <td>-1.087247</td>
      <td>1.340127</td>
      <td>0.518659</td>
      <td>1.178257</td>
      <td>0.609915</td>
    </tr>
    <tr>
      <th>97</th>
      <td>7.325165e-01</td>
      <td>0.332794</td>
      <td>-1.521902</td>
      <td>2.775558e-17</td>
      <td>0.000000e+00</td>
      <td>0.000000</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>-2.775558e-17</td>
      <td>-0.009648</td>
      <td>0.734911</td>
      <td>-1.195284</td>
      <td>0.518659</td>
      <td>-1.418944</td>
      <td>0.609915</td>
    </tr>
    <tr>
      <th>98</th>
      <td>0.000000e+00</td>
      <td>0.767260</td>
      <td>0.000000</td>
      <td>0.000000e+00</td>
      <td>1.134287e+00</td>
      <td>-0.238486</td>
      <td>-2.246184e-01</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000</td>
      <td>-1.202360</td>
      <td>-1.102484</td>
      <td>-0.690518</td>
      <td>1.070849</td>
      <td>0.565276</td>
    </tr>
    <tr>
      <th>99</th>
      <td>-4.668268e-01</td>
      <td>0.000000</td>
      <td>0.947170</td>
      <td>0.000000e+00</td>
      <td>-4.546943e-01</td>
      <td>-0.339889</td>
      <td>-5.551115e-17</td>
      <td>0.000000e+00</td>
      <td>-6.753481e-01</td>
      <td>0.000000</td>
      <td>0.708476</td>
      <td>-1.351057</td>
      <td>-0.594712</td>
      <td>1.220950</td>
      <td>-0.630371</td>
    </tr>
  </tbody>
</table>
<p>100 rows × 15 columns</p>
</div>



We have deliberately turned off `vtreat`'s feature pruning to allow the noise columns in, to demonstrate overfitting. `vtreat` itself has out-of-sample significance estimates which allow for reliable feature pruning.


```python
vtreat_cross_frame.columns
```




    Index(['noise_0_impact_code', 'noise_1_impact_code', 'noise_2_impact_code',
           'noise_3_impact_code', 'noise_4_impact_code', 'noise_5_impact_code',
           'noise_6_impact_code', 'noise_7_impact_code', 'noise_8_impact_code',
           'noise_9_impact_code', 'signal_0_impact_code', 'signal_1_impact_code',
           'signal_2_impact_code', 'signal_3_impact_code', 'signal_4_impact_code'],
          dtype='object')



Notice not all columns were coded; in particular, `vtreat` eliminates constant columns.  The columns considered are reported in `vtreat_coder.score_frame_`.


```python
cols_to_show = ['variable', 'orig_variable', 'treatment', 'R2', 
                'significance', 'default_threshold', 'recommended']

vtreat_coder.score_frame_.loc[:, cols_to_show]
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>variable</th>
      <th>orig_variable</th>
      <th>treatment</th>
      <th>R2</th>
      <th>significance</th>
      <th>default_threshold</th>
      <th>recommended</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>noise_0_impact_code</td>
      <td>noise_0</td>
      <td>impact_code</td>
      <td>0.000004</td>
      <td>9.840082e-01</td>
      <td>0.066667</td>
      <td>False</td>
    </tr>
    <tr>
      <th>1</th>
      <td>noise_1_impact_code</td>
      <td>noise_1</td>
      <td>impact_code</td>
      <td>0.014053</td>
      <td>2.401227e-01</td>
      <td>0.066667</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2</th>
      <td>noise_2_impact_code</td>
      <td>noise_2</td>
      <td>impact_code</td>
      <td>0.010295</td>
      <td>3.151458e-01</td>
      <td>0.066667</td>
      <td>False</td>
    </tr>
    <tr>
      <th>3</th>
      <td>noise_3_impact_code</td>
      <td>noise_3</td>
      <td>impact_code</td>
      <td>0.000676</td>
      <td>7.973835e-01</td>
      <td>0.066667</td>
      <td>False</td>
    </tr>
    <tr>
      <th>4</th>
      <td>noise_4_impact_code</td>
      <td>noise_4</td>
      <td>impact_code</td>
      <td>0.061040</td>
      <td>1.320812e-02</td>
      <td>0.066667</td>
      <td>False</td>
    </tr>
    <tr>
      <th>5</th>
      <td>noise_5_impact_code</td>
      <td>noise_5</td>
      <td>impact_code</td>
      <td>0.003886</td>
      <td>5.377935e-01</td>
      <td>0.066667</td>
      <td>False</td>
    </tr>
    <tr>
      <th>6</th>
      <td>noise_6_impact_code</td>
      <td>noise_6</td>
      <td>impact_code</td>
      <td>0.016734</td>
      <td>1.995865e-01</td>
      <td>0.066667</td>
      <td>False</td>
    </tr>
    <tr>
      <th>7</th>
      <td>noise_7_impact_code</td>
      <td>noise_7</td>
      <td>impact_code</td>
      <td>0.003024</td>
      <td>5.868620e-01</td>
      <td>0.066667</td>
      <td>False</td>
    </tr>
    <tr>
      <th>8</th>
      <td>noise_8_impact_code</td>
      <td>noise_8</td>
      <td>impact_code</td>
      <td>0.050362</td>
      <td>2.479017e-02</td>
      <td>0.066667</td>
      <td>True</td>
    </tr>
    <tr>
      <th>9</th>
      <td>noise_9_impact_code</td>
      <td>noise_9</td>
      <td>impact_code</td>
      <td>0.000023</td>
      <td>9.619327e-01</td>
      <td>0.066667</td>
      <td>False</td>
    </tr>
    <tr>
      <th>10</th>
      <td>signal_0_impact_code</td>
      <td>signal_0</td>
      <td>impact_code</td>
      <td>0.111925</td>
      <td>6.688370e-04</td>
      <td>0.066667</td>
      <td>True</td>
    </tr>
    <tr>
      <th>11</th>
      <td>signal_1_impact_code</td>
      <td>signal_1</td>
      <td>impact_code</td>
      <td>0.273179</td>
      <td>2.432494e-08</td>
      <td>0.066667</td>
      <td>True</td>
    </tr>
    <tr>
      <th>12</th>
      <td>signal_2_impact_code</td>
      <td>signal_2</td>
      <td>impact_code</td>
      <td>0.078100</td>
      <td>4.864999e-03</td>
      <td>0.066667</td>
      <td>True</td>
    </tr>
    <tr>
      <th>13</th>
      <td>signal_3_impact_code</td>
      <td>signal_3</td>
      <td>impact_code</td>
      <td>0.263524</td>
      <td>4.722634e-08</td>
      <td>0.066667</td>
      <td>True</td>
    </tr>
    <tr>
      <th>14</th>
      <td>signal_4_impact_code</td>
      <td>signal_4</td>
      <td>impact_code</td>
      <td>0.065905</td>
      <td>9.929392e-03</td>
      <td>0.066667</td>
      <td>True</td>
    </tr>
  </tbody>
</table>
</div>



The above summary shows the R-squared (`R2`) of each variable when considered as a one-variable model for `y_example_s`, as well as the estimated significance (`significance`) of that fit. The R-squareds and significances are computed in a cross-validated manner, so they are good estimates of future out-of-sample performance.

The `recommended` column marks which variables have significances below the `default_threshold`, which itself is chosen to allow at most one uninformative column through (in expectation). In this case, we see that all the signal variables and only one noise variable are recommended.

From the score frame, we can examine the distribution of cross-validated significances grouped by whether the variable was a signal variable or not.


```python
str_strip = re.compile('_.*$')
sig_frame = pandas.DataFrame({
    'significance': vtreat_coder.score_frame_['significance'],
    'variable_type': [str_strip.sub('', v) for v in 
                      vtreat_coder.score_frame_['variable']]
    })
splt = seaborn.kdeplot(
    sig_frame['significance'][sig_frame['variable_type'] == 'noise'], 
    color="r",
    shade=True,
    label='noise significance')
splt = seaborn.kdeplot(
    sig_frame['significance'][sig_frame['variable_type'] == 'signal'], 
    color="b",
    shade=True,
    label='signal significance')
splt.set(xlim=(0,1))
splt.set_yscale('log')
_ =splt.set_title('distribution of training R2 grouped by variable type\n(log y scale)')
```


![png](output_52_0.png)


It is as we would hope: the signaling variables have p-values concentrated near zero, and the noise variables have significances that are uniformly distributed in the interval `[0, 1]`.

Now we try fitting a linear model using the encoded frame:


```python
good_model = statsmodels.api.OLS(
    y_example_s, 
    statsmodels.api.add_constant(
        vtreat_cross_frame.values, 
        has_constant='add'))
good_fit = good_model.fit()

train_r2 = sklearn.metrics.r2_score(
    y_true=y_example_s, 
    y_pred=good_fit.predict(
        statsmodels.api.add_constant(
            vtreat_cross_frame.values, 
            has_constant='add')))
assert train_r2 > 0.7

good_fit.summary()
```




<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>            <td>y</td>        <th>  R-squared:         </th> <td>   0.840</td>
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th> <td>   0.811</td>
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th> <td>   29.31</td>
</tr>
<tr>
  <th>Date:</th>             <td>Tue, 10 Mar 2020</td> <th>  Prob (F-statistic):</th> <td>4.52e-27</td>
</tr>
<tr>
  <th>Time:</th>                 <td>18:01:40</td>     <th>  Log-Likelihood:    </th> <td> -136.14</td>
</tr>
<tr>
  <th>No. Observations:</th>      <td>   100</td>      <th>  AIC:               </th> <td>   304.3</td>
</tr>
<tr>
  <th>Df Residuals:</th>          <td>    84</td>      <th>  BIC:               </th> <td>   346.0</td>
</tr>
<tr>
  <th>Df Model:</th>              <td>    15</td>      <th>                     </th>     <td> </td>   
</tr>
<tr>
  <th>Covariance Type:</th>      <td>nonrobust</td>    <th>                     </th>     <td> </td>   
</tr>
</table>
<table class="simpletable">
<tr>
    <td></td>       <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>const</th> <td>    0.1934</td> <td>    0.120</td> <td>    1.608</td> <td> 0.112</td> <td>   -0.046</td> <td>    0.433</td>
</tr>
<tr>
  <th>x1</th>    <td>   -0.1831</td> <td>    0.167</td> <td>   -1.096</td> <td> 0.276</td> <td>   -0.515</td> <td>    0.149</td>
</tr>
<tr>
  <th>x2</th>    <td>    0.2513</td> <td>    0.208</td> <td>    1.210</td> <td> 0.230</td> <td>   -0.162</td> <td>    0.664</td>
</tr>
<tr>
  <th>x3</th>    <td>    0.0320</td> <td>    0.159</td> <td>    0.202</td> <td> 0.841</td> <td>   -0.283</td> <td>    0.347</td>
</tr>
<tr>
  <th>x4</th>    <td>    0.1212</td> <td>    0.210</td> <td>    0.577</td> <td> 0.565</td> <td>   -0.296</td> <td>    0.539</td>
</tr>
<tr>
  <th>x5</th>    <td>   -0.1123</td> <td>    0.186</td> <td>   -0.604</td> <td> 0.547</td> <td>   -0.482</td> <td>    0.257</td>
</tr>
<tr>
  <th>x6</th>    <td>   -0.3233</td> <td>    0.165</td> <td>   -1.961</td> <td> 0.053</td> <td>   -0.651</td> <td>    0.005</td>
</tr>
<tr>
  <th>x7</th>    <td>   -0.0417</td> <td>    0.162</td> <td>   -0.258</td> <td> 0.797</td> <td>   -0.363</td> <td>    0.280</td>
</tr>
<tr>
  <th>x8</th>    <td>    0.2390</td> <td>    0.169</td> <td>    1.417</td> <td> 0.160</td> <td>   -0.096</td> <td>    0.574</td>
</tr>
<tr>
  <th>x9</th>    <td>    0.1610</td> <td>    0.149</td> <td>    1.083</td> <td> 0.282</td> <td>   -0.135</td> <td>    0.457</td>
</tr>
<tr>
  <th>x10</th>   <td>   -0.1025</td> <td>    0.198</td> <td>   -0.517</td> <td> 0.606</td> <td>   -0.496</td> <td>    0.292</td>
</tr>
<tr>
  <th>x11</th>   <td>    1.1851</td> <td>    0.135</td> <td>    8.756</td> <td> 0.000</td> <td>    0.916</td> <td>    1.454</td>
</tr>
<tr>
  <th>x12</th>   <td>    0.8837</td> <td>    0.092</td> <td>    9.631</td> <td> 0.000</td> <td>    0.701</td> <td>    1.066</td>
</tr>
<tr>
  <th>x13</th>   <td>    1.4632</td> <td>    0.188</td> <td>    7.780</td> <td> 0.000</td> <td>    1.089</td> <td>    1.837</td>
</tr>
<tr>
  <th>x14</th>   <td>    0.8204</td> <td>    0.094</td> <td>    8.730</td> <td> 0.000</td> <td>    0.634</td> <td>    1.007</td>
</tr>
<tr>
  <th>x15</th>   <td>    1.8506</td> <td>    0.212</td> <td>    8.725</td> <td> 0.000</td> <td>    1.429</td> <td>    2.272</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td> 2.383</td> <th>  Durbin-Watson:     </th> <td>   2.156</td>
</tr>
<tr>
  <th>Prob(Omnibus):</th> <td> 0.304</td> <th>  Jarque-Bera (JB):  </th> <td>   1.621</td>
</tr>
<tr>
  <th>Skew:</th>          <td> 0.057</td> <th>  Prob(JB):          </th> <td>   0.445</td>
</tr>
<tr>
  <th>Kurtosis:</th>      <td> 2.387</td> <th>  Cond. No.          </th> <td>    3.34</td>
</tr>
</table><br/><br/>Warnings:<br/>[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.




```python
sklearn.metrics.r2_score(
    y_true=y_example_s, 
    y_pred=good_fit.predict(
        statsmodels.api.add_constant(
            vtreat_cross_frame.values, 
            has_constant='add')))
```




    0.8396086797764389



The linear model has a reasonable, but not outrageous, R-squared on the training data, and correctly estimates that the signal variables are more significant than the noise variables. 

More importantly, this model performs about the same on fresh identically distributed data. We can show this by generating test sets distributed identically to the training set, evaluating them with the model, and estimating R-squared.


```python
def f():
    d_test_s, y_test_s = mk_data(
        nrow=100,
        n_noise_var=10,
        n_noise_level=50,
        n_signal_var=5)

    vtreat_test_frame = vtreat_coder.transform(d_test_s)

    return sklearn.metrics.r2_score(
                y_true=y_test_s, 
                y_pred=good_fit.predict(
                    statsmodels.api.add_constant(
                        vtreat_test_frame.values, 
                        has_constant='add')))

# the array of R-squared for the repeated tests
test_r2 = numpy.asarray([f() for i in range(100)])
```


```python
assert numpy.mean(test_r2) >= train_r2 - 0.1
```


```python
splot = seaborn.distplot(test_r2)
_ = splot.set_title('distribution of test R2 under repeated draws')
```


![png](output_59_0.png)


We expect that `category_encoders.target_encoder.TargetEncoder` would also handle the above example correctly.  And we could write code to break `vtreat`, just as we wrote code to break `category_encoders.target_encoder.TargetEncoder`. 

The point is `vtreat` is easy to use, and supplies a wide variety of safe and useful variable transforms for predictive modeling and machine learning. 

## So Should We Use Cross-Methods? Or Not?

In this article, we've shown that cross-methods *do* leak information about the training data, so it is not as "safe" in that sense as splitting training data into multiple partitions: one for setting parameters or calculating data transformations, one for training the model, and one for evaluating it. So if you have a large enough data set, partitioning it is probably preferable to cross-methods.
We've also seen that deterministic cross-method schemes like leave-one-out are particularly leaky.

On the other hand, we've also seen that the leak from randomized cross-methods is small enough that linear regression does not seem to see it: the linear models fit to randomized cross-validated encodings above correctly identified noise variables as uninformative most of the time. From experience, we (the authors) have seen that tree ensemble methods like random forest and gradient boosted trees also do not seem too sensitive to the leak ([here's an `xgboost` example](https://github.com/WinVector/pyvtreat/blob/master/Examples/KDD2009Example/KDD2009Example_no_filter.ipynb)).

So what we seem to be able to say is that cross-methods lower the information leak enough that the transformed training data appears safe to use with a reasonable downstream modeling algorithm. This is consistent with the results from superlearning and stacking, which use cross-methods to build "features" corresponding to the individual sub-learners, and then fit a model from these features to learn the overall ensemble model. Just as sometimes it is appropriate to introduce a bit of bias to for a large reduction in variance (bias/variance trade-off), it can be appropriate to pursue a favorable leak/variance trade-off.

It's worth noting that the recommended method to combine the sub-learners in stacking is non-negative linear regression, which is essentially a form of regularized regression. While [regularization is no substitute for cross-methods](http://www.win-vector.com/blog/2019/11/when-cross-validation-is-more-powerful-than-regularization/), it can certainly help reduce the possibility of overfit. As we saw above, linear regression *is* sensitive to the leave-one-out leak, because it can use large coefficients to multiply the leak's magnitude. Regularization would help prevent that, and non-negativity constraints completely eliminate the leak that we demonstrated, as that leak requires negative coefficients. So properly cross-validated encodings are typically "safe" at least when used with regularized regression.

What about tree ensemble methods? Random forest and gradient boosting are higher complexity models, so there is more risk that they might decode the leak. Our speculation is that the averaging inherent in random forest may serve as a "regularization" or smoothing step that helps mitigate this risk; and of course limiting the depth of the trees in a tree ensemble method is also a form of regularization.

### Even hold-out sets can be leaky!

However, re-using a cross-validated set multiple times within the model fitting process, for example with stepwise regression, or using cross-methods for multiple layers of nested models, probably increases the chance that the modeling algorithm will decode the leak. In fact, even hold-out sets leak information when used this way!

The last section of [this article](http://www.win-vector.com/blog/2015/10/a-simpler-explanation-of-differential-privacy/) shows an example of hold-out set leakage during stepwise regression. And [here](http://proceedings.mlr.press/v37/blum15.pdf) is an example of leaderboard hold-out set leakage during Kaggle competitions. The leakage in both these situations occurs because the hold-out set is used multiple times during the model fitting/model tuning/model selection process, and hence leaks information that leads to model overfit.

Model selection is *not* an unbiased procedure, and the bias, however small, can lead to information leakage. For any sort of hyper-parameter tuning or model search, the procedure is biased (though likely of small magnitude) no matter what hold-out procedure we use. Therefore, avoiding cross-validation leakage isn't the only problem.

## Practical Considerations

As we mentioned above, `vtreat` encodes high-cardinality categorical variables by their conditional difference from mean outcome (impact coding), rather than the conditional mean (target coding). Though the proof is out of scope of this article, impact coding has slightly lower variance, and a slightly lower magnitude leak for randomized cross-methods. However, the difference isn't very important in practice.

## Conclusion

We can summarize the takeaways from the experiments that we've shown here:

* If you have enough training data, partitioning it into sets for data transformations, model training, and evaluation may be preferable to cross-methods.
* If partitioning is not an option, cross methods may be good enough for reasonable applications.
* Avoid leave-one-out and other deterministic cross-method schemes.
* When using cross-validated encoded data, prefer regularized methods for the downstream model fitting when possible.



```python

```

