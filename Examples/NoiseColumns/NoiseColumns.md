

```python

```
!pip install /Users/johnmount/Documents/work/pyvtreat/pkg/dist/vtreat-0.1.tar.gz
#!pip install https://github.com/WinVector/pyvtreat/raw/master/pkg/dist/vtreat-0.1.tar.gz

```python
import pandas
import numpy.random
import vtreat
import sklearn.linear_model
import sklearn.metrics
import seaborn
import matplotlib.pyplot
import statsmodels.api
import scipy.stats
import re

```


```python
n_rows = 5000
n_signal_variables = 10
n_noise_variables = 100
n_levels = 500
```


```python
d = pandas.DataFrame({"y":0.01*numpy.random.normal(size = n_rows)})
```


```python
def mk_var_values(n_levels):
    values = {}
    for i in range(n_levels):
        values["level_" + str(i)] = numpy.random.uniform(low=-10, high=10, size=1)[0]
    return values
```


```python
for i in range(n_signal_variables):
    var_name = "var_" + str(i)
    levs = mk_var_values(n_levels)
    keys = [ k for k in levs.keys() ]
    observed = [ keys[i] for i in numpy.random.choice(len(keys), size=n_rows, replace=True)]
    effect = numpy.asarray([ levs[k] for k in observed ])
    d[var_name] = observed
    d["y"] = d["y"] + effect
```


```python
for i in range(n_noise_variables):
    var_name = "noise_" + str(i)
    levs = mk_var_values(n_levels)
    keys = [ k for k in levs.keys() ]
    observed = [ keys[i] for i in numpy.random.choice(len(keys), size=n_rows, replace=True)]
    d[var_name] = observed
```


```python
d.head()
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
      <th>y</th>
      <th>var_0</th>
      <th>var_1</th>
      <th>var_2</th>
      <th>var_3</th>
      <th>var_4</th>
      <th>var_5</th>
      <th>var_6</th>
      <th>var_7</th>
      <th>var_8</th>
      <th>...</th>
      <th>noise_90</th>
      <th>noise_91</th>
      <th>noise_92</th>
      <th>noise_93</th>
      <th>noise_94</th>
      <th>noise_95</th>
      <th>noise_96</th>
      <th>noise_97</th>
      <th>noise_98</th>
      <th>noise_99</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-16.907658</td>
      <td>level_194</td>
      <td>level_21</td>
      <td>level_61</td>
      <td>level_80</td>
      <td>level_139</td>
      <td>level_70</td>
      <td>level_48</td>
      <td>level_27</td>
      <td>level_199</td>
      <td>...</td>
      <td>level_61</td>
      <td>level_45</td>
      <td>level_66</td>
      <td>level_444</td>
      <td>level_299</td>
      <td>level_438</td>
      <td>level_314</td>
      <td>level_442</td>
      <td>level_25</td>
      <td>level_79</td>
    </tr>
    <tr>
      <th>1</th>
      <td>13.173611</td>
      <td>level_426</td>
      <td>level_59</td>
      <td>level_308</td>
      <td>level_187</td>
      <td>level_309</td>
      <td>level_99</td>
      <td>level_398</td>
      <td>level_10</td>
      <td>level_309</td>
      <td>...</td>
      <td>level_467</td>
      <td>level_238</td>
      <td>level_395</td>
      <td>level_214</td>
      <td>level_477</td>
      <td>level_74</td>
      <td>level_65</td>
      <td>level_210</td>
      <td>level_434</td>
      <td>level_229</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-24.634721</td>
      <td>level_259</td>
      <td>level_268</td>
      <td>level_2</td>
      <td>level_66</td>
      <td>level_204</td>
      <td>level_379</td>
      <td>level_440</td>
      <td>level_266</td>
      <td>level_208</td>
      <td>...</td>
      <td>level_465</td>
      <td>level_442</td>
      <td>level_477</td>
      <td>level_83</td>
      <td>level_418</td>
      <td>level_220</td>
      <td>level_317</td>
      <td>level_168</td>
      <td>level_317</td>
      <td>level_330</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-28.889694</td>
      <td>level_4</td>
      <td>level_467</td>
      <td>level_223</td>
      <td>level_185</td>
      <td>level_493</td>
      <td>level_399</td>
      <td>level_436</td>
      <td>level_84</td>
      <td>level_417</td>
      <td>...</td>
      <td>level_153</td>
      <td>level_358</td>
      <td>level_477</td>
      <td>level_10</td>
      <td>level_322</td>
      <td>level_111</td>
      <td>level_242</td>
      <td>level_499</td>
      <td>level_243</td>
      <td>level_275</td>
    </tr>
    <tr>
      <th>4</th>
      <td>11.143388</td>
      <td>level_29</td>
      <td>level_433</td>
      <td>level_89</td>
      <td>level_292</td>
      <td>level_78</td>
      <td>level_119</td>
      <td>level_225</td>
      <td>level_57</td>
      <td>level_119</td>
      <td>...</td>
      <td>level_170</td>
      <td>level_354</td>
      <td>level_16</td>
      <td>level_366</td>
      <td>level_341</td>
      <td>level_95</td>
      <td>level_366</td>
      <td>level_49</td>
      <td>level_470</td>
      <td>level_117</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 111 columns</p>
</div>




```python
is_train = numpy.random.uniform(size=n_rows)<=0.5
```


```python
d_train = d.loc[is_train,:].copy()
d_train.reset_index(inplace=True, drop=True)
y_train = numpy.asarray(d_train["y"])
d_train.drop(["y"], axis=1, inplace=True)
d_test = d.loc[numpy.logical_not(is_train),:].copy()
d_test.reset_index(inplace=True, drop=True)
y_test = numpy.asarray(d_test["y"])
d_test.drop(["y"], axis=1, inplace=True)
```


```python
plan = vtreat.NumericOutcomeTreatment(params=vtreat.vtreat_parameters({'filter_to_recommended':False,
                                                                       'coders':['impact_code']}))
cross_frame = plan.fit_transform(d_train, y_train)
prepared_test = plan.transform(d_test)
naive_train_hierarchical = plan.transform(d_train)
```


```python
p2 = vtreat.NumericOutcomeTreatment(params=vtreat.vtreat_parameters({'filter_to_recommended':False,
                                                                     'coders':['impact_code'],
                                                                     'use_hierarchical_estimate':False}))
p2.fit_transform(d_train, y_train)
naive_train_empirical = p2.transform(d_train)
```


```python
naive_train_empirical.head()
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
      <th>var_0_impact_code</th>
      <th>var_1_impact_code</th>
      <th>var_2_impact_code</th>
      <th>var_3_impact_code</th>
      <th>var_4_impact_code</th>
      <th>var_5_impact_code</th>
      <th>var_6_impact_code</th>
      <th>var_7_impact_code</th>
      <th>var_8_impact_code</th>
      <th>var_9_impact_code</th>
      <th>...</th>
      <th>noise_90_impact_code</th>
      <th>noise_91_impact_code</th>
      <th>noise_92_impact_code</th>
      <th>noise_93_impact_code</th>
      <th>noise_94_impact_code</th>
      <th>noise_95_impact_code</th>
      <th>noise_96_impact_code</th>
      <th>noise_97_impact_code</th>
      <th>noise_98_impact_code</th>
      <th>noise_99_impact_code</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-1.613044</td>
      <td>7.309693</td>
      <td>-17.785767</td>
      <td>1.969784</td>
      <td>-17.954727</td>
      <td>-0.422620</td>
      <td>-10.214963</td>
      <td>-16.316675</td>
      <td>-12.062746</td>
      <td>-6.635621</td>
      <td>...</td>
      <td>-2.450519</td>
      <td>-5.307431</td>
      <td>-16.498058</td>
      <td>-7.187587</td>
      <td>-15.078140</td>
      <td>-15.450775</td>
      <td>9.307387</td>
      <td>-5.057685</td>
      <td>-8.329166</td>
      <td>-1.105957</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-3.345912</td>
      <td>-11.179383</td>
      <td>-9.296097</td>
      <td>-3.085181</td>
      <td>-10.636432</td>
      <td>-12.083500</td>
      <td>-12.189272</td>
      <td>-7.263878</td>
      <td>7.885662</td>
      <td>-5.065825</td>
      <td>...</td>
      <td>-11.529781</td>
      <td>-6.832922</td>
      <td>-14.899030</td>
      <td>-1.822661</td>
      <td>1.508089</td>
      <td>-7.146757</td>
      <td>-3.656139</td>
      <td>-6.747974</td>
      <td>-0.770327</td>
      <td>-8.589438</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1.096256</td>
      <td>-12.897146</td>
      <td>4.730032</td>
      <td>-4.440202</td>
      <td>-15.097173</td>
      <td>-11.504246</td>
      <td>-12.538091</td>
      <td>-0.655624</td>
      <td>-21.253826</td>
      <td>-12.918054</td>
      <td>...</td>
      <td>0.232059</td>
      <td>-4.846785</td>
      <td>-14.899030</td>
      <td>3.691378</td>
      <td>-15.007704</td>
      <td>1.709418</td>
      <td>-18.292191</td>
      <td>3.740072</td>
      <td>-29.605933</td>
      <td>-2.382473</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-6.043885</td>
      <td>4.087619</td>
      <td>-1.826510</td>
      <td>4.862241</td>
      <td>13.001756</td>
      <td>15.588116</td>
      <td>13.205725</td>
      <td>10.677192</td>
      <td>-10.181010</td>
      <td>-10.971738</td>
      <td>...</td>
      <td>1.686216</td>
      <td>-2.996761</td>
      <td>-6.827816</td>
      <td>-7.590614</td>
      <td>-5.550391</td>
      <td>4.442275</td>
      <td>-0.508748</td>
      <td>7.296096</td>
      <td>-1.374595</td>
      <td>-1.105957</td>
    </tr>
    <tr>
      <th>4</th>
      <td>12.592230</td>
      <td>9.351592</td>
      <td>17.908038</td>
      <td>2.314205</td>
      <td>10.352554</td>
      <td>0.822898</td>
      <td>2.348441</td>
      <td>1.894467</td>
      <td>-8.158815</td>
      <td>-2.434019</td>
      <td>...</td>
      <td>8.339193</td>
      <td>4.765558</td>
      <td>4.383108</td>
      <td>9.772782</td>
      <td>19.225731</td>
      <td>9.142069</td>
      <td>14.381890</td>
      <td>6.567419</td>
      <td>-8.329166</td>
      <td>-0.249482</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 110 columns</p>
</div>




```python
naive_train_hierarchical.head()
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
      <th>var_0_impact_code</th>
      <th>var_1_impact_code</th>
      <th>var_2_impact_code</th>
      <th>var_3_impact_code</th>
      <th>var_4_impact_code</th>
      <th>var_5_impact_code</th>
      <th>var_6_impact_code</th>
      <th>var_7_impact_code</th>
      <th>var_8_impact_code</th>
      <th>var_9_impact_code</th>
      <th>...</th>
      <th>noise_90_impact_code</th>
      <th>noise_91_impact_code</th>
      <th>noise_92_impact_code</th>
      <th>noise_93_impact_code</th>
      <th>noise_94_impact_code</th>
      <th>noise_95_impact_code</th>
      <th>noise_96_impact_code</th>
      <th>noise_97_impact_code</th>
      <th>noise_98_impact_code</th>
      <th>noise_99_impact_code</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-1.189601</td>
      <td>3.239000</td>
      <td>-10.838493</td>
      <td>1.201294</td>
      <td>-14.351030</td>
      <td>-0.313430</td>
      <td>-7.045294</td>
      <td>-10.875722</td>
      <td>-9.459114</td>
      <td>-4.171693</td>
      <td>...</td>
      <td>-1.185211</td>
      <td>-3.362592</td>
      <td>-10.064938</td>
      <td>-4.485625</td>
      <td>-9.433327</td>
      <td>-14.284450</td>
      <td>0.536543</td>
      <td>-4.064510</td>
      <td>-3.767950</td>
      <td>-0.714570</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-2.172107</td>
      <td>-9.030694</td>
      <td>-6.358005</td>
      <td>-1.574773</td>
      <td>-8.357936</td>
      <td>-7.614960</td>
      <td>-7.959597</td>
      <td>-5.498445</td>
      <td>5.444707</td>
      <td>-3.834936</td>
      <td>...</td>
      <td>-5.998020</td>
      <td>-2.491869</td>
      <td>-12.681084</td>
      <td>-0.910839</td>
      <td>0.772592</td>
      <td>-4.308604</td>
      <td>-2.195945</td>
      <td>-0.729158</td>
      <td>-0.189000</td>
      <td>-3.163784</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.637642</td>
      <td>-8.293719</td>
      <td>3.224296</td>
      <td>-3.297465</td>
      <td>-8.163491</td>
      <td>-8.223806</td>
      <td>-9.096305</td>
      <td>-0.386405</td>
      <td>-16.039755</td>
      <td>-5.783740</td>
      <td>...</td>
      <td>0.127999</td>
      <td>-1.129770</td>
      <td>-12.681084</td>
      <td>1.998449</td>
      <td>-2.655265</td>
      <td>0.901850</td>
      <td>-9.857703</td>
      <td>1.967284</td>
      <td>-29.360009</td>
      <td>-1.416476</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-3.274597</td>
      <td>3.025480</td>
      <td>-1.451880</td>
      <td>2.872828</td>
      <td>12.325686</td>
      <td>13.472276</td>
      <td>6.355680</td>
      <td>8.019223</td>
      <td>-6.558431</td>
      <td>-7.378532</td>
      <td>...</td>
      <td>0.866719</td>
      <td>-1.471892</td>
      <td>-2.090957</td>
      <td>-3.817411</td>
      <td>-3.029714</td>
      <td>3.699033</td>
      <td>-0.417870</td>
      <td>3.404639</td>
      <td>-0.871065</td>
      <td>-0.714570</td>
    </tr>
    <tr>
      <th>4</th>
      <td>7.388958</td>
      <td>6.270885</td>
      <td>13.937263</td>
      <td>1.246526</td>
      <td>7.238080</td>
      <td>0.648207</td>
      <td>2.019714</td>
      <td>1.235937</td>
      <td>-5.440480</td>
      <td>-1.376015</td>
      <td>...</td>
      <td>1.938650</td>
      <td>2.801780</td>
      <td>2.569122</td>
      <td>4.983974</td>
      <td>16.701618</td>
      <td>6.093368</td>
      <td>6.671650</td>
      <td>4.822364</td>
      <td>-3.767950</td>
      <td>-0.171341</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 110 columns</p>
</div>




```python
cross_frame.head()
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
      <th>var_0_impact_code</th>
      <th>var_1_impact_code</th>
      <th>var_2_impact_code</th>
      <th>var_3_impact_code</th>
      <th>var_4_impact_code</th>
      <th>var_5_impact_code</th>
      <th>var_6_impact_code</th>
      <th>var_7_impact_code</th>
      <th>var_8_impact_code</th>
      <th>var_9_impact_code</th>
      <th>...</th>
      <th>noise_90_impact_code</th>
      <th>noise_91_impact_code</th>
      <th>noise_92_impact_code</th>
      <th>noise_93_impact_code</th>
      <th>noise_94_impact_code</th>
      <th>noise_95_impact_code</th>
      <th>noise_96_impact_code</th>
      <th>noise_97_impact_code</th>
      <th>noise_98_impact_code</th>
      <th>noise_99_impact_code</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-2.429800</td>
      <td>9.478144</td>
      <td>-5.724042e+00</td>
      <td>0.160907</td>
      <td>-3.568846e+00</td>
      <td>1.268282</td>
      <td>-4.093980</td>
      <td>-9.170642</td>
      <td>-9.649399e+00</td>
      <td>-4.061774</td>
      <td>...</td>
      <td>0.003507</td>
      <td>-0.171469</td>
      <td>-6.493029</td>
      <td>-1.300497</td>
      <td>-5.337944</td>
      <td>-18.414072</td>
      <td>0.000000</td>
      <td>-1.903406</td>
      <td>-0.492786</td>
      <td>0.101069</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.201904</td>
      <td>-5.770232</td>
      <td>2.220446e-16</td>
      <td>-1.191729</td>
      <td>-8.419630e+00</td>
      <td>-3.349275</td>
      <td>-7.843296</td>
      <td>-2.072195</td>
      <td>8.781943e+00</td>
      <td>-0.630140</td>
      <td>...</td>
      <td>-3.787456</td>
      <td>-0.109888</td>
      <td>-12.169186</td>
      <td>0.135093</td>
      <td>-0.780486</td>
      <td>-0.642486</td>
      <td>0.541746</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>-0.618028</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3.934290</td>
      <td>-5.949222</td>
      <td>9.318881e+00</td>
      <td>1.507254</td>
      <td>-2.115167e+01</td>
      <td>-6.269846</td>
      <td>-7.639893</td>
      <td>2.294588</td>
      <td>-2.220446e-16</td>
      <td>-2.332152</td>
      <td>...</td>
      <td>2.925051</td>
      <td>2.883001</td>
      <td>-12.169186</td>
      <td>7.332923</td>
      <td>-0.657793</td>
      <td>-0.241049</td>
      <td>-4.664650</td>
      <td>7.967789</td>
      <td>0.000000</td>
      <td>1.698408</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-2.719407</td>
      <td>3.283230</td>
      <td>-6.207167e+00</td>
      <td>1.663967</td>
      <td>-2.220446e-16</td>
      <td>19.386210</td>
      <td>2.871416</td>
      <td>6.496570</td>
      <td>-1.156488e+01</td>
      <td>-9.495806</td>
      <td>...</td>
      <td>-0.316220</td>
      <td>-7.198142</td>
      <td>-3.723235</td>
      <td>-9.582345</td>
      <td>-9.724437</td>
      <td>1.316487</td>
      <td>-2.182906</td>
      <td>2.569129</td>
      <td>-2.448430</td>
      <td>-2.279102</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2.328340</td>
      <td>2.969105</td>
      <td>2.220446e-16</td>
      <td>-0.305870</td>
      <td>1.746467e+00</td>
      <td>-0.847380</td>
      <td>3.515272</td>
      <td>0.791242</td>
      <td>-7.039588e+00</td>
      <td>-19.879679</td>
      <td>...</td>
      <td>0.000000</td>
      <td>5.394772</td>
      <td>2.565390</td>
      <td>17.577781</td>
      <td>0.000000</td>
      <td>1.120931</td>
      <td>4.030480</td>
      <td>4.283975</td>
      <td>-19.284753</td>
      <td>-2.563977</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 110 columns</p>
</div>




```python

```


```python
all_vars = [vi for vi in plan.score_frame_["variable"]]
corr_frame = pandas.DataFrame({"variable":[vi for vi in all_vars if re.match(".*_impact_.*", vi)]})
corr_frame["naive_train_empirical_correlation"] = [ 
    scipy.stats.pearsonr(naive_train_empirical[vi], y_train)[0] for vi in corr_frame["variable"]]
corr_frame["naive_train_hierarchical_correlation"] = [ 
    scipy.stats.pearsonr(naive_train_hierarchical[vi], y_train)[0] for vi in corr_frame["variable"]]
corr_frame["cross_frame_correlation"] = [ 
    scipy.stats.pearsonr(cross_frame[vi], y_train)[0] for vi in corr_frame["variable"]]
corr_frame["test_correlation"] = [ 
    scipy.stats.pearsonr(prepared_test[vi], y_test)[0] for vi in corr_frame["variable"]]
corr_frame["is_noise"] = [re.match("^noise_.*", vi) is not None for vi in corr_frame["variable"]]
corr_frame

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
      <th>naive_train_empirical_correlation</th>
      <th>naive_train_hierarchical_correlation</th>
      <th>cross_frame_correlation</th>
      <th>test_correlation</th>
      <th>is_noise</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>var_0_impact_code</td>
      <td>0.504259</td>
      <td>0.482239</td>
      <td>0.154502</td>
      <td>0.169229</td>
      <td>False</td>
    </tr>
    <tr>
      <th>1</th>
      <td>var_1_impact_code</td>
      <td>0.538102</td>
      <td>0.518626</td>
      <td>0.160459</td>
      <td>0.162919</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2</th>
      <td>var_2_impact_code</td>
      <td>0.518221</td>
      <td>0.488762</td>
      <td>0.153543</td>
      <td>0.206053</td>
      <td>False</td>
    </tr>
    <tr>
      <th>3</th>
      <td>var_3_impact_code</td>
      <td>0.518402</td>
      <td>0.488867</td>
      <td>0.126099</td>
      <td>0.203749</td>
      <td>False</td>
    </tr>
    <tr>
      <th>4</th>
      <td>var_4_impact_code</td>
      <td>0.526746</td>
      <td>0.503099</td>
      <td>0.159614</td>
      <td>0.132191</td>
      <td>False</td>
    </tr>
    <tr>
      <th>5</th>
      <td>var_5_impact_code</td>
      <td>0.548220</td>
      <td>0.519622</td>
      <td>0.194347</td>
      <td>0.198982</td>
      <td>False</td>
    </tr>
    <tr>
      <th>6</th>
      <td>var_6_impact_code</td>
      <td>0.541297</td>
      <td>0.518205</td>
      <td>0.174802</td>
      <td>0.192643</td>
      <td>False</td>
    </tr>
    <tr>
      <th>7</th>
      <td>var_7_impact_code</td>
      <td>0.561650</td>
      <td>0.539927</td>
      <td>0.194140</td>
      <td>0.199807</td>
      <td>False</td>
    </tr>
    <tr>
      <th>8</th>
      <td>var_8_impact_code</td>
      <td>0.514836</td>
      <td>0.492104</td>
      <td>0.151708</td>
      <td>0.148226</td>
      <td>False</td>
    </tr>
    <tr>
      <th>9</th>
      <td>var_9_impact_code</td>
      <td>0.519385</td>
      <td>0.489752</td>
      <td>0.119879</td>
      <td>0.181701</td>
      <td>False</td>
    </tr>
    <tr>
      <th>10</th>
      <td>noise_0_impact_code</td>
      <td>0.443031</td>
      <td>0.414039</td>
      <td>-0.003036</td>
      <td>0.013851</td>
      <td>True</td>
    </tr>
    <tr>
      <th>11</th>
      <td>noise_1_impact_code</td>
      <td>0.464352</td>
      <td>0.434771</td>
      <td>0.016678</td>
      <td>-0.004495</td>
      <td>True</td>
    </tr>
    <tr>
      <th>12</th>
      <td>noise_2_impact_code</td>
      <td>0.444239</td>
      <td>0.409691</td>
      <td>0.013614</td>
      <td>-0.006119</td>
      <td>True</td>
    </tr>
    <tr>
      <th>13</th>
      <td>noise_3_impact_code</td>
      <td>0.434637</td>
      <td>0.400306</td>
      <td>-0.018197</td>
      <td>0.025597</td>
      <td>True</td>
    </tr>
    <tr>
      <th>14</th>
      <td>noise_4_impact_code</td>
      <td>0.426227</td>
      <td>0.395953</td>
      <td>-0.026847</td>
      <td>-0.012815</td>
      <td>True</td>
    </tr>
    <tr>
      <th>15</th>
      <td>noise_5_impact_code</td>
      <td>0.452084</td>
      <td>0.424438</td>
      <td>0.025477</td>
      <td>0.010695</td>
      <td>True</td>
    </tr>
    <tr>
      <th>16</th>
      <td>noise_6_impact_code</td>
      <td>0.432697</td>
      <td>0.400305</td>
      <td>-0.016805</td>
      <td>-0.000264</td>
      <td>True</td>
    </tr>
    <tr>
      <th>17</th>
      <td>noise_7_impact_code</td>
      <td>0.427889</td>
      <td>0.393411</td>
      <td>-0.031050</td>
      <td>-0.023639</td>
      <td>True</td>
    </tr>
    <tr>
      <th>18</th>
      <td>noise_8_impact_code</td>
      <td>0.450677</td>
      <td>0.422882</td>
      <td>0.027459</td>
      <td>0.014889</td>
      <td>True</td>
    </tr>
    <tr>
      <th>19</th>
      <td>noise_9_impact_code</td>
      <td>0.434278</td>
      <td>0.403265</td>
      <td>-0.028636</td>
      <td>-0.036770</td>
      <td>True</td>
    </tr>
    <tr>
      <th>20</th>
      <td>noise_10_impact_code</td>
      <td>0.439710</td>
      <td>0.406303</td>
      <td>-0.005705</td>
      <td>-0.007690</td>
      <td>True</td>
    </tr>
    <tr>
      <th>21</th>
      <td>noise_11_impact_code</td>
      <td>0.443208</td>
      <td>0.409738</td>
      <td>-0.012466</td>
      <td>0.000791</td>
      <td>True</td>
    </tr>
    <tr>
      <th>22</th>
      <td>noise_12_impact_code</td>
      <td>0.414437</td>
      <td>0.377526</td>
      <td>-0.067546</td>
      <td>0.006390</td>
      <td>True</td>
    </tr>
    <tr>
      <th>23</th>
      <td>noise_13_impact_code</td>
      <td>0.438544</td>
      <td>0.407571</td>
      <td>-0.018791</td>
      <td>0.030525</td>
      <td>True</td>
    </tr>
    <tr>
      <th>24</th>
      <td>noise_14_impact_code</td>
      <td>0.440079</td>
      <td>0.407758</td>
      <td>-0.012259</td>
      <td>-0.012128</td>
      <td>True</td>
    </tr>
    <tr>
      <th>25</th>
      <td>noise_15_impact_code</td>
      <td>0.429202</td>
      <td>0.392157</td>
      <td>-0.017894</td>
      <td>0.020234</td>
      <td>True</td>
    </tr>
    <tr>
      <th>26</th>
      <td>noise_16_impact_code</td>
      <td>0.427669</td>
      <td>0.396530</td>
      <td>-0.027919</td>
      <td>-0.024763</td>
      <td>True</td>
    </tr>
    <tr>
      <th>27</th>
      <td>noise_17_impact_code</td>
      <td>0.426906</td>
      <td>0.395428</td>
      <td>-0.017716</td>
      <td>0.029878</td>
      <td>True</td>
    </tr>
    <tr>
      <th>28</th>
      <td>noise_18_impact_code</td>
      <td>0.451607</td>
      <td>0.419544</td>
      <td>0.042701</td>
      <td>0.025945</td>
      <td>True</td>
    </tr>
    <tr>
      <th>29</th>
      <td>noise_19_impact_code</td>
      <td>0.446308</td>
      <td>0.410316</td>
      <td>-0.025135</td>
      <td>0.005952</td>
      <td>True</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>80</th>
      <td>noise_70_impact_code</td>
      <td>0.429975</td>
      <td>0.398612</td>
      <td>-0.012802</td>
      <td>0.012836</td>
      <td>True</td>
    </tr>
    <tr>
      <th>81</th>
      <td>noise_71_impact_code</td>
      <td>0.442892</td>
      <td>0.407122</td>
      <td>-0.002996</td>
      <td>-0.009023</td>
      <td>True</td>
    </tr>
    <tr>
      <th>82</th>
      <td>noise_72_impact_code</td>
      <td>0.431801</td>
      <td>0.398520</td>
      <td>-0.017823</td>
      <td>0.001748</td>
      <td>True</td>
    </tr>
    <tr>
      <th>83</th>
      <td>noise_73_impact_code</td>
      <td>0.443764</td>
      <td>0.414423</td>
      <td>0.006232</td>
      <td>-0.018436</td>
      <td>True</td>
    </tr>
    <tr>
      <th>84</th>
      <td>noise_74_impact_code</td>
      <td>0.437725</td>
      <td>0.407911</td>
      <td>0.028112</td>
      <td>0.046364</td>
      <td>True</td>
    </tr>
    <tr>
      <th>85</th>
      <td>noise_75_impact_code</td>
      <td>0.426404</td>
      <td>0.392840</td>
      <td>-0.007590</td>
      <td>-0.006132</td>
      <td>True</td>
    </tr>
    <tr>
      <th>86</th>
      <td>noise_76_impact_code</td>
      <td>0.442780</td>
      <td>0.410784</td>
      <td>0.022923</td>
      <td>0.018810</td>
      <td>True</td>
    </tr>
    <tr>
      <th>87</th>
      <td>noise_77_impact_code</td>
      <td>0.455887</td>
      <td>0.428344</td>
      <td>0.021716</td>
      <td>-0.018354</td>
      <td>True</td>
    </tr>
    <tr>
      <th>88</th>
      <td>noise_78_impact_code</td>
      <td>0.429144</td>
      <td>0.398206</td>
      <td>0.003492</td>
      <td>-0.005884</td>
      <td>True</td>
    </tr>
    <tr>
      <th>89</th>
      <td>noise_79_impact_code</td>
      <td>0.454408</td>
      <td>0.426773</td>
      <td>0.016167</td>
      <td>-0.004693</td>
      <td>True</td>
    </tr>
    <tr>
      <th>90</th>
      <td>noise_80_impact_code</td>
      <td>0.424959</td>
      <td>0.390585</td>
      <td>-0.018181</td>
      <td>-0.025978</td>
      <td>True</td>
    </tr>
    <tr>
      <th>91</th>
      <td>noise_81_impact_code</td>
      <td>0.466269</td>
      <td>0.436475</td>
      <td>0.028275</td>
      <td>-0.033592</td>
      <td>True</td>
    </tr>
    <tr>
      <th>92</th>
      <td>noise_82_impact_code</td>
      <td>0.440117</td>
      <td>0.409350</td>
      <td>-0.010639</td>
      <td>-0.032945</td>
      <td>True</td>
    </tr>
    <tr>
      <th>93</th>
      <td>noise_83_impact_code</td>
      <td>0.443430</td>
      <td>0.408802</td>
      <td>0.035951</td>
      <td>-0.002458</td>
      <td>True</td>
    </tr>
    <tr>
      <th>94</th>
      <td>noise_84_impact_code</td>
      <td>0.435263</td>
      <td>0.405229</td>
      <td>0.017726</td>
      <td>0.006820</td>
      <td>True</td>
    </tr>
    <tr>
      <th>95</th>
      <td>noise_85_impact_code</td>
      <td>0.441118</td>
      <td>0.408062</td>
      <td>-0.021455</td>
      <td>0.012142</td>
      <td>True</td>
    </tr>
    <tr>
      <th>96</th>
      <td>noise_86_impact_code</td>
      <td>0.437059</td>
      <td>0.405814</td>
      <td>-0.003858</td>
      <td>-0.004620</td>
      <td>True</td>
    </tr>
    <tr>
      <th>97</th>
      <td>noise_87_impact_code</td>
      <td>0.464369</td>
      <td>0.432352</td>
      <td>0.036833</td>
      <td>-0.025537</td>
      <td>True</td>
    </tr>
    <tr>
      <th>98</th>
      <td>noise_88_impact_code</td>
      <td>0.457550</td>
      <td>0.423249</td>
      <td>0.027249</td>
      <td>-0.023764</td>
      <td>True</td>
    </tr>
    <tr>
      <th>99</th>
      <td>noise_89_impact_code</td>
      <td>0.432526</td>
      <td>0.396910</td>
      <td>-0.026990</td>
      <td>0.048902</td>
      <td>True</td>
    </tr>
    <tr>
      <th>100</th>
      <td>noise_90_impact_code</td>
      <td>0.423632</td>
      <td>0.387705</td>
      <td>-0.046140</td>
      <td>-0.006964</td>
      <td>True</td>
    </tr>
    <tr>
      <th>101</th>
      <td>noise_91_impact_code</td>
      <td>0.456524</td>
      <td>0.424754</td>
      <td>0.018694</td>
      <td>0.009385</td>
      <td>True</td>
    </tr>
    <tr>
      <th>102</th>
      <td>noise_92_impact_code</td>
      <td>0.443327</td>
      <td>0.411846</td>
      <td>-0.028164</td>
      <td>-0.005633</td>
      <td>True</td>
    </tr>
    <tr>
      <th>103</th>
      <td>noise_93_impact_code</td>
      <td>0.478243</td>
      <td>0.448259</td>
      <td>0.077066</td>
      <td>-0.028764</td>
      <td>True</td>
    </tr>
    <tr>
      <th>104</th>
      <td>noise_94_impact_code</td>
      <td>0.437055</td>
      <td>0.408547</td>
      <td>-0.011759</td>
      <td>-0.019852</td>
      <td>True</td>
    </tr>
    <tr>
      <th>105</th>
      <td>noise_95_impact_code</td>
      <td>0.461499</td>
      <td>0.433211</td>
      <td>0.014670</td>
      <td>0.035832</td>
      <td>True</td>
    </tr>
    <tr>
      <th>106</th>
      <td>noise_96_impact_code</td>
      <td>0.462444</td>
      <td>0.430473</td>
      <td>0.010972</td>
      <td>0.014855</td>
      <td>True</td>
    </tr>
    <tr>
      <th>107</th>
      <td>noise_97_impact_code</td>
      <td>0.459866</td>
      <td>0.430345</td>
      <td>0.042655</td>
      <td>0.000810</td>
      <td>True</td>
    </tr>
    <tr>
      <th>108</th>
      <td>noise_98_impact_code</td>
      <td>0.450753</td>
      <td>0.412405</td>
      <td>0.016810</td>
      <td>-0.003017</td>
      <td>True</td>
    </tr>
    <tr>
      <th>109</th>
      <td>noise_99_impact_code</td>
      <td>0.450513</td>
      <td>0.413606</td>
      <td>0.002333</td>
      <td>0.001708</td>
      <td>True</td>
    </tr>
  </tbody>
</table>
<p>110 rows × 6 columns</p>
</div>




```python
print(scipy.stats.pearsonr(corr_frame['naive_train_empirical_correlation'], corr_frame['test_correlation']))
seaborn.scatterplot(x = "naive_train_empirical_correlation", y = "test_correlation", data = corr_frame, hue = "is_noise")
matplotlib.pyplot.plot([-1, 1], [-1, 1], color="red")
matplotlib.pyplot.xlim(-.2,1)
matplotlib.pyplot.ylim(-.2,1)
matplotlib.pyplot.gca().set_aspect('equal', adjustable='box')
```

    (0.8367044120276735, 5.232582447486599e-30)



![png](output_18_1.png)



```python
print(scipy.stats.pearsonr(corr_frame['naive_train_hierarchical_correlation'], corr_frame['test_correlation']))
seaborn.scatterplot(x = "naive_train_hierarchical_correlation", y = "test_correlation", data = corr_frame, hue = "is_noise")
matplotlib.pyplot.plot([-1, 1], [-1, 1], color="red")
matplotlib.pyplot.xlim(-.2,1)
matplotlib.pyplot.ylim(-.2,1)
matplotlib.pyplot.gca().set_aspect('equal', adjustable='box')
```

    (0.8344368404075787, 1.0335060354554691e-29)



![png](output_19_1.png)



```python
print(scipy.stats.pearsonr(corr_frame['cross_frame_correlation'], corr_frame['test_correlation']))
seaborn.scatterplot(x = "cross_frame_correlation", y = "test_correlation", data = corr_frame,  hue = "is_noise")
matplotlib.pyplot.plot([-1, 1], [-1, 1], color="red")
matplotlib.pyplot.xlim(-.2,1)
matplotlib.pyplot.ylim(-.2,1)
matplotlib.pyplot.gca().set_aspect('equal', adjustable='box')
```

    (0.8324368768399079, 1.8678518361774684e-29)



![png](output_20_1.png)



```python
plan.score_frame_.tail()
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
      <th>treatment</th>
      <th>y_aware</th>
      <th>has_range</th>
      <th>PearsonR</th>
      <th>significance</th>
      <th>vcount</th>
      <th>recommended</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>105</th>
      <td>noise_95_impact_code</td>
      <td>impact_code</td>
      <td>True</td>
      <td>True</td>
      <td>0.014670</td>
      <td>0.460701</td>
      <td>110.0</td>
      <td>False</td>
    </tr>
    <tr>
      <th>106</th>
      <td>noise_96_impact_code</td>
      <td>impact_code</td>
      <td>True</td>
      <td>True</td>
      <td>0.010972</td>
      <td>0.581136</td>
      <td>110.0</td>
      <td>False</td>
    </tr>
    <tr>
      <th>107</th>
      <td>noise_97_impact_code</td>
      <td>impact_code</td>
      <td>True</td>
      <td>True</td>
      <td>0.042655</td>
      <td>0.031887</td>
      <td>110.0</td>
      <td>False</td>
    </tr>
    <tr>
      <th>108</th>
      <td>noise_98_impact_code</td>
      <td>impact_code</td>
      <td>True</td>
      <td>True</td>
      <td>0.016810</td>
      <td>0.397925</td>
      <td>110.0</td>
      <td>False</td>
    </tr>
    <tr>
      <th>109</th>
      <td>noise_99_impact_code</td>
      <td>impact_code</td>
      <td>True</td>
      <td>True</td>
      <td>0.002333</td>
      <td>0.906629</td>
      <td>110.0</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
</div>




```python
recommended_vars = [vi for vi in plan.score_frame_["variable"][plan.score_frame_["recommended"]]]
recommended_vars
```




    ['var_0_impact_code',
     'var_1_impact_code',
     'var_2_impact_code',
     'var_3_impact_code',
     'var_4_impact_code',
     'var_5_impact_code',
     'var_6_impact_code',
     'var_7_impact_code',
     'var_8_impact_code',
     'var_9_impact_code',
     'noise_47_impact_code',
     'noise_93_impact_code']




```python

```


```python
plot_train = pandas.DataFrame({"y":y_train})
plot_test = pandas.DataFrame({"y":y_test})
```


```python
fitter = sklearn.linear_model.LinearRegression(fit_intercept = True)
```


```python
fitter.fit(cross_frame[all_vars], y_train)
plot_train["predict_cross_all_vars"] = fitter.predict(cross_frame[all_vars])
plot_test["predict_cross_all_vars"] = fitter.predict(prepared_test[all_vars])
```


```python
fitter.fit(cross_frame[recommended_vars], y_train)
plot_train["predict_cross_recommended_vars"] = fitter.predict(cross_frame[recommended_vars])
plot_test["predict_cross_recommended_vars"] = fitter.predict(prepared_test[recommended_vars])
```


```python
fitter.fit(naive_train_empirical[all_vars], y_train)
plot_train["predict_naive_empirical_all_vars"] = fitter.predict(naive_train_empirical[all_vars])
plot_test["predict_naive_empirical_all_vars"] = fitter.predict(prepared_test[all_vars])
```


```python
fitter.fit(naive_train_hierarchical[all_vars], y_train)
plot_train["predict_naive_hierarchical_all_vars"] = fitter.predict(naive_train_hierarchical[all_vars])
plot_test["predict_naive_hierarchical_all_vars"] = fitter.predict(prepared_test[all_vars])
```


```python
plot_test.head()
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
      <th>y</th>
      <th>predict_cross_all_vars</th>
      <th>predict_cross_recommended_vars</th>
      <th>predict_naive_empirical_all_vars</th>
      <th>predict_naive_hierarchical_all_vars</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>13.173611</td>
      <td>0.998098</td>
      <td>-0.029227</td>
      <td>1.462602</td>
      <td>1.004548</td>
    </tr>
    <tr>
      <th>1</th>
      <td>11.143388</td>
      <td>-2.017126</td>
      <td>-1.096311</td>
      <td>0.795206</td>
      <td>0.655695</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-11.141977</td>
      <td>3.897548</td>
      <td>3.258649</td>
      <td>1.526012</td>
      <td>1.452242</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-12.880687</td>
      <td>4.356965</td>
      <td>1.616418</td>
      <td>1.864823</td>
      <td>3.279771</td>
    </tr>
    <tr>
      <th>4</th>
      <td>20.844042</td>
      <td>10.905027</td>
      <td>13.402822</td>
      <td>4.405788</td>
      <td>6.753046</td>
    </tr>
  </tbody>
</table>
</div>




```python

```


```python
def rmse(x, y):
    return numpy.sqrt(numpy.mean((x-y)**2))
```


```python
print(rmse(plot_train["predict_naive_empirical_all_vars"], plot_train["y"]))
seaborn.scatterplot(x="predict_naive_empirical_all_vars", y ="y", data = plot_train)
plt = matplotlib.pyplot.title("Naive empirical prediction on train")
```

    3.2437430581263125



![png](output_33_1.png)



```python
print(rmse(plot_train["predict_naive_hierarchical_all_vars"], plot_train["y"]))
seaborn.scatterplot(x="predict_naive_hierarchical_all_vars", y ="y", data = plot_train)
plt = matplotlib.pyplot.title("Naive hierarchical prediction on train")
```

    4.687901104748199



![png](output_34_1.png)



```python
print(rmse(plot_train["predict_cross_all_vars"], plot_train["y"]))
seaborn.scatterplot(x="predict_cross_all_vars", y ="y", data = plot_train)
plt = matplotlib.pyplot.title("Cross prediction on (all vars) train")
```

    15.253371570178853



![png](output_35_1.png)



```python
print(rmse(plot_train["predict_cross_recommended_vars"], plot_train["y"]))
seaborn.scatterplot(x="predict_cross_recommended_vars", y ="y", data = plot_train)
plt = matplotlib.pyplot.title("Cross prediction (recommended vars) on train")
```

    15.632679155163398



![png](output_36_1.png)



```python
print(rmse(plot_test["predict_naive_empirical_all_vars"], plot_test["y"]))
seaborn.scatterplot(x="predict_naive_empirical_all_vars", y ="y", data = plot_test)
plt = matplotlib.pyplot.title("Naive empirical prediction on test")
```

    17.94717031406752



![png](output_37_1.png)



```python
print(rmse(plot_test["predict_naive_hierarchical_all_vars"], plot_test["y"]))
seaborn.scatterplot(x="predict_naive_hierarchical_all_vars", y ="y", data = plot_test)
plt = matplotlib.pyplot.title("Naive hierarchical prediction on test")
```

    17.681689705718867



![png](output_38_1.png)



```python
print(rmse(plot_test["predict_cross_all_vars"], plot_test["y"]))
seaborn.scatterplot(x="predict_cross_all_vars", y ="y", data = plot_test)
plt = matplotlib.pyplot.title("Cross prediction on (all vars) test")
```

    15.768320960731803



![png](output_39_1.png)



```python
print(rmse(plot_test["predict_cross_recommended_vars"], plot_test["y"]))
seaborn.scatterplot(x="predict_cross_recommended_vars", y ="y", data = plot_test)
plt = matplotlib.pyplot.title("Cross prediction (recommended vars) on test")
```

    15.493141572688373



![png](output_40_1.png)



```python
smf1 = statsmodels.api.OLS(
    y_train, 
    statsmodels.api.add_constant(naive_train_empirical[all_vars])).fit()
smf1.summary()
```




<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>            <td>y</td>        <th>  R-squared:         </th> <td>   0.968</td> 
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th> <td>   0.966</td> 
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th> <td>   664.0</td> 
</tr>
<tr>
  <th>Date:</th>             <td>Fri, 26 Jul 2019</td> <th>  Prob (F-statistic):</th>  <td>  0.00</td>  
</tr>
<tr>
  <th>Time:</th>                 <td>06:56:36</td>     <th>  Log-Likelihood:    </th> <td> -6569.6</td> 
</tr>
<tr>
  <th>No. Observations:</th>      <td>  2531</td>      <th>  AIC:               </th> <td>1.336e+04</td>
</tr>
<tr>
  <th>Df Residuals:</th>          <td>  2420</td>      <th>  BIC:               </th> <td>1.401e+04</td>
</tr>
<tr>
  <th>Df Model:</th>              <td>   110</td>      <th>                     </th>     <td> </td>    
</tr>
<tr>
  <th>Covariance Type:</th>      <td>nonrobust</td>    <th>                     </th>     <td> </td>    
</tr>
</table>
<table class="simpletable">
<tr>
            <td></td>              <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>const</th>                <td>    1.3312</td> <td>    0.066</td> <td>   20.188</td> <td> 0.000</td> <td>    1.202</td> <td>    1.460</td>
</tr>
<tr>
  <th>var_0_impact_code</th>    <td>    0.0710</td> <td>    0.008</td> <td>    8.361</td> <td> 0.000</td> <td>    0.054</td> <td>    0.088</td>
</tr>
<tr>
  <th>var_1_impact_code</th>    <td>    0.0540</td> <td>    0.008</td> <td>    6.563</td> <td> 0.000</td> <td>    0.038</td> <td>    0.070</td>
</tr>
<tr>
  <th>var_2_impact_code</th>    <td>    0.0635</td> <td>    0.008</td> <td>    7.587</td> <td> 0.000</td> <td>    0.047</td> <td>    0.080</td>
</tr>
<tr>
  <th>var_3_impact_code</th>    <td>    0.0596</td> <td>    0.008</td> <td>    7.148</td> <td> 0.000</td> <td>    0.043</td> <td>    0.076</td>
</tr>
<tr>
  <th>var_4_impact_code</th>    <td>    0.0580</td> <td>    0.008</td> <td>    6.988</td> <td> 0.000</td> <td>    0.042</td> <td>    0.074</td>
</tr>
<tr>
  <th>var_5_impact_code</th>    <td>    0.0671</td> <td>    0.008</td> <td>    8.371</td> <td> 0.000</td> <td>    0.051</td> <td>    0.083</td>
</tr>
<tr>
  <th>var_6_impact_code</th>    <td>    0.0682</td> <td>    0.008</td> <td>    8.376</td> <td> 0.000</td> <td>    0.052</td> <td>    0.084</td>
</tr>
<tr>
  <th>var_7_impact_code</th>    <td>    0.0583</td> <td>    0.008</td> <td>    7.261</td> <td> 0.000</td> <td>    0.043</td> <td>    0.074</td>
</tr>
<tr>
  <th>var_8_impact_code</th>    <td>    0.0626</td> <td>    0.008</td> <td>    7.455</td> <td> 0.000</td> <td>    0.046</td> <td>    0.079</td>
</tr>
<tr>
  <th>var_9_impact_code</th>    <td>    0.0779</td> <td>    0.008</td> <td>    9.338</td> <td> 0.000</td> <td>    0.062</td> <td>    0.094</td>
</tr>
<tr>
  <th>noise_0_impact_code</th>  <td>    0.0354</td> <td>    0.009</td> <td>    3.800</td> <td> 0.000</td> <td>    0.017</td> <td>    0.054</td>
</tr>
<tr>
  <th>noise_1_impact_code</th>  <td>    0.0365</td> <td>    0.009</td> <td>    4.039</td> <td> 0.000</td> <td>    0.019</td> <td>    0.054</td>
</tr>
<tr>
  <th>noise_2_impact_code</th>  <td>    0.0408</td> <td>    0.009</td> <td>    4.369</td> <td> 0.000</td> <td>    0.022</td> <td>    0.059</td>
</tr>
<tr>
  <th>noise_3_impact_code</th>  <td>    0.0334</td> <td>    0.010</td> <td>    3.494</td> <td> 0.000</td> <td>    0.015</td> <td>    0.052</td>
</tr>
<tr>
  <th>noise_4_impact_code</th>  <td>    0.0456</td> <td>    0.010</td> <td>    4.744</td> <td> 0.000</td> <td>    0.027</td> <td>    0.064</td>
</tr>
<tr>
  <th>noise_5_impact_code</th>  <td>    0.0369</td> <td>    0.009</td> <td>    4.009</td> <td> 0.000</td> <td>    0.019</td> <td>    0.055</td>
</tr>
<tr>
  <th>noise_6_impact_code</th>  <td>    0.0458</td> <td>    0.010</td> <td>    4.808</td> <td> 0.000</td> <td>    0.027</td> <td>    0.065</td>
</tr>
<tr>
  <th>noise_7_impact_code</th>  <td>    0.0198</td> <td>    0.010</td> <td>    2.055</td> <td> 0.040</td> <td>    0.001</td> <td>    0.039</td>
</tr>
<tr>
  <th>noise_8_impact_code</th>  <td>    0.0381</td> <td>    0.009</td> <td>    4.126</td> <td> 0.000</td> <td>    0.020</td> <td>    0.056</td>
</tr>
<tr>
  <th>noise_9_impact_code</th>  <td>    0.0315</td> <td>    0.009</td> <td>    3.320</td> <td> 0.001</td> <td>    0.013</td> <td>    0.050</td>
</tr>
<tr>
  <th>noise_10_impact_code</th> <td>    0.0322</td> <td>    0.009</td> <td>    3.432</td> <td> 0.001</td> <td>    0.014</td> <td>    0.051</td>
</tr>
<tr>
  <th>noise_11_impact_code</th> <td>    0.0479</td> <td>    0.009</td> <td>    5.147</td> <td> 0.000</td> <td>    0.030</td> <td>    0.066</td>
</tr>
<tr>
  <th>noise_12_impact_code</th> <td>    0.0376</td> <td>    0.010</td> <td>    3.841</td> <td> 0.000</td> <td>    0.018</td> <td>    0.057</td>
</tr>
<tr>
  <th>noise_13_impact_code</th> <td>    0.0389</td> <td>    0.009</td> <td>    4.139</td> <td> 0.000</td> <td>    0.020</td> <td>    0.057</td>
</tr>
<tr>
  <th>noise_14_impact_code</th> <td>    0.0423</td> <td>    0.009</td> <td>    4.517</td> <td> 0.000</td> <td>    0.024</td> <td>    0.061</td>
</tr>
<tr>
  <th>noise_15_impact_code</th> <td>    0.0410</td> <td>    0.010</td> <td>    4.298</td> <td> 0.000</td> <td>    0.022</td> <td>    0.060</td>
</tr>
<tr>
  <th>noise_16_impact_code</th> <td>    0.0316</td> <td>    0.010</td> <td>    3.290</td> <td> 0.001</td> <td>    0.013</td> <td>    0.050</td>
</tr>
<tr>
  <th>noise_17_impact_code</th> <td>    0.0452</td> <td>    0.010</td> <td>    4.702</td> <td> 0.000</td> <td>    0.026</td> <td>    0.064</td>
</tr>
<tr>
  <th>noise_18_impact_code</th> <td>    0.0347</td> <td>    0.009</td> <td>    3.770</td> <td> 0.000</td> <td>    0.017</td> <td>    0.053</td>
</tr>
<tr>
  <th>noise_19_impact_code</th> <td>    0.0506</td> <td>    0.009</td> <td>    5.457</td> <td> 0.000</td> <td>    0.032</td> <td>    0.069</td>
</tr>
<tr>
  <th>noise_20_impact_code</th> <td>    0.0544</td> <td>    0.009</td> <td>    6.012</td> <td> 0.000</td> <td>    0.037</td> <td>    0.072</td>
</tr>
<tr>
  <th>noise_21_impact_code</th> <td>    0.0479</td> <td>    0.010</td> <td>    4.979</td> <td> 0.000</td> <td>    0.029</td> <td>    0.067</td>
</tr>
<tr>
  <th>noise_22_impact_code</th> <td>    0.0279</td> <td>    0.010</td> <td>    2.880</td> <td> 0.004</td> <td>    0.009</td> <td>    0.047</td>
</tr>
<tr>
  <th>noise_23_impact_code</th> <td>    0.0438</td> <td>    0.009</td> <td>    4.618</td> <td> 0.000</td> <td>    0.025</td> <td>    0.062</td>
</tr>
<tr>
  <th>noise_24_impact_code</th> <td>    0.0310</td> <td>    0.009</td> <td>    3.304</td> <td> 0.001</td> <td>    0.013</td> <td>    0.049</td>
</tr>
<tr>
  <th>noise_25_impact_code</th> <td>    0.0543</td> <td>    0.009</td> <td>    5.778</td> <td> 0.000</td> <td>    0.036</td> <td>    0.073</td>
</tr>
<tr>
  <th>noise_26_impact_code</th> <td>    0.0557</td> <td>    0.009</td> <td>    6.255</td> <td> 0.000</td> <td>    0.038</td> <td>    0.073</td>
</tr>
<tr>
  <th>noise_27_impact_code</th> <td>    0.0583</td> <td>    0.009</td> <td>    6.245</td> <td> 0.000</td> <td>    0.040</td> <td>    0.077</td>
</tr>
<tr>
  <th>noise_28_impact_code</th> <td>    0.0268</td> <td>    0.010</td> <td>    2.793</td> <td> 0.005</td> <td>    0.008</td> <td>    0.046</td>
</tr>
<tr>
  <th>noise_29_impact_code</th> <td>    0.0292</td> <td>    0.010</td> <td>    3.073</td> <td> 0.002</td> <td>    0.011</td> <td>    0.048</td>
</tr>
<tr>
  <th>noise_30_impact_code</th> <td>    0.0421</td> <td>    0.009</td> <td>    4.439</td> <td> 0.000</td> <td>    0.024</td> <td>    0.061</td>
</tr>
<tr>
  <th>noise_31_impact_code</th> <td>    0.0380</td> <td>    0.009</td> <td>    4.166</td> <td> 0.000</td> <td>    0.020</td> <td>    0.056</td>
</tr>
<tr>
  <th>noise_32_impact_code</th> <td>    0.0448</td> <td>    0.009</td> <td>    4.836</td> <td> 0.000</td> <td>    0.027</td> <td>    0.063</td>
</tr>
<tr>
  <th>noise_33_impact_code</th> <td>    0.0533</td> <td>    0.009</td> <td>    5.778</td> <td> 0.000</td> <td>    0.035</td> <td>    0.071</td>
</tr>
<tr>
  <th>noise_34_impact_code</th> <td>    0.0336</td> <td>    0.009</td> <td>    3.572</td> <td> 0.000</td> <td>    0.015</td> <td>    0.052</td>
</tr>
<tr>
  <th>noise_35_impact_code</th> <td>    0.0340</td> <td>    0.010</td> <td>    3.527</td> <td> 0.000</td> <td>    0.015</td> <td>    0.053</td>
</tr>
<tr>
  <th>noise_36_impact_code</th> <td>    0.0371</td> <td>    0.009</td> <td>    4.040</td> <td> 0.000</td> <td>    0.019</td> <td>    0.055</td>
</tr>
<tr>
  <th>noise_37_impact_code</th> <td>    0.0428</td> <td>    0.009</td> <td>    4.558</td> <td> 0.000</td> <td>    0.024</td> <td>    0.061</td>
</tr>
<tr>
  <th>noise_38_impact_code</th> <td>    0.0410</td> <td>    0.009</td> <td>    4.460</td> <td> 0.000</td> <td>    0.023</td> <td>    0.059</td>
</tr>
<tr>
  <th>noise_39_impact_code</th> <td>    0.0460</td> <td>    0.009</td> <td>    4.907</td> <td> 0.000</td> <td>    0.028</td> <td>    0.064</td>
</tr>
<tr>
  <th>noise_40_impact_code</th> <td>    0.0502</td> <td>    0.009</td> <td>    5.299</td> <td> 0.000</td> <td>    0.032</td> <td>    0.069</td>
</tr>
<tr>
  <th>noise_41_impact_code</th> <td>    0.0486</td> <td>    0.009</td> <td>    5.141</td> <td> 0.000</td> <td>    0.030</td> <td>    0.067</td>
</tr>
<tr>
  <th>noise_42_impact_code</th> <td>    0.0419</td> <td>    0.009</td> <td>    4.687</td> <td> 0.000</td> <td>    0.024</td> <td>    0.059</td>
</tr>
<tr>
  <th>noise_43_impact_code</th> <td>    0.0433</td> <td>    0.009</td> <td>    4.564</td> <td> 0.000</td> <td>    0.025</td> <td>    0.062</td>
</tr>
<tr>
  <th>noise_44_impact_code</th> <td>    0.0430</td> <td>    0.009</td> <td>    4.596</td> <td> 0.000</td> <td>    0.025</td> <td>    0.061</td>
</tr>
<tr>
  <th>noise_45_impact_code</th> <td>    0.0433</td> <td>    0.009</td> <td>    4.572</td> <td> 0.000</td> <td>    0.025</td> <td>    0.062</td>
</tr>
<tr>
  <th>noise_46_impact_code</th> <td>    0.0356</td> <td>    0.009</td> <td>    3.781</td> <td> 0.000</td> <td>    0.017</td> <td>    0.054</td>
</tr>
<tr>
  <th>noise_47_impact_code</th> <td>    0.0428</td> <td>    0.009</td> <td>    4.768</td> <td> 0.000</td> <td>    0.025</td> <td>    0.060</td>
</tr>
<tr>
  <th>noise_48_impact_code</th> <td>    0.0428</td> <td>    0.009</td> <td>    4.606</td> <td> 0.000</td> <td>    0.025</td> <td>    0.061</td>
</tr>
<tr>
  <th>noise_49_impact_code</th> <td>    0.0447</td> <td>    0.009</td> <td>    4.788</td> <td> 0.000</td> <td>    0.026</td> <td>    0.063</td>
</tr>
<tr>
  <th>noise_50_impact_code</th> <td>    0.0350</td> <td>    0.009</td> <td>    3.729</td> <td> 0.000</td> <td>    0.017</td> <td>    0.053</td>
</tr>
<tr>
  <th>noise_51_impact_code</th> <td>    0.0328</td> <td>    0.009</td> <td>    3.506</td> <td> 0.000</td> <td>    0.014</td> <td>    0.051</td>
</tr>
<tr>
  <th>noise_52_impact_code</th> <td>    0.0527</td> <td>    0.009</td> <td>    5.654</td> <td> 0.000</td> <td>    0.034</td> <td>    0.071</td>
</tr>
<tr>
  <th>noise_53_impact_code</th> <td>    0.0453</td> <td>    0.009</td> <td>    4.938</td> <td> 0.000</td> <td>    0.027</td> <td>    0.063</td>
</tr>
<tr>
  <th>noise_54_impact_code</th> <td>    0.0352</td> <td>    0.009</td> <td>    3.761</td> <td> 0.000</td> <td>    0.017</td> <td>    0.054</td>
</tr>
<tr>
  <th>noise_55_impact_code</th> <td>    0.0654</td> <td>    0.009</td> <td>    6.951</td> <td> 0.000</td> <td>    0.047</td> <td>    0.084</td>
</tr>
<tr>
  <th>noise_56_impact_code</th> <td>    0.0355</td> <td>    0.009</td> <td>    3.813</td> <td> 0.000</td> <td>    0.017</td> <td>    0.054</td>
</tr>
<tr>
  <th>noise_57_impact_code</th> <td>    0.0443</td> <td>    0.010</td> <td>    4.592</td> <td> 0.000</td> <td>    0.025</td> <td>    0.063</td>
</tr>
<tr>
  <th>noise_58_impact_code</th> <td>    0.0479</td> <td>    0.009</td> <td>    5.194</td> <td> 0.000</td> <td>    0.030</td> <td>    0.066</td>
</tr>
<tr>
  <th>noise_59_impact_code</th> <td>    0.0507</td> <td>    0.010</td> <td>    5.254</td> <td> 0.000</td> <td>    0.032</td> <td>    0.070</td>
</tr>
<tr>
  <th>noise_60_impact_code</th> <td>    0.0250</td> <td>    0.010</td> <td>    2.625</td> <td> 0.009</td> <td>    0.006</td> <td>    0.044</td>
</tr>
<tr>
  <th>noise_61_impact_code</th> <td>    0.0399</td> <td>    0.010</td> <td>    4.191</td> <td> 0.000</td> <td>    0.021</td> <td>    0.059</td>
</tr>
<tr>
  <th>noise_62_impact_code</th> <td>    0.0332</td> <td>    0.009</td> <td>    3.667</td> <td> 0.000</td> <td>    0.015</td> <td>    0.051</td>
</tr>
<tr>
  <th>noise_63_impact_code</th> <td>    0.0275</td> <td>    0.010</td> <td>    2.865</td> <td> 0.004</td> <td>    0.009</td> <td>    0.046</td>
</tr>
<tr>
  <th>noise_64_impact_code</th> <td>    0.0319</td> <td>    0.009</td> <td>    3.506</td> <td> 0.000</td> <td>    0.014</td> <td>    0.050</td>
</tr>
<tr>
  <th>noise_65_impact_code</th> <td>    0.0334</td> <td>    0.010</td> <td>    3.499</td> <td> 0.000</td> <td>    0.015</td> <td>    0.052</td>
</tr>
<tr>
  <th>noise_66_impact_code</th> <td>    0.0197</td> <td>    0.009</td> <td>    2.151</td> <td> 0.032</td> <td>    0.002</td> <td>    0.038</td>
</tr>
<tr>
  <th>noise_67_impact_code</th> <td>    0.0355</td> <td>    0.009</td> <td>    3.831</td> <td> 0.000</td> <td>    0.017</td> <td>    0.054</td>
</tr>
<tr>
  <th>noise_68_impact_code</th> <td>    0.0482</td> <td>    0.009</td> <td>    5.270</td> <td> 0.000</td> <td>    0.030</td> <td>    0.066</td>
</tr>
<tr>
  <th>noise_69_impact_code</th> <td>    0.0217</td> <td>    0.010</td> <td>    2.273</td> <td> 0.023</td> <td>    0.003</td> <td>    0.040</td>
</tr>
<tr>
  <th>noise_70_impact_code</th> <td>    0.0319</td> <td>    0.010</td> <td>    3.319</td> <td> 0.001</td> <td>    0.013</td> <td>    0.051</td>
</tr>
<tr>
  <th>noise_71_impact_code</th> <td>    0.0366</td> <td>    0.009</td> <td>    3.926</td> <td> 0.000</td> <td>    0.018</td> <td>    0.055</td>
</tr>
<tr>
  <th>noise_72_impact_code</th> <td>    0.0347</td> <td>    0.010</td> <td>    3.632</td> <td> 0.000</td> <td>    0.016</td> <td>    0.053</td>
</tr>
<tr>
  <th>noise_73_impact_code</th> <td>    0.0505</td> <td>    0.009</td> <td>    5.417</td> <td> 0.000</td> <td>    0.032</td> <td>    0.069</td>
</tr>
<tr>
  <th>noise_74_impact_code</th> <td>    0.0268</td> <td>    0.009</td> <td>    2.841</td> <td> 0.005</td> <td>    0.008</td> <td>    0.045</td>
</tr>
<tr>
  <th>noise_75_impact_code</th> <td>    0.0454</td> <td>    0.010</td> <td>    4.696</td> <td> 0.000</td> <td>    0.026</td> <td>    0.064</td>
</tr>
<tr>
  <th>noise_76_impact_code</th> <td>    0.0286</td> <td>    0.009</td> <td>    3.068</td> <td> 0.002</td> <td>    0.010</td> <td>    0.047</td>
</tr>
<tr>
  <th>noise_77_impact_code</th> <td>    0.0362</td> <td>    0.009</td> <td>    3.980</td> <td> 0.000</td> <td>    0.018</td> <td>    0.054</td>
</tr>
<tr>
  <th>noise_78_impact_code</th> <td>    0.0298</td> <td>    0.010</td> <td>    3.110</td> <td> 0.002</td> <td>    0.011</td> <td>    0.049</td>
</tr>
<tr>
  <th>noise_79_impact_code</th> <td>    0.0409</td> <td>    0.009</td> <td>    4.469</td> <td> 0.000</td> <td>    0.023</td> <td>    0.059</td>
</tr>
<tr>
  <th>noise_80_impact_code</th> <td>    0.0539</td> <td>    0.010</td> <td>    5.609</td> <td> 0.000</td> <td>    0.035</td> <td>    0.073</td>
</tr>
<tr>
  <th>noise_81_impact_code</th> <td>    0.0366</td> <td>    0.009</td> <td>    4.065</td> <td> 0.000</td> <td>    0.019</td> <td>    0.054</td>
</tr>
<tr>
  <th>noise_82_impact_code</th> <td>    0.0356</td> <td>    0.009</td> <td>    3.795</td> <td> 0.000</td> <td>    0.017</td> <td>    0.054</td>
</tr>
<tr>
  <th>noise_83_impact_code</th> <td>    0.0400</td> <td>    0.009</td> <td>    4.299</td> <td> 0.000</td> <td>    0.022</td> <td>    0.058</td>
</tr>
<tr>
  <th>noise_84_impact_code</th> <td>    0.0525</td> <td>    0.009</td> <td>    5.542</td> <td> 0.000</td> <td>    0.034</td> <td>    0.071</td>
</tr>
<tr>
  <th>noise_85_impact_code</th> <td>    0.0319</td> <td>    0.009</td> <td>    3.382</td> <td> 0.001</td> <td>    0.013</td> <td>    0.050</td>
</tr>
<tr>
  <th>noise_86_impact_code</th> <td>    0.0351</td> <td>    0.009</td> <td>    3.720</td> <td> 0.000</td> <td>    0.017</td> <td>    0.054</td>
</tr>
<tr>
  <th>noise_87_impact_code</th> <td>    0.0412</td> <td>    0.009</td> <td>    4.580</td> <td> 0.000</td> <td>    0.024</td> <td>    0.059</td>
</tr>
<tr>
  <th>noise_88_impact_code</th> <td>    0.0579</td> <td>    0.009</td> <td>    6.366</td> <td> 0.000</td> <td>    0.040</td> <td>    0.076</td>
</tr>
<tr>
  <th>noise_89_impact_code</th> <td>    0.0432</td> <td>    0.010</td> <td>    4.547</td> <td> 0.000</td> <td>    0.025</td> <td>    0.062</td>
</tr>
<tr>
  <th>noise_90_impact_code</th> <td>    0.0482</td> <td>    0.010</td> <td>    5.007</td> <td> 0.000</td> <td>    0.029</td> <td>    0.067</td>
</tr>
<tr>
  <th>noise_91_impact_code</th> <td>    0.0367</td> <td>    0.009</td> <td>    4.028</td> <td> 0.000</td> <td>    0.019</td> <td>    0.055</td>
</tr>
<tr>
  <th>noise_92_impact_code</th> <td>    0.0387</td> <td>    0.009</td> <td>    4.122</td> <td> 0.000</td> <td>    0.020</td> <td>    0.057</td>
</tr>
<tr>
  <th>noise_93_impact_code</th> <td>    0.0440</td> <td>    0.009</td> <td>    5.008</td> <td> 0.000</td> <td>    0.027</td> <td>    0.061</td>
</tr>
<tr>
  <th>noise_94_impact_code</th> <td>    0.0370</td> <td>    0.009</td> <td>    3.921</td> <td> 0.000</td> <td>    0.019</td> <td>    0.056</td>
</tr>
<tr>
  <th>noise_95_impact_code</th> <td>    0.0393</td> <td>    0.009</td> <td>    4.339</td> <td> 0.000</td> <td>    0.022</td> <td>    0.057</td>
</tr>
<tr>
  <th>noise_96_impact_code</th> <td>    0.0445</td> <td>    0.009</td> <td>    4.930</td> <td> 0.000</td> <td>    0.027</td> <td>    0.062</td>
</tr>
<tr>
  <th>noise_97_impact_code</th> <td>    0.0442</td> <td>    0.009</td> <td>    4.891</td> <td> 0.000</td> <td>    0.026</td> <td>    0.062</td>
</tr>
<tr>
  <th>noise_98_impact_code</th> <td>    0.0337</td> <td>    0.009</td> <td>    3.634</td> <td> 0.000</td> <td>    0.015</td> <td>    0.052</td>
</tr>
<tr>
  <th>noise_99_impact_code</th> <td>    0.0603</td> <td>    0.009</td> <td>    6.554</td> <td> 0.000</td> <td>    0.042</td> <td>    0.078</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td> 3.367</td> <th>  Durbin-Watson:     </th> <td>   1.964</td>
</tr>
<tr>
  <th>Prob(Omnibus):</th> <td> 0.186</td> <th>  Jarque-Bera (JB):  </th> <td>   3.549</td>
</tr>
<tr>
  <th>Skew:</th>          <td> 0.035</td> <th>  Prob(JB):          </th> <td>   0.170</td>
</tr>
<tr>
  <th>Kurtosis:</th>      <td> 3.169</td> <th>  Cond. No.          </th> <td>    39.7</td>
</tr>
</table><br/><br/>Warnings:<br/>[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.




```python
sklearn.metrics.r2_score(plot_train["y"],plot_train["predict_naive_empirical_all_vars"])
```




    0.9679285560701428




```python
sklearn.metrics.r2_score(plot_test["y"],plot_test["predict_naive_empirical_all_vars"])
```




    0.06528961393944832




```python
smf2 = statsmodels.api.OLS(
    y_train, 
    statsmodels.api.add_constant(cross_frame[all_vars])).fit()
smf2.summary()
```




<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>            <td>y</td>        <th>  R-squared:         </th> <td>   0.291</td> 
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th> <td>   0.259</td> 
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th> <td>   9.022</td> 
</tr>
<tr>
  <th>Date:</th>             <td>Fri, 26 Jul 2019</td> <th>  Prob (F-statistic):</th> <td>1.36e-114</td>
</tr>
<tr>
  <th>Time:</th>                 <td>06:56:36</td>     <th>  Log-Likelihood:    </th> <td> -10488.</td> 
</tr>
<tr>
  <th>No. Observations:</th>      <td>  2531</td>      <th>  AIC:               </th> <td>2.120e+04</td>
</tr>
<tr>
  <th>Df Residuals:</th>          <td>  2420</td>      <th>  BIC:               </th> <td>2.185e+04</td>
</tr>
<tr>
  <th>Df Model:</th>              <td>   110</td>      <th>                     </th>     <td> </td>    
</tr>
<tr>
  <th>Covariance Type:</th>      <td>nonrobust</td>    <th>                     </th>     <td> </td>    
</tr>
</table>
<table class="simpletable">
<tr>
            <td></td>              <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>const</th>                <td>    1.4072</td> <td>    0.316</td> <td>    4.459</td> <td> 0.000</td> <td>    0.788</td> <td>    2.026</td>
</tr>
<tr>
  <th>var_0_impact_code</th>    <td>    0.4141</td> <td>    0.049</td> <td>    8.369</td> <td> 0.000</td> <td>    0.317</td> <td>    0.511</td>
</tr>
<tr>
  <th>var_1_impact_code</th>    <td>    0.3821</td> <td>    0.045</td> <td>    8.436</td> <td> 0.000</td> <td>    0.293</td> <td>    0.471</td>
</tr>
<tr>
  <th>var_2_impact_code</th>    <td>    0.4082</td> <td>    0.046</td> <td>    8.794</td> <td> 0.000</td> <td>    0.317</td> <td>    0.499</td>
</tr>
<tr>
  <th>var_3_impact_code</th>    <td>    0.3654</td> <td>    0.049</td> <td>    7.493</td> <td> 0.000</td> <td>    0.270</td> <td>    0.461</td>
</tr>
<tr>
  <th>var_4_impact_code</th>    <td>    0.3920</td> <td>    0.048</td> <td>    8.170</td> <td> 0.000</td> <td>    0.298</td> <td>    0.486</td>
</tr>
<tr>
  <th>var_5_impact_code</th>    <td>    0.4784</td> <td>    0.044</td> <td>   10.816</td> <td> 0.000</td> <td>    0.392</td> <td>    0.565</td>
</tr>
<tr>
  <th>var_6_impact_code</th>    <td>    0.4373</td> <td>    0.045</td> <td>    9.675</td> <td> 0.000</td> <td>    0.349</td> <td>    0.526</td>
</tr>
<tr>
  <th>var_7_impact_code</th>    <td>    0.4361</td> <td>    0.044</td> <td>    9.986</td> <td> 0.000</td> <td>    0.350</td> <td>    0.522</td>
</tr>
<tr>
  <th>var_8_impact_code</th>    <td>    0.3955</td> <td>    0.048</td> <td>    8.230</td> <td> 0.000</td> <td>    0.301</td> <td>    0.490</td>
</tr>
<tr>
  <th>var_9_impact_code</th>    <td>    0.3835</td> <td>    0.048</td> <td>    8.022</td> <td> 0.000</td> <td>    0.290</td> <td>    0.477</td>
</tr>
<tr>
  <th>noise_0_impact_code</th>  <td>    0.0126</td> <td>    0.061</td> <td>    0.205</td> <td> 0.838</td> <td>   -0.108</td> <td>    0.133</td>
</tr>
<tr>
  <th>noise_1_impact_code</th>  <td>    0.0169</td> <td>    0.054</td> <td>    0.311</td> <td> 0.756</td> <td>   -0.090</td> <td>    0.124</td>
</tr>
<tr>
  <th>noise_2_impact_code</th>  <td>    0.0517</td> <td>    0.060</td> <td>    0.856</td> <td> 0.392</td> <td>   -0.067</td> <td>    0.170</td>
</tr>
<tr>
  <th>noise_3_impact_code</th>  <td>   -0.0431</td> <td>    0.058</td> <td>   -0.746</td> <td> 0.456</td> <td>   -0.156</td> <td>    0.070</td>
</tr>
<tr>
  <th>noise_4_impact_code</th>  <td>   -0.0668</td> <td>    0.059</td> <td>   -1.130</td> <td> 0.259</td> <td>   -0.183</td> <td>    0.049</td>
</tr>
<tr>
  <th>noise_5_impact_code</th>  <td>    0.0727</td> <td>    0.056</td> <td>    1.303</td> <td> 0.193</td> <td>   -0.037</td> <td>    0.182</td>
</tr>
<tr>
  <th>noise_6_impact_code</th>  <td>    0.0114</td> <td>    0.057</td> <td>    0.200</td> <td> 0.842</td> <td>   -0.100</td> <td>    0.123</td>
</tr>
<tr>
  <th>noise_7_impact_code</th>  <td>   -0.0737</td> <td>    0.062</td> <td>   -1.183</td> <td> 0.237</td> <td>   -0.196</td> <td>    0.048</td>
</tr>
<tr>
  <th>noise_8_impact_code</th>  <td>    0.1413</td> <td>    0.057</td> <td>    2.500</td> <td> 0.012</td> <td>    0.030</td> <td>    0.252</td>
</tr>
<tr>
  <th>noise_9_impact_code</th>  <td>   -0.0822</td> <td>    0.061</td> <td>   -1.348</td> <td> 0.178</td> <td>   -0.202</td> <td>    0.037</td>
</tr>
<tr>
  <th>noise_10_impact_code</th> <td>   -0.0523</td> <td>    0.061</td> <td>   -0.861</td> <td> 0.390</td> <td>   -0.171</td> <td>    0.067</td>
</tr>
<tr>
  <th>noise_11_impact_code</th> <td>   -0.0121</td> <td>    0.058</td> <td>   -0.210</td> <td> 0.834</td> <td>   -0.126</td> <td>    0.101</td>
</tr>
<tr>
  <th>noise_12_impact_code</th> <td>   -0.2130</td> <td>    0.062</td> <td>   -3.422</td> <td> 0.001</td> <td>   -0.335</td> <td>   -0.091</td>
</tr>
<tr>
  <th>noise_13_impact_code</th> <td>   -0.0388</td> <td>    0.058</td> <td>   -0.670</td> <td> 0.503</td> <td>   -0.152</td> <td>    0.075</td>
</tr>
<tr>
  <th>noise_14_impact_code</th> <td>   -0.0503</td> <td>    0.059</td> <td>   -0.855</td> <td> 0.393</td> <td>   -0.166</td> <td>    0.065</td>
</tr>
<tr>
  <th>noise_15_impact_code</th> <td>    0.0004</td> <td>    0.057</td> <td>    0.006</td> <td> 0.995</td> <td>   -0.112</td> <td>    0.113</td>
</tr>
<tr>
  <th>noise_16_impact_code</th> <td>   -0.0348</td> <td>    0.060</td> <td>   -0.578</td> <td> 0.563</td> <td>   -0.153</td> <td>    0.083</td>
</tr>
<tr>
  <th>noise_17_impact_code</th> <td>    0.0113</td> <td>    0.057</td> <td>    0.199</td> <td> 0.842</td> <td>   -0.100</td> <td>    0.123</td>
</tr>
<tr>
  <th>noise_18_impact_code</th> <td>    0.1067</td> <td>    0.058</td> <td>    1.839</td> <td> 0.066</td> <td>   -0.007</td> <td>    0.221</td>
</tr>
<tr>
  <th>noise_19_impact_code</th> <td>   -0.0402</td> <td>    0.056</td> <td>   -0.713</td> <td> 0.476</td> <td>   -0.151</td> <td>    0.070</td>
</tr>
<tr>
  <th>noise_20_impact_code</th> <td>    0.0613</td> <td>    0.057</td> <td>    1.080</td> <td> 0.280</td> <td>   -0.050</td> <td>    0.173</td>
</tr>
<tr>
  <th>noise_21_impact_code</th> <td>   -0.0879</td> <td>    0.059</td> <td>   -1.482</td> <td> 0.139</td> <td>   -0.204</td> <td>    0.028</td>
</tr>
<tr>
  <th>noise_22_impact_code</th> <td>   -0.1077</td> <td>    0.058</td> <td>   -1.844</td> <td> 0.065</td> <td>   -0.222</td> <td>    0.007</td>
</tr>
<tr>
  <th>noise_23_impact_code</th> <td>   -0.0557</td> <td>    0.057</td> <td>   -0.976</td> <td> 0.329</td> <td>   -0.168</td> <td>    0.056</td>
</tr>
<tr>
  <th>noise_24_impact_code</th> <td>   -0.0321</td> <td>    0.056</td> <td>   -0.569</td> <td> 0.569</td> <td>   -0.142</td> <td>    0.078</td>
</tr>
<tr>
  <th>noise_25_impact_code</th> <td>   -0.0159</td> <td>    0.057</td> <td>   -0.279</td> <td> 0.781</td> <td>   -0.128</td> <td>    0.096</td>
</tr>
<tr>
  <th>noise_26_impact_code</th> <td>    0.0443</td> <td>    0.055</td> <td>    0.807</td> <td> 0.420</td> <td>   -0.063</td> <td>    0.152</td>
</tr>
<tr>
  <th>noise_27_impact_code</th> <td>    0.0185</td> <td>    0.060</td> <td>    0.307</td> <td> 0.759</td> <td>   -0.099</td> <td>    0.136</td>
</tr>
<tr>
  <th>noise_28_impact_code</th> <td>   -0.0661</td> <td>    0.060</td> <td>   -1.096</td> <td> 0.273</td> <td>   -0.184</td> <td>    0.052</td>
</tr>
<tr>
  <th>noise_29_impact_code</th> <td>   -0.0772</td> <td>    0.059</td> <td>   -1.317</td> <td> 0.188</td> <td>   -0.192</td> <td>    0.038</td>
</tr>
<tr>
  <th>noise_30_impact_code</th> <td>   -0.0319</td> <td>    0.062</td> <td>   -0.514</td> <td> 0.607</td> <td>   -0.153</td> <td>    0.090</td>
</tr>
<tr>
  <th>noise_31_impact_code</th> <td>    0.0752</td> <td>    0.056</td> <td>    1.341</td> <td> 0.180</td> <td>   -0.035</td> <td>    0.185</td>
</tr>
<tr>
  <th>noise_32_impact_code</th> <td>   -0.0195</td> <td>    0.059</td> <td>   -0.334</td> <td> 0.739</td> <td>   -0.134</td> <td>    0.095</td>
</tr>
<tr>
  <th>noise_33_impact_code</th> <td>   -0.0128</td> <td>    0.054</td> <td>   -0.236</td> <td> 0.813</td> <td>   -0.119</td> <td>    0.094</td>
</tr>
<tr>
  <th>noise_34_impact_code</th> <td>    0.0690</td> <td>    0.057</td> <td>    1.207</td> <td> 0.228</td> <td>   -0.043</td> <td>    0.181</td>
</tr>
<tr>
  <th>noise_35_impact_code</th> <td>   -0.0245</td> <td>    0.058</td> <td>   -0.425</td> <td> 0.671</td> <td>   -0.138</td> <td>    0.089</td>
</tr>
<tr>
  <th>noise_36_impact_code</th> <td>    0.0423</td> <td>    0.055</td> <td>    0.763</td> <td> 0.446</td> <td>   -0.066</td> <td>    0.151</td>
</tr>
<tr>
  <th>noise_37_impact_code</th> <td>   -0.0346</td> <td>    0.060</td> <td>   -0.582</td> <td> 0.561</td> <td>   -0.151</td> <td>    0.082</td>
</tr>
<tr>
  <th>noise_38_impact_code</th> <td>    0.0355</td> <td>    0.058</td> <td>    0.616</td> <td> 0.538</td> <td>   -0.077</td> <td>    0.148</td>
</tr>
<tr>
  <th>noise_39_impact_code</th> <td>    0.0113</td> <td>    0.056</td> <td>    0.202</td> <td> 0.840</td> <td>   -0.098</td> <td>    0.121</td>
</tr>
<tr>
  <th>noise_40_impact_code</th> <td>    0.0257</td> <td>    0.058</td> <td>    0.446</td> <td> 0.655</td> <td>   -0.087</td> <td>    0.139</td>
</tr>
<tr>
  <th>noise_41_impact_code</th> <td>   -0.0015</td> <td>    0.059</td> <td>   -0.025</td> <td> 0.980</td> <td>   -0.117</td> <td>    0.114</td>
</tr>
<tr>
  <th>noise_42_impact_code</th> <td>    0.0896</td> <td>    0.054</td> <td>    1.644</td> <td> 0.100</td> <td>   -0.017</td> <td>    0.196</td>
</tr>
<tr>
  <th>noise_43_impact_code</th> <td>   -0.0691</td> <td>    0.057</td> <td>   -1.214</td> <td> 0.225</td> <td>   -0.181</td> <td>    0.042</td>
</tr>
<tr>
  <th>noise_44_impact_code</th> <td>    0.0036</td> <td>    0.059</td> <td>    0.060</td> <td> 0.952</td> <td>   -0.113</td> <td>    0.120</td>
</tr>
<tr>
  <th>noise_45_impact_code</th> <td>   -0.0347</td> <td>    0.063</td> <td>   -0.554</td> <td> 0.579</td> <td>   -0.158</td> <td>    0.088</td>
</tr>
<tr>
  <th>noise_46_impact_code</th> <td>   -0.0365</td> <td>    0.060</td> <td>   -0.610</td> <td> 0.542</td> <td>   -0.154</td> <td>    0.081</td>
</tr>
<tr>
  <th>noise_47_impact_code</th> <td>    0.0878</td> <td>    0.056</td> <td>    1.561</td> <td> 0.119</td> <td>   -0.022</td> <td>    0.198</td>
</tr>
<tr>
  <th>noise_48_impact_code</th> <td>    0.1114</td> <td>    0.058</td> <td>    1.905</td> <td> 0.057</td> <td>   -0.003</td> <td>    0.226</td>
</tr>
<tr>
  <th>noise_49_impact_code</th> <td>   -0.0212</td> <td>    0.057</td> <td>   -0.372</td> <td> 0.710</td> <td>   -0.133</td> <td>    0.091</td>
</tr>
<tr>
  <th>noise_50_impact_code</th> <td>    0.0117</td> <td>    0.057</td> <td>    0.204</td> <td> 0.838</td> <td>   -0.101</td> <td>    0.124</td>
</tr>
<tr>
  <th>noise_51_impact_code</th> <td>   -0.0353</td> <td>    0.057</td> <td>   -0.615</td> <td> 0.539</td> <td>   -0.148</td> <td>    0.077</td>
</tr>
<tr>
  <th>noise_52_impact_code</th> <td>   -0.0187</td> <td>    0.058</td> <td>   -0.321</td> <td> 0.748</td> <td>   -0.133</td> <td>    0.095</td>
</tr>
<tr>
  <th>noise_53_impact_code</th> <td>    0.0426</td> <td>    0.057</td> <td>    0.748</td> <td> 0.455</td> <td>   -0.069</td> <td>    0.154</td>
</tr>
<tr>
  <th>noise_54_impact_code</th> <td>    0.0012</td> <td>    0.058</td> <td>    0.020</td> <td> 0.984</td> <td>   -0.112</td> <td>    0.114</td>
</tr>
<tr>
  <th>noise_55_impact_code</th> <td>    0.0496</td> <td>    0.056</td> <td>    0.892</td> <td> 0.372</td> <td>   -0.059</td> <td>    0.159</td>
</tr>
<tr>
  <th>noise_56_impact_code</th> <td>   -0.0151</td> <td>    0.059</td> <td>   -0.255</td> <td> 0.799</td> <td>   -0.131</td> <td>    0.101</td>
</tr>
<tr>
  <th>noise_57_impact_code</th> <td>   -0.0757</td> <td>    0.061</td> <td>   -1.244</td> <td> 0.213</td> <td>   -0.195</td> <td>    0.044</td>
</tr>
<tr>
  <th>noise_58_impact_code</th> <td>    0.0726</td> <td>    0.058</td> <td>    1.246</td> <td> 0.213</td> <td>   -0.042</td> <td>    0.187</td>
</tr>
<tr>
  <th>noise_59_impact_code</th> <td>    0.0656</td> <td>    0.063</td> <td>    1.039</td> <td> 0.299</td> <td>   -0.058</td> <td>    0.189</td>
</tr>
<tr>
  <th>noise_60_impact_code</th> <td>   -0.1550</td> <td>    0.060</td> <td>   -2.598</td> <td> 0.009</td> <td>   -0.272</td> <td>   -0.038</td>
</tr>
<tr>
  <th>noise_61_impact_code</th> <td>   -0.0610</td> <td>    0.059</td> <td>   -1.036</td> <td> 0.301</td> <td>   -0.177</td> <td>    0.055</td>
</tr>
<tr>
  <th>noise_62_impact_code</th> <td>    0.0309</td> <td>    0.059</td> <td>    0.527</td> <td> 0.598</td> <td>   -0.084</td> <td>    0.146</td>
</tr>
<tr>
  <th>noise_63_impact_code</th> <td>   -0.0074</td> <td>    0.059</td> <td>   -0.126</td> <td> 0.899</td> <td>   -0.123</td> <td>    0.108</td>
</tr>
<tr>
  <th>noise_64_impact_code</th> <td>    0.0227</td> <td>    0.056</td> <td>    0.407</td> <td> 0.684</td> <td>   -0.087</td> <td>    0.132</td>
</tr>
<tr>
  <th>noise_65_impact_code</th> <td>   -0.1022</td> <td>    0.056</td> <td>   -1.823</td> <td> 0.068</td> <td>   -0.212</td> <td>    0.008</td>
</tr>
<tr>
  <th>noise_66_impact_code</th> <td>   -0.0137</td> <td>    0.054</td> <td>   -0.256</td> <td> 0.798</td> <td>   -0.119</td> <td>    0.091</td>
</tr>
<tr>
  <th>noise_67_impact_code</th> <td>   -0.0350</td> <td>    0.058</td> <td>   -0.604</td> <td> 0.546</td> <td>   -0.149</td> <td>    0.079</td>
</tr>
<tr>
  <th>noise_68_impact_code</th> <td>   -0.0125</td> <td>    0.058</td> <td>   -0.216</td> <td> 0.829</td> <td>   -0.126</td> <td>    0.101</td>
</tr>
<tr>
  <th>noise_69_impact_code</th> <td>   -0.1775</td> <td>    0.059</td> <td>   -3.018</td> <td> 0.003</td> <td>   -0.293</td> <td>   -0.062</td>
</tr>
<tr>
  <th>noise_70_impact_code</th> <td>   -0.0053</td> <td>    0.060</td> <td>   -0.088</td> <td> 0.930</td> <td>   -0.123</td> <td>    0.112</td>
</tr>
<tr>
  <th>noise_71_impact_code</th> <td>   -0.0268</td> <td>    0.056</td> <td>   -0.476</td> <td> 0.634</td> <td>   -0.137</td> <td>    0.083</td>
</tr>
<tr>
  <th>noise_72_impact_code</th> <td>   -0.0183</td> <td>    0.057</td> <td>   -0.321</td> <td> 0.748</td> <td>   -0.130</td> <td>    0.093</td>
</tr>
<tr>
  <th>noise_73_impact_code</th> <td>    0.0160</td> <td>    0.057</td> <td>    0.279</td> <td> 0.781</td> <td>   -0.096</td> <td>    0.128</td>
</tr>
<tr>
  <th>noise_74_impact_code</th> <td>    0.0669</td> <td>    0.061</td> <td>    1.095</td> <td> 0.273</td> <td>   -0.053</td> <td>    0.187</td>
</tr>
<tr>
  <th>noise_75_impact_code</th> <td>   -0.0748</td> <td>    0.063</td> <td>   -1.188</td> <td> 0.235</td> <td>   -0.198</td> <td>    0.049</td>
</tr>
<tr>
  <th>noise_76_impact_code</th> <td>    0.0508</td> <td>    0.055</td> <td>    0.919</td> <td> 0.358</td> <td>   -0.058</td> <td>    0.159</td>
</tr>
<tr>
  <th>noise_77_impact_code</th> <td>    0.0462</td> <td>    0.057</td> <td>    0.812</td> <td> 0.417</td> <td>   -0.065</td> <td>    0.158</td>
</tr>
<tr>
  <th>noise_78_impact_code</th> <td>    0.0545</td> <td>    0.057</td> <td>    0.955</td> <td> 0.340</td> <td>   -0.057</td> <td>    0.166</td>
</tr>
<tr>
  <th>noise_79_impact_code</th> <td>    0.0162</td> <td>    0.058</td> <td>    0.280</td> <td> 0.780</td> <td>   -0.097</td> <td>    0.129</td>
</tr>
<tr>
  <th>noise_80_impact_code</th> <td>   -0.0590</td> <td>    0.061</td> <td>   -0.963</td> <td> 0.336</td> <td>   -0.179</td> <td>    0.061</td>
</tr>
<tr>
  <th>noise_81_impact_code</th> <td>    0.0500</td> <td>    0.056</td> <td>    0.887</td> <td> 0.375</td> <td>   -0.060</td> <td>    0.160</td>
</tr>
<tr>
  <th>noise_82_impact_code</th> <td>   -0.0614</td> <td>    0.056</td> <td>   -1.090</td> <td> 0.276</td> <td>   -0.172</td> <td>    0.049</td>
</tr>
<tr>
  <th>noise_83_impact_code</th> <td>    0.0776</td> <td>    0.055</td> <td>    1.403</td> <td> 0.161</td> <td>   -0.031</td> <td>    0.186</td>
</tr>
<tr>
  <th>noise_84_impact_code</th> <td>    0.0649</td> <td>    0.059</td> <td>    1.108</td> <td> 0.268</td> <td>   -0.050</td> <td>    0.180</td>
</tr>
<tr>
  <th>noise_85_impact_code</th> <td>   -0.0900</td> <td>    0.061</td> <td>   -1.467</td> <td> 0.143</td> <td>   -0.210</td> <td>    0.030</td>
</tr>
<tr>
  <th>noise_86_impact_code</th> <td>    0.0564</td> <td>    0.058</td> <td>    0.976</td> <td> 0.329</td> <td>   -0.057</td> <td>    0.170</td>
</tr>
<tr>
  <th>noise_87_impact_code</th> <td>    0.1159</td> <td>    0.053</td> <td>    2.190</td> <td> 0.029</td> <td>    0.012</td> <td>    0.220</td>
</tr>
<tr>
  <th>noise_88_impact_code</th> <td>    0.0868</td> <td>    0.057</td> <td>    1.533</td> <td> 0.125</td> <td>   -0.024</td> <td>    0.198</td>
</tr>
<tr>
  <th>noise_89_impact_code</th> <td>   -0.0855</td> <td>    0.058</td> <td>   -1.462</td> <td> 0.144</td> <td>   -0.200</td> <td>    0.029</td>
</tr>
<tr>
  <th>noise_90_impact_code</th> <td>   -0.1549</td> <td>    0.062</td> <td>   -2.505</td> <td> 0.012</td> <td>   -0.276</td> <td>   -0.034</td>
</tr>
<tr>
  <th>noise_91_impact_code</th> <td>    0.0358</td> <td>    0.053</td> <td>    0.670</td> <td> 0.503</td> <td>   -0.069</td> <td>    0.141</td>
</tr>
<tr>
  <th>noise_92_impact_code</th> <td>   -0.0938</td> <td>    0.058</td> <td>   -1.614</td> <td> 0.107</td> <td>   -0.208</td> <td>    0.020</td>
</tr>
<tr>
  <th>noise_93_impact_code</th> <td>    0.1404</td> <td>    0.053</td> <td>    2.664</td> <td> 0.008</td> <td>    0.037</td> <td>    0.244</td>
</tr>
<tr>
  <th>noise_94_impact_code</th> <td>   -0.0052</td> <td>    0.058</td> <td>   -0.089</td> <td> 0.929</td> <td>   -0.119</td> <td>    0.109</td>
</tr>
<tr>
  <th>noise_95_impact_code</th> <td>   -0.0151</td> <td>    0.056</td> <td>   -0.267</td> <td> 0.790</td> <td>   -0.126</td> <td>    0.096</td>
</tr>
<tr>
  <th>noise_96_impact_code</th> <td>    0.0114</td> <td>    0.055</td> <td>    0.207</td> <td> 0.836</td> <td>   -0.096</td> <td>    0.119</td>
</tr>
<tr>
  <th>noise_97_impact_code</th> <td>    0.0623</td> <td>    0.055</td> <td>    1.132</td> <td> 0.258</td> <td>   -0.046</td> <td>    0.170</td>
</tr>
<tr>
  <th>noise_98_impact_code</th> <td>   -0.0212</td> <td>    0.055</td> <td>   -0.383</td> <td> 0.702</td> <td>   -0.130</td> <td>    0.087</td>
</tr>
<tr>
  <th>noise_99_impact_code</th> <td>    0.0558</td> <td>    0.058</td> <td>    0.966</td> <td> 0.334</td> <td>   -0.057</td> <td>    0.169</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td> 4.778</td> <th>  Durbin-Watson:     </th> <td>   1.952</td>
</tr>
<tr>
  <th>Prob(Omnibus):</th> <td> 0.092</td> <th>  Jarque-Bera (JB):  </th> <td>   4.760</td>
</tr>
<tr>
  <th>Skew:</th>          <td> 0.089</td> <th>  Prob(JB):          </th> <td>  0.0926</td>
</tr>
<tr>
  <th>Kurtosis:</th>      <td> 2.885</td> <th>  Cond. No.          </th> <td>    7.77</td>
</tr>
</table><br/><br/>Warnings:<br/>[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.




```python
sklearn.metrics.r2_score(plot_train["y"],plot_train["predict_cross_all_vars"])
```




    0.29081862513775736




```python
sklearn.metrics.r2_score(plot_test["y"],plot_test["predict_cross_all_vars"])
```




    0.278467325722243




```python
smf3 = statsmodels.api.OLS(
    y_train, 
    statsmodels.api.add_constant(cross_frame[recommended_vars])).fit()
smf3.summary()
```




<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>            <td>y</td>        <th>  R-squared:         </th> <td>   0.255</td> 
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th> <td>   0.252</td> 
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th> <td>   71.86</td> 
</tr>
<tr>
  <th>Date:</th>             <td>Fri, 26 Jul 2019</td> <th>  Prob (F-statistic):</th> <td>2.69e-151</td>
</tr>
<tr>
  <th>Time:</th>                 <td>06:56:36</td>     <th>  Log-Likelihood:    </th> <td> -10550.</td> 
</tr>
<tr>
  <th>No. Observations:</th>      <td>  2531</td>      <th>  AIC:               </th> <td>2.113e+04</td>
</tr>
<tr>
  <th>Df Residuals:</th>          <td>  2518</td>      <th>  BIC:               </th> <td>2.120e+04</td>
</tr>
<tr>
  <th>Df Model:</th>              <td>    12</td>      <th>                     </th>     <td> </td>    
</tr>
<tr>
  <th>Covariance Type:</th>      <td>nonrobust</td>    <th>                     </th>     <td> </td>    
</tr>
</table>
<table class="simpletable">
<tr>
            <td></td>              <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>const</th>                <td>    1.4013</td> <td>    0.312</td> <td>    4.494</td> <td> 0.000</td> <td>    0.790</td> <td>    2.013</td>
</tr>
<tr>
  <th>var_0_impact_code</th>    <td>    0.4064</td> <td>    0.049</td> <td>    8.328</td> <td> 0.000</td> <td>    0.311</td> <td>    0.502</td>
</tr>
<tr>
  <th>var_1_impact_code</th>    <td>    0.3970</td> <td>    0.045</td> <td>    8.918</td> <td> 0.000</td> <td>    0.310</td> <td>    0.484</td>
</tr>
<tr>
  <th>var_2_impact_code</th>    <td>    0.4142</td> <td>    0.046</td> <td>    9.048</td> <td> 0.000</td> <td>    0.324</td> <td>    0.504</td>
</tr>
<tr>
  <th>var_3_impact_code</th>    <td>    0.3878</td> <td>    0.048</td> <td>    8.035</td> <td> 0.000</td> <td>    0.293</td> <td>    0.482</td>
</tr>
<tr>
  <th>var_4_impact_code</th>    <td>    0.3827</td> <td>    0.047</td> <td>    8.140</td> <td> 0.000</td> <td>    0.290</td> <td>    0.475</td>
</tr>
<tr>
  <th>var_5_impact_code</th>    <td>    0.4875</td> <td>    0.044</td> <td>   11.186</td> <td> 0.000</td> <td>    0.402</td> <td>    0.573</td>
</tr>
<tr>
  <th>var_6_impact_code</th>    <td>    0.4440</td> <td>    0.045</td> <td>    9.953</td> <td> 0.000</td> <td>    0.357</td> <td>    0.531</td>
</tr>
<tr>
  <th>var_7_impact_code</th>    <td>    0.4238</td> <td>    0.043</td> <td>    9.896</td> <td> 0.000</td> <td>    0.340</td> <td>    0.508</td>
</tr>
<tr>
  <th>var_8_impact_code</th>    <td>    0.3859</td> <td>    0.047</td> <td>    8.130</td> <td> 0.000</td> <td>    0.293</td> <td>    0.479</td>
</tr>
<tr>
  <th>var_9_impact_code</th>    <td>    0.3795</td> <td>    0.047</td> <td>    8.102</td> <td> 0.000</td> <td>    0.288</td> <td>    0.471</td>
</tr>
<tr>
  <th>noise_47_impact_code</th> <td>    0.1028</td> <td>    0.055</td> <td>    1.853</td> <td> 0.064</td> <td>   -0.006</td> <td>    0.212</td>
</tr>
<tr>
  <th>noise_93_impact_code</th> <td>    0.1541</td> <td>    0.052</td> <td>    2.955</td> <td> 0.003</td> <td>    0.052</td> <td>    0.256</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td> 4.497</td> <th>  Durbin-Watson:     </th> <td>   1.922</td>
</tr>
<tr>
  <th>Prob(Omnibus):</th> <td> 0.106</td> <th>  Jarque-Bera (JB):  </th> <td>   4.364</td>
</tr>
<tr>
  <th>Skew:</th>          <td> 0.074</td> <th>  Prob(JB):          </th> <td>   0.113</td>
</tr>
<tr>
  <th>Kurtosis:</th>      <td> 2.860</td> <th>  Cond. No.          </th> <td>    7.43</td>
</tr>
</table><br/><br/>Warnings:<br/>[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.




```python
sklearn.metrics.r2_score(plot_train["y"],plot_train["predict_cross_recommended_vars"])
```




    0.25510947375181725




```python
sklearn.metrics.r2_score(plot_test["y"],plot_test["predict_cross_recommended_vars"])
```




    0.30343110262504314




```python

```
