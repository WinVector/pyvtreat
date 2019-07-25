

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
n_noise_variables = 20
n_levels = 1000
```


```python
d = pandas.DataFrame({"y":numpy.random.normal(size = n_rows)})
```


```python
def mk_var_values(n_levels):
    values = {}
    for i in range(n_levels):
        values["level_" + str(i)] = numpy.random.normal(size=1)[0]
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
      <th>noise_10</th>
      <th>noise_11</th>
      <th>noise_12</th>
      <th>noise_13</th>
      <th>noise_14</th>
      <th>noise_15</th>
      <th>noise_16</th>
      <th>noise_17</th>
      <th>noise_18</th>
      <th>noise_19</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-1.512479</td>
      <td>level_62</td>
      <td>level_622</td>
      <td>level_687</td>
      <td>level_818</td>
      <td>level_99</td>
      <td>level_270</td>
      <td>level_223</td>
      <td>level_108</td>
      <td>level_812</td>
      <td>...</td>
      <td>level_964</td>
      <td>level_462</td>
      <td>level_17</td>
      <td>level_249</td>
      <td>level_633</td>
      <td>level_502</td>
      <td>level_874</td>
      <td>level_98</td>
      <td>level_788</td>
      <td>level_379</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-1.275709</td>
      <td>level_219</td>
      <td>level_882</td>
      <td>level_894</td>
      <td>level_137</td>
      <td>level_192</td>
      <td>level_775</td>
      <td>level_808</td>
      <td>level_196</td>
      <td>level_716</td>
      <td>...</td>
      <td>level_112</td>
      <td>level_126</td>
      <td>level_72</td>
      <td>level_315</td>
      <td>level_492</td>
      <td>level_857</td>
      <td>level_887</td>
      <td>level_16</td>
      <td>level_129</td>
      <td>level_845</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-2.053840</td>
      <td>level_557</td>
      <td>level_394</td>
      <td>level_236</td>
      <td>level_451</td>
      <td>level_629</td>
      <td>level_939</td>
      <td>level_287</td>
      <td>level_799</td>
      <td>level_162</td>
      <td>...</td>
      <td>level_6</td>
      <td>level_749</td>
      <td>level_156</td>
      <td>level_182</td>
      <td>level_773</td>
      <td>level_34</td>
      <td>level_474</td>
      <td>level_369</td>
      <td>level_72</td>
      <td>level_38</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-3.870142</td>
      <td>level_983</td>
      <td>level_469</td>
      <td>level_644</td>
      <td>level_896</td>
      <td>level_4</td>
      <td>level_553</td>
      <td>level_193</td>
      <td>level_231</td>
      <td>level_700</td>
      <td>...</td>
      <td>level_930</td>
      <td>level_206</td>
      <td>level_8</td>
      <td>level_655</td>
      <td>level_535</td>
      <td>level_145</td>
      <td>level_163</td>
      <td>level_253</td>
      <td>level_76</td>
      <td>level_874</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-1.878621</td>
      <td>level_733</td>
      <td>level_347</td>
      <td>level_563</td>
      <td>level_895</td>
      <td>level_560</td>
      <td>level_306</td>
      <td>level_717</td>
      <td>level_816</td>
      <td>level_297</td>
      <td>...</td>
      <td>level_10</td>
      <td>level_668</td>
      <td>level_752</td>
      <td>level_341</td>
      <td>level_875</td>
      <td>level_720</td>
      <td>level_722</td>
      <td>level_500</td>
      <td>level_217</td>
      <td>level_205</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 31 columns</p>
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
      <th>noise_10_impact_code</th>
      <th>noise_11_impact_code</th>
      <th>noise_12_impact_code</th>
      <th>noise_13_impact_code</th>
      <th>noise_14_impact_code</th>
      <th>noise_15_impact_code</th>
      <th>noise_16_impact_code</th>
      <th>noise_17_impact_code</th>
      <th>noise_18_impact_code</th>
      <th>noise_19_impact_code</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-2.381684</td>
      <td>-2.104251</td>
      <td>-4.765004</td>
      <td>-0.252371</td>
      <td>3.598255</td>
      <td>0.099845</td>
      <td>-2.034106</td>
      <td>-0.899741</td>
      <td>-0.038524</td>
      <td>1.763707</td>
      <td>...</td>
      <td>1.612743</td>
      <td>-0.901466</td>
      <td>0.485914</td>
      <td>-0.049948</td>
      <td>-0.912003</td>
      <td>-0.202120</td>
      <td>0.898147</td>
      <td>-1.536223</td>
      <td>-0.747063</td>
      <td>-2.543203</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-1.399175</td>
      <td>-1.939733</td>
      <td>-0.698308</td>
      <td>0.194228</td>
      <td>-2.077583</td>
      <td>0.170268</td>
      <td>-1.196393</td>
      <td>0.564307</td>
      <td>-2.077583</td>
      <td>-0.309568</td>
      <td>...</td>
      <td>0.403824</td>
      <td>-1.389051</td>
      <td>0.130655</td>
      <td>1.388417</td>
      <td>-0.508354</td>
      <td>-1.144589</td>
      <td>-1.754423</td>
      <td>-1.589817</td>
      <td>0.388164</td>
      <td>-2.146584</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-1.417156</td>
      <td>-1.514796</td>
      <td>-3.893886</td>
      <td>-0.249653</td>
      <td>-3.624847</td>
      <td>-2.176076</td>
      <td>-0.317911</td>
      <td>0.503130</td>
      <td>-0.435899</td>
      <td>-2.439520</td>
      <td>...</td>
      <td>-3.893886</td>
      <td>0.101694</td>
      <td>-1.479740</td>
      <td>-1.491016</td>
      <td>-3.893886</td>
      <td>-2.031196</td>
      <td>0.188012</td>
      <td>-3.893886</td>
      <td>-2.144517</td>
      <td>-3.893886</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-0.858530</td>
      <td>-1.974885</td>
      <td>-2.999218</td>
      <td>-1.902365</td>
      <td>1.375042</td>
      <td>-1.590862</td>
      <td>-1.588066</td>
      <td>-0.357176</td>
      <td>-4.601834</td>
      <td>1.458501</td>
      <td>...</td>
      <td>-2.788086</td>
      <td>1.160561</td>
      <td>1.155715</td>
      <td>-0.138130</td>
      <td>-1.902365</td>
      <td>-0.110150</td>
      <td>0.338963</td>
      <td>-0.250018</td>
      <td>1.715388</td>
      <td>0.415228</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-2.035177</td>
      <td>-3.035073</td>
      <td>-1.998686</td>
      <td>-0.757349</td>
      <td>-1.416043</td>
      <td>-2.952257</td>
      <td>-2.649014</td>
      <td>-1.626104</td>
      <td>-3.189519</td>
      <td>-2.952257</td>
      <td>...</td>
      <td>1.488838</td>
      <td>-0.586472</td>
      <td>-2.592636</td>
      <td>-4.707346</td>
      <td>-3.194351</td>
      <td>-2.584087</td>
      <td>-3.494238</td>
      <td>-2.952257</td>
      <td>-2.290325</td>
      <td>-2.240008</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 30 columns</p>
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
      <th>noise_10_impact_code</th>
      <th>noise_11_impact_code</th>
      <th>noise_12_impact_code</th>
      <th>noise_13_impact_code</th>
      <th>noise_14_impact_code</th>
      <th>noise_15_impact_code</th>
      <th>noise_16_impact_code</th>
      <th>noise_17_impact_code</th>
      <th>noise_18_impact_code</th>
      <th>noise_19_impact_code</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-1.909948</td>
      <td>-0.947676</td>
      <td>-3.326575e+00</td>
      <td>-2.262960e-01</td>
      <td>1.253636e+00</td>
      <td>5.112139e-02</td>
      <td>-1.972252</td>
      <td>-0.811711</td>
      <td>-0.028256</td>
      <td>0.589089</td>
      <td>...</td>
      <td>9.921386e-01</td>
      <td>-0.440438</td>
      <td>0.414720</td>
      <td>-0.025565</td>
      <td>-8.881952e-01</td>
      <td>-0.143835</td>
      <td>0.465830</td>
      <td>0.000000</td>
      <td>-0.604520</td>
      <td>-1.863388</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-1.252605</td>
      <td>-1.894818</td>
      <td>-5.684474e-01</td>
      <td>1.535153e-01</td>
      <td>-3.469447e-18</td>
      <td>1.223043e-01</td>
      <td>-0.903133</td>
      <td>0.404011</td>
      <td>0.000000</td>
      <td>-0.213788</td>
      <td>...</td>
      <td>1.557779e-01</td>
      <td>-0.903957</td>
      <td>0.046638</td>
      <td>0.745954</td>
      <td>-2.732874e-01</td>
      <td>-0.910694</td>
      <td>-1.595830</td>
      <td>-1.333192</td>
      <td>0.294616</td>
      <td>-1.140305</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-0.774806</td>
      <td>-0.534919</td>
      <td>-3.469447e-18</td>
      <td>-7.342844e-02</td>
      <td>-2.253121e+00</td>
      <td>-9.787758e-01</td>
      <td>-0.061632</td>
      <td>0.213264</td>
      <td>-0.136761</td>
      <td>-1.690215</td>
      <td>...</td>
      <td>3.469447e-18</td>
      <td>0.015921</td>
      <td>-1.235856</td>
      <td>-0.670184</td>
      <td>-3.469447e-18</td>
      <td>-1.901039</td>
      <td>0.084539</td>
      <td>0.000000</td>
      <td>-1.381208</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-0.564615</td>
      <td>-1.958810</td>
      <td>-2.554432e+00</td>
      <td>3.469447e-18</td>
      <td>1.069939e+00</td>
      <td>-1.537477e+00</td>
      <td>-1.348056</td>
      <td>-0.281568</td>
      <td>-2.187472</td>
      <td>0.815940</td>
      <td>...</td>
      <td>-1.631420e+00</td>
      <td>0.728091</td>
      <td>0.259439</td>
      <td>-0.081337</td>
      <td>-3.469447e-18</td>
      <td>-0.098083</td>
      <td>0.205748</td>
      <td>-0.183591</td>
      <td>1.163523</td>
      <td>0.266100</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-1.929813</td>
      <td>-0.915852</td>
      <td>-1.577363e+00</td>
      <td>-1.982328e-01</td>
      <td>-7.533978e-01</td>
      <td>3.469447e-18</td>
      <td>-2.571700</td>
      <td>-1.406151</td>
      <td>-3.128655</td>
      <td>0.000000</td>
      <td>...</td>
      <td>4.852605e-01</td>
      <td>-0.276976</td>
      <td>-2.473991</td>
      <td>-2.199724</td>
      <td>-2.347868e+00</td>
      <td>-2.208081</td>
      <td>-2.851071</td>
      <td>0.000000</td>
      <td>-1.731833</td>
      <td>-1.894122</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 30 columns</p>
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
      <th>noise_10_impact_code</th>
      <th>noise_11_impact_code</th>
      <th>noise_12_impact_code</th>
      <th>noise_13_impact_code</th>
      <th>noise_14_impact_code</th>
      <th>noise_15_impact_code</th>
      <th>noise_16_impact_code</th>
      <th>noise_17_impact_code</th>
      <th>noise_18_impact_code</th>
      <th>noise_19_impact_code</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.000000</td>
      <td>-4.410653e-01</td>
      <td>-3.802091e+00</td>
      <td>3.785141e-01</td>
      <td>0.000000e+00</td>
      <td>1.734723e-18</td>
      <td>-2.618627</td>
      <td>-4.462575e-01</td>
      <td>2.253484e-01</td>
      <td>0.000000e+00</td>
      <td>...</td>
      <td>-1.734723e-18</td>
      <td>-0.127851</td>
      <td>1.041374e+00</td>
      <td>0.165078</td>
      <td>-0.587222</td>
      <td>-0.261306</td>
      <td>4.885900e-01</td>
      <td>0.000000</td>
      <td>1.734723e-18</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-0.379997</td>
      <td>-1.797717e+00</td>
      <td>-3.187352e-01</td>
      <td>3.050309e-01</td>
      <td>0.000000e+00</td>
      <td>1.734723e-18</td>
      <td>0.222960</td>
      <td>1.734723e-18</td>
      <td>0.000000e+00</td>
      <td>1.700485e-01</td>
      <td>...</td>
      <td>-2.151714e+00</td>
      <td>0.109171</td>
      <td>-1.734723e-18</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.448745</td>
      <td>1.734723e-18</td>
      <td>-0.826536</td>
      <td>5.321886e-01</td>
      <td>-0.256076</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.000000</td>
      <td>-8.673617e-19</td>
      <td>0.000000e+00</td>
      <td>-8.673617e-19</td>
      <td>-1.895280e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000</td>
      <td>1.943479e+00</td>
      <td>2.284351e-01</td>
      <td>-8.673617e-19</td>
      <td>...</td>
      <td>0.000000e+00</td>
      <td>0.000000</td>
      <td>0.000000e+00</td>
      <td>-0.220404</td>
      <td>0.000000</td>
      <td>-1.695084</td>
      <td>1.896790e+00</td>
      <td>0.000000</td>
      <td>-8.673617e-19</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-0.344378</td>
      <td>0.000000e+00</td>
      <td>-2.578087e+00</td>
      <td>0.000000e+00</td>
      <td>1.765915e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000</td>
      <td>-1.193337e-01</td>
      <td>3.469447e-18</td>
      <td>2.831001e+00</td>
      <td>...</td>
      <td>-1.273807e+00</td>
      <td>2.738213</td>
      <td>0.000000e+00</td>
      <td>-0.597742</td>
      <td>0.000000</td>
      <td>0.358807</td>
      <td>-3.469447e-18</td>
      <td>0.229794</td>
      <td>1.686652e+00</td>
      <td>0.608921</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-1.592341</td>
      <td>-3.407191e-01</td>
      <td>-6.938894e-18</td>
      <td>0.000000e+00</td>
      <td>6.938894e-18</td>
      <td>0.000000e+00</td>
      <td>0.000000</td>
      <td>-1.828641e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>...</td>
      <td>1.025920e+00</td>
      <td>0.158180</td>
      <td>0.000000e+00</td>
      <td>0.000000</td>
      <td>-2.087549</td>
      <td>-1.452941</td>
      <td>0.000000e+00</td>
      <td>0.000000</td>
      <td>-9.940506e-01</td>
      <td>0.000000</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 30 columns</p>
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
      <td>0.644788</td>
      <td>0.544526</td>
      <td>0.092948</td>
      <td>0.124266</td>
      <td>False</td>
    </tr>
    <tr>
      <th>1</th>
      <td>var_1_impact_code</td>
      <td>0.661523</td>
      <td>0.563951</td>
      <td>0.131639</td>
      <td>0.088187</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2</th>
      <td>var_2_impact_code</td>
      <td>0.649108</td>
      <td>0.552370</td>
      <td>0.090711</td>
      <td>0.107474</td>
      <td>False</td>
    </tr>
    <tr>
      <th>3</th>
      <td>var_3_impact_code</td>
      <td>0.661329</td>
      <td>0.563282</td>
      <td>0.113515</td>
      <td>0.133576</td>
      <td>False</td>
    </tr>
    <tr>
      <th>4</th>
      <td>var_4_impact_code</td>
      <td>0.642346</td>
      <td>0.565297</td>
      <td>0.097320</td>
      <td>0.093026</td>
      <td>False</td>
    </tr>
    <tr>
      <th>5</th>
      <td>var_5_impact_code</td>
      <td>0.626977</td>
      <td>0.530813</td>
      <td>0.119348</td>
      <td>0.095392</td>
      <td>False</td>
    </tr>
    <tr>
      <th>6</th>
      <td>var_6_impact_code</td>
      <td>0.660658</td>
      <td>0.559396</td>
      <td>0.134944</td>
      <td>0.096875</td>
      <td>False</td>
    </tr>
    <tr>
      <th>7</th>
      <td>var_7_impact_code</td>
      <td>0.642193</td>
      <td>0.541202</td>
      <td>0.084799</td>
      <td>0.098976</td>
      <td>False</td>
    </tr>
    <tr>
      <th>8</th>
      <td>var_8_impact_code</td>
      <td>0.659190</td>
      <td>0.575860</td>
      <td>0.136418</td>
      <td>0.117148</td>
      <td>False</td>
    </tr>
    <tr>
      <th>9</th>
      <td>var_9_impact_code</td>
      <td>0.634396</td>
      <td>0.536409</td>
      <td>0.054540</td>
      <td>0.115450</td>
      <td>False</td>
    </tr>
    <tr>
      <th>10</th>
      <td>noise_0_impact_code</td>
      <td>0.604909</td>
      <td>0.485920</td>
      <td>0.005627</td>
      <td>0.004318</td>
      <td>True</td>
    </tr>
    <tr>
      <th>11</th>
      <td>noise_1_impact_code</td>
      <td>0.617102</td>
      <td>0.512492</td>
      <td>0.044977</td>
      <td>0.037706</td>
      <td>True</td>
    </tr>
    <tr>
      <th>12</th>
      <td>noise_2_impact_code</td>
      <td>0.615657</td>
      <td>0.512926</td>
      <td>-0.002675</td>
      <td>0.000393</td>
      <td>True</td>
    </tr>
    <tr>
      <th>13</th>
      <td>noise_3_impact_code</td>
      <td>0.608467</td>
      <td>0.489610</td>
      <td>0.020148</td>
      <td>-0.023784</td>
      <td>True</td>
    </tr>
    <tr>
      <th>14</th>
      <td>noise_4_impact_code</td>
      <td>0.595733</td>
      <td>0.487942</td>
      <td>0.009957</td>
      <td>-0.028563</td>
      <td>True</td>
    </tr>
    <tr>
      <th>15</th>
      <td>noise_5_impact_code</td>
      <td>0.605024</td>
      <td>0.499236</td>
      <td>0.005567</td>
      <td>0.009298</td>
      <td>True</td>
    </tr>
    <tr>
      <th>16</th>
      <td>noise_6_impact_code</td>
      <td>0.603284</td>
      <td>0.501242</td>
      <td>0.032507</td>
      <td>-0.013591</td>
      <td>True</td>
    </tr>
    <tr>
      <th>17</th>
      <td>noise_7_impact_code</td>
      <td>0.574083</td>
      <td>0.454062</td>
      <td>-0.020388</td>
      <td>-0.001293</td>
      <td>True</td>
    </tr>
    <tr>
      <th>18</th>
      <td>noise_8_impact_code</td>
      <td>0.608494</td>
      <td>0.501292</td>
      <td>-0.004978</td>
      <td>0.001596</td>
      <td>True</td>
    </tr>
    <tr>
      <th>19</th>
      <td>noise_9_impact_code</td>
      <td>0.612545</td>
      <td>0.502883</td>
      <td>-0.035834</td>
      <td>0.002772</td>
      <td>True</td>
    </tr>
    <tr>
      <th>20</th>
      <td>noise_10_impact_code</td>
      <td>0.604721</td>
      <td>0.484834</td>
      <td>-0.023485</td>
      <td>-0.005252</td>
      <td>True</td>
    </tr>
    <tr>
      <th>21</th>
      <td>noise_11_impact_code</td>
      <td>0.615790</td>
      <td>0.494956</td>
      <td>-0.013121</td>
      <td>-0.016374</td>
      <td>True</td>
    </tr>
    <tr>
      <th>22</th>
      <td>noise_12_impact_code</td>
      <td>0.600669</td>
      <td>0.493000</td>
      <td>-0.002599</td>
      <td>0.005426</td>
      <td>True</td>
    </tr>
    <tr>
      <th>23</th>
      <td>noise_13_impact_code</td>
      <td>0.612948</td>
      <td>0.511174</td>
      <td>0.039950</td>
      <td>0.019232</td>
      <td>True</td>
    </tr>
    <tr>
      <th>24</th>
      <td>noise_14_impact_code</td>
      <td>0.615691</td>
      <td>0.502706</td>
      <td>-0.010617</td>
      <td>-0.024321</td>
      <td>True</td>
    </tr>
    <tr>
      <th>25</th>
      <td>noise_15_impact_code</td>
      <td>0.625304</td>
      <td>0.521038</td>
      <td>-0.005396</td>
      <td>-0.004469</td>
      <td>True</td>
    </tr>
    <tr>
      <th>26</th>
      <td>noise_16_impact_code</td>
      <td>0.605771</td>
      <td>0.499178</td>
      <td>0.024809</td>
      <td>-0.001361</td>
      <td>True</td>
    </tr>
    <tr>
      <th>27</th>
      <td>noise_17_impact_code</td>
      <td>0.605892</td>
      <td>0.505174</td>
      <td>-0.002311</td>
      <td>-0.021745</td>
      <td>True</td>
    </tr>
    <tr>
      <th>28</th>
      <td>noise_18_impact_code</td>
      <td>0.606938</td>
      <td>0.510770</td>
      <td>0.008704</td>
      <td>-0.027868</td>
      <td>True</td>
    </tr>
    <tr>
      <th>29</th>
      <td>noise_19_impact_code</td>
      <td>0.619884</td>
      <td>0.518057</td>
      <td>0.012430</td>
      <td>0.016912</td>
      <td>True</td>
    </tr>
  </tbody>
</table>
</div>




```python
print(scipy.stats.pearsonr(corr_frame['naive_train_empirical_correlation'], corr_frame['test_correlation']))
seaborn.scatterplot(x = "naive_train_empirical_correlation", y = "test_correlation", data = corr_frame, hue = "is_noise")
matplotlib.pyplot.plot([-1, 1], [-1, 1], color="red")
matplotlib.pyplot.xlim(-.2,1)
matplotlib.pyplot.ylim(-.2,1)
matplotlib.pyplot.gca().set_aspect('equal', adjustable='box')
```

    (0.8586098467356158, 1.2887439554616704e-09)



![png](output_18_1.png)



```python
print(scipy.stats.pearsonr(corr_frame['naive_train_hierarchical_correlation'], corr_frame['test_correlation']))
seaborn.scatterplot(x = "naive_train_hierarchical_correlation", y = "test_correlation", data = corr_frame, hue = "is_noise")
matplotlib.pyplot.plot([-1, 1], [-1, 1], color="red")
matplotlib.pyplot.xlim(-.2,1)
matplotlib.pyplot.ylim(-.2,1)
matplotlib.pyplot.gca().set_aspect('equal', adjustable='box')
```

    (0.8578727344570392, 1.3794662871152999e-09)



![png](output_19_1.png)



```python
print(scipy.stats.pearsonr(corr_frame['cross_frame_correlation'], corr_frame['test_correlation']))
seaborn.scatterplot(x = "cross_frame_correlation", y = "test_correlation", data = corr_frame,  hue = "is_noise")
matplotlib.pyplot.plot([-1, 1], [-1, 1], color="red")
matplotlib.pyplot.xlim(-.2,1)
matplotlib.pyplot.ylim(-.2,1)
matplotlib.pyplot.gca().set_aspect('equal', adjustable='box')
```

    (0.8835126008896232, 1.0040423087447391e-10)



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
      <th>25</th>
      <td>noise_15_impact_code</td>
      <td>impact_code</td>
      <td>True</td>
      <td>True</td>
      <td>-0.005396</td>
      <td>0.786317</td>
      <td>30.0</td>
      <td>False</td>
    </tr>
    <tr>
      <th>26</th>
      <td>noise_16_impact_code</td>
      <td>impact_code</td>
      <td>True</td>
      <td>True</td>
      <td>0.024809</td>
      <td>0.212503</td>
      <td>30.0</td>
      <td>False</td>
    </tr>
    <tr>
      <th>27</th>
      <td>noise_17_impact_code</td>
      <td>impact_code</td>
      <td>True</td>
      <td>True</td>
      <td>-0.002311</td>
      <td>0.907565</td>
      <td>30.0</td>
      <td>False</td>
    </tr>
    <tr>
      <th>28</th>
      <td>noise_18_impact_code</td>
      <td>impact_code</td>
      <td>True</td>
      <td>True</td>
      <td>0.008704</td>
      <td>0.661875</td>
      <td>30.0</td>
      <td>False</td>
    </tr>
    <tr>
      <th>29</th>
      <td>noise_19_impact_code</td>
      <td>impact_code</td>
      <td>True</td>
      <td>True</td>
      <td>0.012430</td>
      <td>0.532271</td>
      <td>30.0</td>
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
     'noise_1_impact_code']




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
      <td>-1.275709</td>
      <td>-0.571424</td>
      <td>-0.625263</td>
      <td>-0.344465</td>
      <td>-0.834632</td>
    </tr>
    <tr>
      <th>1</th>
      <td>3.229611</td>
      <td>0.851995</td>
      <td>0.922912</td>
      <td>0.627789</td>
      <td>1.143239</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-1.110627</td>
      <td>-1.454640</td>
      <td>-1.529919</td>
      <td>-0.545726</td>
      <td>-0.741694</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-3.574484</td>
      <td>-0.030518</td>
      <td>0.066862</td>
      <td>-0.467567</td>
      <td>-0.760004</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5.308181</td>
      <td>1.446994</td>
      <td>1.590186</td>
      <td>0.428473</td>
      <td>0.726234</td>
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

    0.7140043412841672



![png](output_33_1.png)



```python
print(rmse(plot_train["predict_naive_hierarchical_all_vars"], plot_train["y"]))
seaborn.scatterplot(x="predict_naive_hierarchical_all_vars", y ="y", data = plot_train)
plt = matplotlib.pyplot.title("Naive hierarchical prediction on train")
```

    1.1472072730966285



![png](output_34_1.png)



```python
print(rmse(plot_train["predict_cross_all_vars"], plot_train["y"]))
seaborn.scatterplot(x="predict_cross_all_vars", y ="y", data = plot_train)
plt = matplotlib.pyplot.title("Cross prediction on (all vars) train")
```

    3.0988752035926592



![png](output_35_1.png)



```python
print(rmse(plot_train["predict_cross_recommended_vars"], plot_train["y"]))
seaborn.scatterplot(x="predict_cross_recommended_vars", y ="y", data = plot_train)
plt = matplotlib.pyplot.title("Cross prediction (recommended vars) on train")
```

    3.1081115173514657



![png](output_36_1.png)



```python
print(rmse(plot_test["predict_naive_empirical_all_vars"], plot_test["y"]))
seaborn.scatterplot(x="predict_naive_empirical_all_vars", y ="y", data = plot_test)
plt = matplotlib.pyplot.title("Naive empirical prediction on test")
```

    3.2394949260318953



![png](output_37_1.png)



```python
print(rmse(plot_test["predict_naive_hierarchical_all_vars"], plot_test["y"]))
seaborn.scatterplot(x="predict_naive_hierarchical_all_vars", y ="y", data = plot_test)
plt = matplotlib.pyplot.title("Naive hierarchical prediction on test")
```

    3.22983740774229



![png](output_38_1.png)



```python
print(rmse(plot_test["predict_cross_all_vars"], plot_test["y"]))
seaborn.scatterplot(x="predict_cross_all_vars", y ="y", data = plot_test)
plt = matplotlib.pyplot.title("Cross prediction on (all vars) test")
```

    3.1421760904178466



![png](output_39_1.png)



```python
print(rmse(plot_test["predict_cross_recommended_vars"], plot_test["y"]))
seaborn.scatterplot(x="predict_cross_recommended_vars", y ="y", data = plot_test)
plt = matplotlib.pyplot.title("Cross prediction (recommended vars) on test")
```

    3.132116151044606



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
  <th>Dep. Variable:</th>            <td>y</td>        <th>  R-squared:         </th> <td>   0.953</td>
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th> <td>   0.953</td>
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th> <td>   1701.</td>
</tr>
<tr>
  <th>Date:</th>             <td>Thu, 25 Jul 2019</td> <th>  Prob (F-statistic):</th>  <td>  0.00</td> 
</tr>
<tr>
  <th>Time:</th>                 <td>15:49:06</td>     <th>  Log-Likelihood:    </th> <td> -2734.4</td>
</tr>
<tr>
  <th>No. Observations:</th>      <td>  2527</td>      <th>  AIC:               </th> <td>   5531.</td>
</tr>
<tr>
  <th>Df Residuals:</th>          <td>  2496</td>      <th>  BIC:               </th> <td>   5712.</td>
</tr>
<tr>
  <th>Df Model:</th>              <td>    30</td>      <th>                     </th>     <td> </td>   
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
  <th>const</th>                <td>    0.0237</td> <td>    0.014</td> <td>    1.661</td> <td> 0.097</td> <td>   -0.004</td> <td>    0.052</td>
</tr>
<tr>
  <th>var_0_impact_code</th>    <td>    0.0780</td> <td>    0.009</td> <td>    8.956</td> <td> 0.000</td> <td>    0.061</td> <td>    0.095</td>
</tr>
<tr>
  <th>var_1_impact_code</th>    <td>    0.1058</td> <td>    0.009</td> <td>   12.393</td> <td> 0.000</td> <td>    0.089</td> <td>    0.123</td>
</tr>
<tr>
  <th>var_2_impact_code</th>    <td>    0.0905</td> <td>    0.009</td> <td>   10.485</td> <td> 0.000</td> <td>    0.074</td> <td>    0.107</td>
</tr>
<tr>
  <th>var_3_impact_code</th>    <td>    0.0890</td> <td>    0.009</td> <td>   10.359</td> <td> 0.000</td> <td>    0.072</td> <td>    0.106</td>
</tr>
<tr>
  <th>var_4_impact_code</th>    <td>    0.0903</td> <td>    0.009</td> <td>   10.443</td> <td> 0.000</td> <td>    0.073</td> <td>    0.107</td>
</tr>
<tr>
  <th>var_5_impact_code</th>    <td>    0.0792</td> <td>    0.009</td> <td>    9.049</td> <td> 0.000</td> <td>    0.062</td> <td>    0.096</td>
</tr>
<tr>
  <th>var_6_impact_code</th>    <td>    0.0882</td> <td>    0.009</td> <td>   10.240</td> <td> 0.000</td> <td>    0.071</td> <td>    0.105</td>
</tr>
<tr>
  <th>var_7_impact_code</th>    <td>    0.0899</td> <td>    0.009</td> <td>   10.400</td> <td> 0.000</td> <td>    0.073</td> <td>    0.107</td>
</tr>
<tr>
  <th>var_8_impact_code</th>    <td>    0.1038</td> <td>    0.009</td> <td>   12.156</td> <td> 0.000</td> <td>    0.087</td> <td>    0.121</td>
</tr>
<tr>
  <th>var_9_impact_code</th>    <td>    0.0912</td> <td>    0.009</td> <td>   10.489</td> <td> 0.000</td> <td>    0.074</td> <td>    0.108</td>
</tr>
<tr>
  <th>noise_0_impact_code</th>  <td>    0.0829</td> <td>    0.009</td> <td>    9.346</td> <td> 0.000</td> <td>    0.065</td> <td>    0.100</td>
</tr>
<tr>
  <th>noise_1_impact_code</th>  <td>    0.0882</td> <td>    0.009</td> <td>   10.003</td> <td> 0.000</td> <td>    0.071</td> <td>    0.105</td>
</tr>
<tr>
  <th>noise_2_impact_code</th>  <td>    0.0734</td> <td>    0.009</td> <td>    8.298</td> <td> 0.000</td> <td>    0.056</td> <td>    0.091</td>
</tr>
<tr>
  <th>noise_3_impact_code</th>  <td>    0.0834</td> <td>    0.009</td> <td>    9.419</td> <td> 0.000</td> <td>    0.066</td> <td>    0.101</td>
</tr>
<tr>
  <th>noise_4_impact_code</th>  <td>    0.0526</td> <td>    0.009</td> <td>    5.817</td> <td> 0.000</td> <td>    0.035</td> <td>    0.070</td>
</tr>
<tr>
  <th>noise_5_impact_code</th>  <td>    0.0821</td> <td>    0.009</td> <td>    9.209</td> <td> 0.000</td> <td>    0.065</td> <td>    0.100</td>
</tr>
<tr>
  <th>noise_6_impact_code</th>  <td>    0.0830</td> <td>    0.009</td> <td>    9.355</td> <td> 0.000</td> <td>    0.066</td> <td>    0.100</td>
</tr>
<tr>
  <th>noise_7_impact_code</th>  <td>    0.0710</td> <td>    0.009</td> <td>    7.752</td> <td> 0.000</td> <td>    0.053</td> <td>    0.089</td>
</tr>
<tr>
  <th>noise_8_impact_code</th>  <td>    0.0556</td> <td>    0.009</td> <td>    6.222</td> <td> 0.000</td> <td>    0.038</td> <td>    0.073</td>
</tr>
<tr>
  <th>noise_9_impact_code</th>  <td>    0.0768</td> <td>    0.009</td> <td>    8.686</td> <td> 0.000</td> <td>    0.059</td> <td>    0.094</td>
</tr>
<tr>
  <th>noise_10_impact_code</th> <td>    0.0727</td> <td>    0.009</td> <td>    8.160</td> <td> 0.000</td> <td>    0.055</td> <td>    0.090</td>
</tr>
<tr>
  <th>noise_11_impact_code</th> <td>    0.0825</td> <td>    0.009</td> <td>    9.335</td> <td> 0.000</td> <td>    0.065</td> <td>    0.100</td>
</tr>
<tr>
  <th>noise_12_impact_code</th> <td>    0.0909</td> <td>    0.009</td> <td>   10.223</td> <td> 0.000</td> <td>    0.073</td> <td>    0.108</td>
</tr>
<tr>
  <th>noise_13_impact_code</th> <td>    0.0896</td> <td>    0.009</td> <td>   10.175</td> <td> 0.000</td> <td>    0.072</td> <td>    0.107</td>
</tr>
<tr>
  <th>noise_14_impact_code</th> <td>    0.0830</td> <td>    0.009</td> <td>    9.443</td> <td> 0.000</td> <td>    0.066</td> <td>    0.100</td>
</tr>
<tr>
  <th>noise_15_impact_code</th> <td>    0.0673</td> <td>    0.009</td> <td>    7.643</td> <td> 0.000</td> <td>    0.050</td> <td>    0.085</td>
</tr>
<tr>
  <th>noise_16_impact_code</th> <td>    0.0683</td> <td>    0.009</td> <td>    7.634</td> <td> 0.000</td> <td>    0.051</td> <td>    0.086</td>
</tr>
<tr>
  <th>noise_17_impact_code</th> <td>    0.0703</td> <td>    0.009</td> <td>    7.900</td> <td> 0.000</td> <td>    0.053</td> <td>    0.088</td>
</tr>
<tr>
  <th>noise_18_impact_code</th> <td>    0.0802</td> <td>    0.009</td> <td>    9.021</td> <td> 0.000</td> <td>    0.063</td> <td>    0.098</td>
</tr>
<tr>
  <th>noise_19_impact_code</th> <td>    0.0910</td> <td>    0.009</td> <td>   10.396</td> <td> 0.000</td> <td>    0.074</td> <td>    0.108</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td> 5.755</td> <th>  Durbin-Watson:     </th> <td>   2.012</td>
</tr>
<tr>
  <th>Prob(Omnibus):</th> <td> 0.056</td> <th>  Jarque-Bera (JB):  </th> <td>   6.402</td>
</tr>
<tr>
  <th>Skew:</th>          <td>-0.051</td> <th>  Prob(JB):          </th> <td>  0.0407</td>
</tr>
<tr>
  <th>Kurtosis:</th>      <td> 3.225</td> <th>  Cond. No.          </th> <td>    7.19</td>
</tr>
</table><br/><br/>Warnings:<br/>[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.




```python
sklearn.metrics.r2_score(plot_train["y"],plot_train["predict_naive_empirical_all_vars"])
```




    0.9533556878472528




```python
sklearn.metrics.r2_score(plot_test["y"],plot_test["predict_naive_empirical_all_vars"])
```




    0.04402895774168669




```python
smf2 = statsmodels.api.OLS(
    y_train, 
    statsmodels.api.add_constant(cross_frame[all_vars])).fit()
smf2.summary()
```




<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>            <td>y</td>        <th>  R-squared:         </th> <td>   0.121</td> 
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th> <td>   0.111</td> 
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th> <td>   11.49</td> 
</tr>
<tr>
  <th>Date:</th>             <td>Thu, 25 Jul 2019</td> <th>  Prob (F-statistic):</th> <td>3.36e-51</td> 
</tr>
<tr>
  <th>Time:</th>                 <td>15:49:06</td>     <th>  Log-Likelihood:    </th> <td> -6443.8</td> 
</tr>
<tr>
  <th>No. Observations:</th>      <td>  2527</td>      <th>  AIC:               </th> <td>1.295e+04</td>
</tr>
<tr>
  <th>Df Residuals:</th>          <td>  2496</td>      <th>  BIC:               </th> <td>1.313e+04</td>
</tr>
<tr>
  <th>Df Model:</th>              <td>    30</td>      <th>                     </th>     <td> </td>    
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
  <th>const</th>                <td>    0.0087</td> <td>    0.062</td> <td>    0.139</td> <td> 0.889</td> <td>   -0.114</td> <td>    0.131</td>
</tr>
<tr>
  <th>var_0_impact_code</th>    <td>    0.2479</td> <td>    0.057</td> <td>    4.328</td> <td> 0.000</td> <td>    0.136</td> <td>    0.360</td>
</tr>
<tr>
  <th>var_1_impact_code</th>    <td>    0.3416</td> <td>    0.052</td> <td>    6.537</td> <td> 0.000</td> <td>    0.239</td> <td>    0.444</td>
</tr>
<tr>
  <th>var_2_impact_code</th>    <td>    0.2574</td> <td>    0.056</td> <td>    4.636</td> <td> 0.000</td> <td>    0.149</td> <td>    0.366</td>
</tr>
<tr>
  <th>var_3_impact_code</th>    <td>    0.3248</td> <td>    0.052</td> <td>    6.232</td> <td> 0.000</td> <td>    0.223</td> <td>    0.427</td>
</tr>
<tr>
  <th>var_4_impact_code</th>    <td>    0.3262</td> <td>    0.056</td> <td>    5.850</td> <td> 0.000</td> <td>    0.217</td> <td>    0.436</td>
</tr>
<tr>
  <th>var_5_impact_code</th>    <td>    0.3327</td> <td>    0.058</td> <td>    5.735</td> <td> 0.000</td> <td>    0.219</td> <td>    0.446</td>
</tr>
<tr>
  <th>var_6_impact_code</th>    <td>    0.3398</td> <td>    0.051</td> <td>    6.706</td> <td> 0.000</td> <td>    0.240</td> <td>    0.439</td>
</tr>
<tr>
  <th>var_7_impact_code</th>    <td>    0.2294</td> <td>    0.057</td> <td>    4.040</td> <td> 0.000</td> <td>    0.118</td> <td>    0.341</td>
</tr>
<tr>
  <th>var_8_impact_code</th>    <td>    0.3869</td> <td>    0.055</td> <td>    6.982</td> <td> 0.000</td> <td>    0.278</td> <td>    0.496</td>
</tr>
<tr>
  <th>var_9_impact_code</th>    <td>    0.1673</td> <td>    0.055</td> <td>    3.041</td> <td> 0.002</td> <td>    0.059</td> <td>    0.275</td>
</tr>
<tr>
  <th>noise_0_impact_code</th>  <td>    0.0292</td> <td>    0.060</td> <td>    0.489</td> <td> 0.625</td> <td>   -0.088</td> <td>    0.147</td>
</tr>
<tr>
  <th>noise_1_impact_code</th>  <td>    0.1455</td> <td>    0.059</td> <td>    2.479</td> <td> 0.013</td> <td>    0.030</td> <td>    0.261</td>
</tr>
<tr>
  <th>noise_2_impact_code</th>  <td>   -0.0128</td> <td>    0.059</td> <td>   -0.217</td> <td> 0.829</td> <td>   -0.129</td> <td>    0.103</td>
</tr>
<tr>
  <th>noise_3_impact_code</th>  <td>    0.0399</td> <td>    0.060</td> <td>    0.661</td> <td> 0.509</td> <td>   -0.079</td> <td>    0.158</td>
</tr>
<tr>
  <th>noise_4_impact_code</th>  <td>    0.0190</td> <td>    0.059</td> <td>    0.321</td> <td> 0.748</td> <td>   -0.097</td> <td>    0.135</td>
</tr>
<tr>
  <th>noise_5_impact_code</th>  <td>   -0.0021</td> <td>    0.059</td> <td>   -0.035</td> <td> 0.972</td> <td>   -0.117</td> <td>    0.113</td>
</tr>
<tr>
  <th>noise_6_impact_code</th>  <td>    0.0942</td> <td>    0.059</td> <td>    1.595</td> <td> 0.111</td> <td>   -0.022</td> <td>    0.210</td>
</tr>
<tr>
  <th>noise_7_impact_code</th>  <td>   -0.0429</td> <td>    0.062</td> <td>   -0.692</td> <td> 0.489</td> <td>   -0.164</td> <td>    0.079</td>
</tr>
<tr>
  <th>noise_8_impact_code</th>  <td>   -0.0312</td> <td>    0.059</td> <td>   -0.528</td> <td> 0.597</td> <td>   -0.147</td> <td>    0.085</td>
</tr>
<tr>
  <th>noise_9_impact_code</th>  <td>   -0.0973</td> <td>    0.063</td> <td>   -1.548</td> <td> 0.122</td> <td>   -0.220</td> <td>    0.026</td>
</tr>
<tr>
  <th>noise_10_impact_code</th> <td>   -0.0740</td> <td>    0.061</td> <td>   -1.212</td> <td> 0.226</td> <td>   -0.194</td> <td>    0.046</td>
</tr>
<tr>
  <th>noise_11_impact_code</th> <td>   -0.0606</td> <td>    0.063</td> <td>   -0.959</td> <td> 0.338</td> <td>   -0.185</td> <td>    0.063</td>
</tr>
<tr>
  <th>noise_12_impact_code</th> <td>   -0.0123</td> <td>    0.058</td> <td>   -0.213</td> <td> 0.831</td> <td>   -0.126</td> <td>    0.101</td>
</tr>
<tr>
  <th>noise_13_impact_code</th> <td>    0.1073</td> <td>    0.059</td> <td>    1.823</td> <td> 0.068</td> <td>   -0.008</td> <td>    0.223</td>
</tr>
<tr>
  <th>noise_14_impact_code</th> <td>   -0.0154</td> <td>    0.057</td> <td>   -0.268</td> <td> 0.788</td> <td>   -0.128</td> <td>    0.097</td>
</tr>
<tr>
  <th>noise_15_impact_code</th> <td>   -0.0128</td> <td>    0.061</td> <td>   -0.209</td> <td> 0.835</td> <td>   -0.133</td> <td>    0.108</td>
</tr>
<tr>
  <th>noise_16_impact_code</th> <td>    0.0798</td> <td>    0.060</td> <td>    1.326</td> <td> 0.185</td> <td>   -0.038</td> <td>    0.198</td>
</tr>
<tr>
  <th>noise_17_impact_code</th> <td>    0.0146</td> <td>    0.060</td> <td>    0.246</td> <td> 0.806</td> <td>   -0.102</td> <td>    0.131</td>
</tr>
<tr>
  <th>noise_18_impact_code</th> <td>    0.0140</td> <td>    0.059</td> <td>    0.237</td> <td> 0.813</td> <td>   -0.102</td> <td>    0.130</td>
</tr>
<tr>
  <th>noise_19_impact_code</th> <td>    0.0472</td> <td>    0.057</td> <td>    0.832</td> <td> 0.405</td> <td>   -0.064</td> <td>    0.159</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td> 1.792</td> <th>  Durbin-Watson:     </th> <td>   1.975</td>
</tr>
<tr>
  <th>Prob(Omnibus):</th> <td> 0.408</td> <th>  Jarque-Bera (JB):  </th> <td>   1.828</td>
</tr>
<tr>
  <th>Skew:</th>          <td>-0.046</td> <th>  Prob(JB):          </th> <td>   0.401</td>
</tr>
<tr>
  <th>Kurtosis:</th>      <td> 2.906</td> <th>  Cond. No.          </th> <td>    1.36</td>
</tr>
</table><br/><br/>Warnings:<br/>[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.




```python
sklearn.metrics.r2_score(plot_train["y"],plot_train["predict_cross_all_vars"])
```




    0.12137175133292588




```python
sklearn.metrics.r2_score(plot_test["y"],plot_test["predict_cross_all_vars"])
```




    0.10060355162051227




```python
smf3 = statsmodels.api.OLS(
    y_train, 
    statsmodels.api.add_constant(cross_frame[recommended_vars])).fit()
smf3.summary()
```




<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>            <td>y</td>        <th>  R-squared:         </th> <td>   0.116</td> 
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th> <td>   0.112</td> 
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th> <td>   30.04</td> 
</tr>
<tr>
  <th>Date:</th>             <td>Thu, 25 Jul 2019</td> <th>  Prob (F-statistic):</th> <td>4.20e-60</td> 
</tr>
<tr>
  <th>Time:</th>                 <td>15:49:06</td>     <th>  Log-Likelihood:    </th> <td> -6451.3</td> 
</tr>
<tr>
  <th>No. Observations:</th>      <td>  2527</td>      <th>  AIC:               </th> <td>1.293e+04</td>
</tr>
<tr>
  <th>Df Residuals:</th>          <td>  2515</td>      <th>  BIC:               </th> <td>1.300e+04</td>
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
           <td></td>              <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>const</th>               <td>    0.0067</td> <td>    0.062</td> <td>    0.108</td> <td> 0.914</td> <td>   -0.115</td> <td>    0.129</td>
</tr>
<tr>
  <th>var_0_impact_code</th>   <td>    0.2554</td> <td>    0.057</td> <td>    4.476</td> <td> 0.000</td> <td>    0.143</td> <td>    0.367</td>
</tr>
<tr>
  <th>var_1_impact_code</th>   <td>    0.3397</td> <td>    0.052</td> <td>    6.528</td> <td> 0.000</td> <td>    0.238</td> <td>    0.442</td>
</tr>
<tr>
  <th>var_2_impact_code</th>   <td>    0.2548</td> <td>    0.055</td> <td>    4.612</td> <td> 0.000</td> <td>    0.146</td> <td>    0.363</td>
</tr>
<tr>
  <th>var_3_impact_code</th>   <td>    0.3229</td> <td>    0.052</td> <td>    6.226</td> <td> 0.000</td> <td>    0.221</td> <td>    0.425</td>
</tr>
<tr>
  <th>var_4_impact_code</th>   <td>    0.3239</td> <td>    0.055</td> <td>    5.844</td> <td> 0.000</td> <td>    0.215</td> <td>    0.433</td>
</tr>
<tr>
  <th>var_5_impact_code</th>   <td>    0.3335</td> <td>    0.058</td> <td>    5.775</td> <td> 0.000</td> <td>    0.220</td> <td>    0.447</td>
</tr>
<tr>
  <th>var_6_impact_code</th>   <td>    0.3403</td> <td>    0.050</td> <td>    6.744</td> <td> 0.000</td> <td>    0.241</td> <td>    0.439</td>
</tr>
<tr>
  <th>var_7_impact_code</th>   <td>    0.2368</td> <td>    0.057</td> <td>    4.188</td> <td> 0.000</td> <td>    0.126</td> <td>    0.348</td>
</tr>
<tr>
  <th>var_8_impact_code</th>   <td>    0.3952</td> <td>    0.055</td> <td>    7.172</td> <td> 0.000</td> <td>    0.287</td> <td>    0.503</td>
</tr>
<tr>
  <th>var_9_impact_code</th>   <td>    0.1638</td> <td>    0.055</td> <td>    2.997</td> <td> 0.003</td> <td>    0.057</td> <td>    0.271</td>
</tr>
<tr>
  <th>noise_1_impact_code</th> <td>    0.1497</td> <td>    0.058</td> <td>    2.568</td> <td> 0.010</td> <td>    0.035</td> <td>    0.264</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td> 1.857</td> <th>  Durbin-Watson:     </th> <td>   1.981</td>
</tr>
<tr>
  <th>Prob(Omnibus):</th> <td> 0.395</td> <th>  Jarque-Bera (JB):  </th> <td>   1.859</td>
</tr>
<tr>
  <th>Skew:</th>          <td>-0.037</td> <th>  Prob(JB):          </th> <td>   0.395</td>
</tr>
<tr>
  <th>Kurtosis:</th>      <td> 2.890</td> <th>  Cond. No.          </th> <td>    1.28</td>
</tr>
</table><br/><br/>Warnings:<br/>[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.




```python
sklearn.metrics.r2_score(plot_train["y"],plot_train["predict_cross_recommended_vars"])
```




    0.11612637705646545




```python
sklearn.metrics.r2_score(plot_test["y"],plot_test["predict_cross_recommended_vars"])
```




    0.10635331848351703




```python

```
