

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
      <td>1.383513</td>
      <td>level_352</td>
      <td>level_101</td>
      <td>level_48</td>
      <td>level_309</td>
      <td>level_474</td>
      <td>level_52</td>
      <td>level_96</td>
      <td>level_24</td>
      <td>level_104</td>
      <td>...</td>
      <td>level_363</td>
      <td>level_193</td>
      <td>level_34</td>
      <td>level_312</td>
      <td>level_452</td>
      <td>level_335</td>
      <td>level_385</td>
      <td>level_128</td>
      <td>level_26</td>
      <td>level_309</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-0.026570</td>
      <td>level_26</td>
      <td>level_167</td>
      <td>level_291</td>
      <td>level_167</td>
      <td>level_143</td>
      <td>level_384</td>
      <td>level_184</td>
      <td>level_484</td>
      <td>level_410</td>
      <td>...</td>
      <td>level_5</td>
      <td>level_224</td>
      <td>level_371</td>
      <td>level_36</td>
      <td>level_120</td>
      <td>level_427</td>
      <td>level_436</td>
      <td>level_117</td>
      <td>level_214</td>
      <td>level_173</td>
    </tr>
    <tr>
      <th>2</th>
      <td>27.258577</td>
      <td>level_297</td>
      <td>level_424</td>
      <td>level_50</td>
      <td>level_433</td>
      <td>level_192</td>
      <td>level_254</td>
      <td>level_390</td>
      <td>level_386</td>
      <td>level_262</td>
      <td>...</td>
      <td>level_38</td>
      <td>level_179</td>
      <td>level_153</td>
      <td>level_325</td>
      <td>level_288</td>
      <td>level_117</td>
      <td>level_363</td>
      <td>level_307</td>
      <td>level_403</td>
      <td>level_397</td>
    </tr>
    <tr>
      <th>3</th>
      <td>12.502685</td>
      <td>level_449</td>
      <td>level_83</td>
      <td>level_491</td>
      <td>level_371</td>
      <td>level_296</td>
      <td>level_208</td>
      <td>level_372</td>
      <td>level_436</td>
      <td>level_136</td>
      <td>...</td>
      <td>level_470</td>
      <td>level_382</td>
      <td>level_79</td>
      <td>level_198</td>
      <td>level_221</td>
      <td>level_82</td>
      <td>level_28</td>
      <td>level_41</td>
      <td>level_69</td>
      <td>level_498</td>
    </tr>
    <tr>
      <th>4</th>
      <td>16.899068</td>
      <td>level_385</td>
      <td>level_274</td>
      <td>level_259</td>
      <td>level_406</td>
      <td>level_419</td>
      <td>level_442</td>
      <td>level_435</td>
      <td>level_69</td>
      <td>level_486</td>
      <td>...</td>
      <td>level_423</td>
      <td>level_155</td>
      <td>level_165</td>
      <td>level_111</td>
      <td>level_235</td>
      <td>level_3</td>
      <td>level_104</td>
      <td>level_477</td>
      <td>level_68</td>
      <td>level_199</td>
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
      <td>-11.684808</td>
      <td>-18.477390</td>
      <td>6.801226</td>
      <td>17.629075</td>
      <td>2.982388</td>
      <td>4.588775</td>
      <td>-4.962607</td>
      <td>1.169723</td>
      <td>11.835071</td>
      <td>-2.951798</td>
      <td>...</td>
      <td>-10.828920</td>
      <td>0.019370</td>
      <td>5.755351</td>
      <td>-1.725535</td>
      <td>-9.240895</td>
      <td>7.420990</td>
      <td>-11.527138</td>
      <td>-1.127728</td>
      <td>-5.007127</td>
      <td>0.373246</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-23.375041</td>
      <td>9.870295</td>
      <td>6.085724</td>
      <td>5.484843</td>
      <td>10.598310</td>
      <td>14.951526</td>
      <td>10.611742</td>
      <td>0.333619</td>
      <td>11.242828</td>
      <td>6.782952</td>
      <td>...</td>
      <td>9.312092</td>
      <td>4.284838</td>
      <td>8.441365</td>
      <td>-9.045142</td>
      <td>-0.053117</td>
      <td>-4.577667</td>
      <td>12.756304</td>
      <td>9.386368</td>
      <td>4.219720</td>
      <td>-1.614600</td>
    </tr>
    <tr>
      <th>2</th>
      <td>16.036522</td>
      <td>2.468665</td>
      <td>2.632923</td>
      <td>2.496307</td>
      <td>13.086242</td>
      <td>-2.276703</td>
      <td>-4.466609</td>
      <td>3.039245</td>
      <td>5.604963</td>
      <td>0.497867</td>
      <td>...</td>
      <td>2.466489</td>
      <td>2.644322</td>
      <td>14.700274</td>
      <td>4.121955</td>
      <td>14.715039</td>
      <td>5.481840</td>
      <td>1.811169</td>
      <td>0.878590</td>
      <td>-3.501312</td>
      <td>9.886655</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-8.134826</td>
      <td>0.657330</td>
      <td>3.422340</td>
      <td>0.976318</td>
      <td>-6.963903</td>
      <td>-3.528208</td>
      <td>2.646056</td>
      <td>-7.726300</td>
      <td>1.340601</td>
      <td>-3.991396</td>
      <td>...</td>
      <td>-4.460876</td>
      <td>4.853681</td>
      <td>8.687255</td>
      <td>3.428803</td>
      <td>1.385715</td>
      <td>-13.991781</td>
      <td>-12.372685</td>
      <td>-10.019619</td>
      <td>0.448116</td>
      <td>-5.343279</td>
    </tr>
    <tr>
      <th>4</th>
      <td>13.835043</td>
      <td>4.758503</td>
      <td>0.302671</td>
      <td>6.721130</td>
      <td>1.771526</td>
      <td>-12.802620</td>
      <td>3.487949</td>
      <td>4.072454</td>
      <td>-8.989508</td>
      <td>6.145415</td>
      <td>...</td>
      <td>-0.438651</td>
      <td>-0.604513</td>
      <td>2.417146</td>
      <td>11.634948</td>
      <td>5.382935</td>
      <td>-0.739027</td>
      <td>-5.140218</td>
      <td>-2.147282</td>
      <td>-4.231404</td>
      <td>-14.797894</td>
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
      <td>-6.922003</td>
      <td>-10.204846</td>
      <td>4.498360</td>
      <td>3.225548</td>
      <td>2.137947</td>
      <td>2.793662</td>
      <td>-4.309458</td>
      <td>1.035947</td>
      <td>9.147490</td>
      <td>-2.488887</td>
      <td>...</td>
      <td>-4.839085</td>
      <td>0.012283</td>
      <td>4.007744</td>
      <td>-1.116215</td>
      <td>-5.657213</td>
      <td>5.113925</td>
      <td>-8.925330</td>
      <td>-0.616798</td>
      <td>-4.639828</td>
      <td>0.239002</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-10.205297</td>
      <td>7.973106</td>
      <td>4.011322</td>
      <td>4.404550</td>
      <td>9.972326</td>
      <td>9.931526</td>
      <td>5.974046</td>
      <td>0.187770</td>
      <td>8.255927</td>
      <td>5.897103</td>
      <td>...</td>
      <td>8.798308</td>
      <td>1.883913</td>
      <td>4.974202</td>
      <td>-2.921928</td>
      <td>-0.036870</td>
      <td>-3.184418</td>
      <td>0.000000</td>
      <td>7.551188</td>
      <td>2.175405</td>
      <td>-1.197707</td>
    </tr>
    <tr>
      <th>2</th>
      <td>9.570785</td>
      <td>1.857906</td>
      <td>2.022711</td>
      <td>1.808472</td>
      <td>9.770240</td>
      <td>-1.920170</td>
      <td>-2.021494</td>
      <td>1.846224</td>
      <td>2.793820</td>
      <td>0.248783</td>
      <td>...</td>
      <td>1.755253</td>
      <td>1.177430</td>
      <td>10.151408</td>
      <td>2.216398</td>
      <td>10.093101</td>
      <td>4.168719</td>
      <td>1.402266</td>
      <td>0.505821</td>
      <td>-0.623865</td>
      <td>7.419761</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-5.914109</td>
      <td>0.319000</td>
      <td>2.421041</td>
      <td>0.853791</td>
      <td>-4.456241</td>
      <td>-2.976404</td>
      <td>2.377143</td>
      <td>-5.417539</td>
      <td>1.201333</td>
      <td>-3.407620</td>
      <td>...</td>
      <td>-3.547500</td>
      <td>4.466916</td>
      <td>4.693266</td>
      <td>2.450456</td>
      <td>0.823708</td>
      <td>-8.452016</td>
      <td>-8.078761</td>
      <td>-4.850079</td>
      <td>0.222080</td>
      <td>-3.254473</td>
    </tr>
    <tr>
      <th>4</th>
      <td>8.169779</td>
      <td>2.310153</td>
      <td>0.145656</td>
      <td>5.000864</td>
      <td>0.789880</td>
      <td>-6.448085</td>
      <td>2.508478</td>
      <td>2.617945</td>
      <td>-5.990377</td>
      <td>3.917369</td>
      <td>...</td>
      <td>-0.208001</td>
      <td>-0.467094</td>
      <td>1.137206</td>
      <td>6.845210</td>
      <td>0.868897</td>
      <td>-0.579928</td>
      <td>-2.843802</td>
      <td>-0.635050</td>
      <td>-2.623697</td>
      <td>-3.766177</td>
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
      <td>-5.322267</td>
      <td>-15.999631</td>
      <td>1.893563</td>
      <td>5.551115e-17</td>
      <td>4.961150e+00</td>
      <td>3.013425</td>
      <td>-9.614742</td>
      <td>-0.447051</td>
      <td>3.989984e+00</td>
      <td>0.000000</td>
      <td>...</td>
      <td>-5.917140</td>
      <td>-0.191053</td>
      <td>6.893068</td>
      <td>-3.579628</td>
      <td>-20.043028</td>
      <td>3.516064</td>
      <td>-10.185356</td>
      <td>-4.816471</td>
      <td>-6.188825</td>
      <td>1.924721</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-22.195557</td>
      <td>4.387807</td>
      <td>1.792151</td>
      <td>6.014366e+00</td>
      <td>7.136561e+00</td>
      <td>12.898983</td>
      <td>2.057459</td>
      <td>-11.425445</td>
      <td>6.468179e+00</td>
      <td>0.000000</td>
      <td>...</td>
      <td>7.597271</td>
      <td>0.451719</td>
      <td>4.553115</td>
      <td>-2.418727</td>
      <td>-2.382286</td>
      <td>-5.694873</td>
      <td>0.000000</td>
      <td>6.317929</td>
      <td>-5.062048</td>
      <td>-8.481378</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4.753891</td>
      <td>-2.452018</td>
      <td>-1.008761</td>
      <td>-2.298021e+00</td>
      <td>0.000000e+00</td>
      <td>-4.826413</td>
      <td>-6.702588</td>
      <td>0.031657</td>
      <td>-2.932530e+00</td>
      <td>-1.410207</td>
      <td>...</td>
      <td>1.414789</td>
      <td>-2.777331</td>
      <td>8.245171</td>
      <td>1.179748</td>
      <td>5.124789</td>
      <td>-0.471257</td>
      <td>-4.123788</td>
      <td>-3.408455</td>
      <td>-0.056268</td>
      <td>5.054761</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.000000</td>
      <td>-0.087741</td>
      <td>3.335645</td>
      <td>5.551115e-17</td>
      <td>-4.011519e+00</td>
      <td>-4.878556</td>
      <td>1.902082</td>
      <td>-8.515607</td>
      <td>-5.551115e-17</td>
      <td>-3.783779</td>
      <td>...</td>
      <td>-4.105557</td>
      <td>5.946857</td>
      <td>0.270925</td>
      <td>2.089126</td>
      <td>3.905227</td>
      <td>-16.358475</td>
      <td>-7.568151</td>
      <td>-4.497116</td>
      <td>0.169481</td>
      <td>-11.246009</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.000000</td>
      <td>2.414175</td>
      <td>-0.914418</td>
      <td>3.074964e+00</td>
      <td>5.551115e-17</td>
      <td>-11.875552</td>
      <td>0.864088</td>
      <td>0.395670</td>
      <td>-9.489654e+00</td>
      <td>2.649852</td>
      <td>...</td>
      <td>-0.282157</td>
      <td>-4.193109</td>
      <td>-1.587273</td>
      <td>0.694894</td>
      <td>0.000000</td>
      <td>-1.325272</td>
      <td>-2.810178</td>
      <td>-8.803324</td>
      <td>-6.757839</td>
      <td>-11.492468</td>
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
      <td>0.537988</td>
      <td>0.516793</td>
      <td>0.187620</td>
      <td>0.166293</td>
      <td>False</td>
    </tr>
    <tr>
      <th>1</th>
      <td>var_1_impact_code</td>
      <td>0.529995</td>
      <td>0.500877</td>
      <td>0.159943</td>
      <td>0.129622</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2</th>
      <td>var_2_impact_code</td>
      <td>0.524644</td>
      <td>0.498001</td>
      <td>0.130256</td>
      <td>0.198606</td>
      <td>False</td>
    </tr>
    <tr>
      <th>3</th>
      <td>var_3_impact_code</td>
      <td>0.534853</td>
      <td>0.509648</td>
      <td>0.187316</td>
      <td>0.162451</td>
      <td>False</td>
    </tr>
    <tr>
      <th>4</th>
      <td>var_4_impact_code</td>
      <td>0.541740</td>
      <td>0.518620</td>
      <td>0.196953</td>
      <td>0.197392</td>
      <td>False</td>
    </tr>
    <tr>
      <th>5</th>
      <td>var_5_impact_code</td>
      <td>0.510184</td>
      <td>0.481632</td>
      <td>0.148966</td>
      <td>0.175461</td>
      <td>False</td>
    </tr>
    <tr>
      <th>6</th>
      <td>var_6_impact_code</td>
      <td>0.495153</td>
      <td>0.467347</td>
      <td>0.115854</td>
      <td>0.168722</td>
      <td>False</td>
    </tr>
    <tr>
      <th>7</th>
      <td>var_7_impact_code</td>
      <td>0.534236</td>
      <td>0.509114</td>
      <td>0.168382</td>
      <td>0.168450</td>
      <td>False</td>
    </tr>
    <tr>
      <th>8</th>
      <td>var_8_impact_code</td>
      <td>0.519112</td>
      <td>0.496166</td>
      <td>0.146696</td>
      <td>0.175128</td>
      <td>False</td>
    </tr>
    <tr>
      <th>9</th>
      <td>var_9_impact_code</td>
      <td>0.501486</td>
      <td>0.477278</td>
      <td>0.125437</td>
      <td>0.161875</td>
      <td>False</td>
    </tr>
    <tr>
      <th>10</th>
      <td>noise_0_impact_code</td>
      <td>0.446225</td>
      <td>0.411167</td>
      <td>0.005974</td>
      <td>0.019324</td>
      <td>True</td>
    </tr>
    <tr>
      <th>11</th>
      <td>noise_1_impact_code</td>
      <td>0.433265</td>
      <td>0.405773</td>
      <td>-0.007060</td>
      <td>-0.019402</td>
      <td>True</td>
    </tr>
    <tr>
      <th>12</th>
      <td>noise_2_impact_code</td>
      <td>0.429106</td>
      <td>0.399946</td>
      <td>-0.031154</td>
      <td>-0.005404</td>
      <td>True</td>
    </tr>
    <tr>
      <th>13</th>
      <td>noise_3_impact_code</td>
      <td>0.431931</td>
      <td>0.400783</td>
      <td>-0.017268</td>
      <td>-0.010527</td>
      <td>True</td>
    </tr>
    <tr>
      <th>14</th>
      <td>noise_4_impact_code</td>
      <td>0.443634</td>
      <td>0.410297</td>
      <td>0.000321</td>
      <td>-0.022053</td>
      <td>True</td>
    </tr>
    <tr>
      <th>15</th>
      <td>noise_5_impact_code</td>
      <td>0.446253</td>
      <td>0.416318</td>
      <td>0.008559</td>
      <td>-0.014757</td>
      <td>True</td>
    </tr>
    <tr>
      <th>16</th>
      <td>noise_6_impact_code</td>
      <td>0.430318</td>
      <td>0.397335</td>
      <td>0.006583</td>
      <td>-0.026171</td>
      <td>True</td>
    </tr>
    <tr>
      <th>17</th>
      <td>noise_7_impact_code</td>
      <td>0.448627</td>
      <td>0.418781</td>
      <td>0.008814</td>
      <td>0.009707</td>
      <td>True</td>
    </tr>
    <tr>
      <th>18</th>
      <td>noise_8_impact_code</td>
      <td>0.444763</td>
      <td>0.411237</td>
      <td>0.030997</td>
      <td>-0.004198</td>
      <td>True</td>
    </tr>
    <tr>
      <th>19</th>
      <td>noise_9_impact_code</td>
      <td>0.444665</td>
      <td>0.415596</td>
      <td>0.011580</td>
      <td>-0.005400</td>
      <td>True</td>
    </tr>
    <tr>
      <th>20</th>
      <td>noise_10_impact_code</td>
      <td>0.447439</td>
      <td>0.416262</td>
      <td>0.013152</td>
      <td>-0.024589</td>
      <td>True</td>
    </tr>
    <tr>
      <th>21</th>
      <td>noise_11_impact_code</td>
      <td>0.442466</td>
      <td>0.412047</td>
      <td>-0.002127</td>
      <td>0.037705</td>
      <td>True</td>
    </tr>
    <tr>
      <th>22</th>
      <td>noise_12_impact_code</td>
      <td>0.449557</td>
      <td>0.417318</td>
      <td>-0.006071</td>
      <td>-0.006933</td>
      <td>True</td>
    </tr>
    <tr>
      <th>23</th>
      <td>noise_13_impact_code</td>
      <td>0.433448</td>
      <td>0.400939</td>
      <td>0.020533</td>
      <td>0.004272</td>
      <td>True</td>
    </tr>
    <tr>
      <th>24</th>
      <td>noise_14_impact_code</td>
      <td>0.433950</td>
      <td>0.403115</td>
      <td>-0.011415</td>
      <td>0.049834</td>
      <td>True</td>
    </tr>
    <tr>
      <th>25</th>
      <td>noise_15_impact_code</td>
      <td>0.463960</td>
      <td>0.428932</td>
      <td>0.019939</td>
      <td>0.028064</td>
      <td>True</td>
    </tr>
    <tr>
      <th>26</th>
      <td>noise_16_impact_code</td>
      <td>0.451931</td>
      <td>0.420896</td>
      <td>0.026720</td>
      <td>0.000889</td>
      <td>True</td>
    </tr>
    <tr>
      <th>27</th>
      <td>noise_17_impact_code</td>
      <td>0.439838</td>
      <td>0.409508</td>
      <td>0.003635</td>
      <td>0.003331</td>
      <td>True</td>
    </tr>
    <tr>
      <th>28</th>
      <td>noise_18_impact_code</td>
      <td>0.457331</td>
      <td>0.429523</td>
      <td>0.053861</td>
      <td>0.025831</td>
      <td>True</td>
    </tr>
    <tr>
      <th>29</th>
      <td>noise_19_impact_code</td>
      <td>0.418562</td>
      <td>0.383158</td>
      <td>-0.080065</td>
      <td>0.010104</td>
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
      <td>0.432872</td>
      <td>0.405282</td>
      <td>-0.020934</td>
      <td>0.011292</td>
      <td>True</td>
    </tr>
    <tr>
      <th>81</th>
      <td>noise_71_impact_code</td>
      <td>0.429623</td>
      <td>0.400085</td>
      <td>-0.032647</td>
      <td>-0.012477</td>
      <td>True</td>
    </tr>
    <tr>
      <th>82</th>
      <td>noise_72_impact_code</td>
      <td>0.436041</td>
      <td>0.402801</td>
      <td>0.007439</td>
      <td>-0.012775</td>
      <td>True</td>
    </tr>
    <tr>
      <th>83</th>
      <td>noise_73_impact_code</td>
      <td>0.442455</td>
      <td>0.406389</td>
      <td>-0.006272</td>
      <td>-0.030470</td>
      <td>True</td>
    </tr>
    <tr>
      <th>84</th>
      <td>noise_74_impact_code</td>
      <td>0.429879</td>
      <td>0.400453</td>
      <td>-0.020780</td>
      <td>0.001609</td>
      <td>True</td>
    </tr>
    <tr>
      <th>85</th>
      <td>noise_75_impact_code</td>
      <td>0.454616</td>
      <td>0.424296</td>
      <td>0.007740</td>
      <td>-0.007346</td>
      <td>True</td>
    </tr>
    <tr>
      <th>86</th>
      <td>noise_76_impact_code</td>
      <td>0.422217</td>
      <td>0.385097</td>
      <td>-0.042085</td>
      <td>-0.010987</td>
      <td>True</td>
    </tr>
    <tr>
      <th>87</th>
      <td>noise_77_impact_code</td>
      <td>0.472292</td>
      <td>0.442930</td>
      <td>0.042566</td>
      <td>0.042495</td>
      <td>True</td>
    </tr>
    <tr>
      <th>88</th>
      <td>noise_78_impact_code</td>
      <td>0.444402</td>
      <td>0.412618</td>
      <td>0.007244</td>
      <td>0.011182</td>
      <td>True</td>
    </tr>
    <tr>
      <th>89</th>
      <td>noise_79_impact_code</td>
      <td>0.442697</td>
      <td>0.412583</td>
      <td>0.031478</td>
      <td>-0.021351</td>
      <td>True</td>
    </tr>
    <tr>
      <th>90</th>
      <td>noise_80_impact_code</td>
      <td>0.448210</td>
      <td>0.414825</td>
      <td>0.033774</td>
      <td>-0.017440</td>
      <td>True</td>
    </tr>
    <tr>
      <th>91</th>
      <td>noise_81_impact_code</td>
      <td>0.435966</td>
      <td>0.400709</td>
      <td>-0.007978</td>
      <td>0.034094</td>
      <td>True</td>
    </tr>
    <tr>
      <th>92</th>
      <td>noise_82_impact_code</td>
      <td>0.447206</td>
      <td>0.412945</td>
      <td>-0.013301</td>
      <td>0.001408</td>
      <td>True</td>
    </tr>
    <tr>
      <th>93</th>
      <td>noise_83_impact_code</td>
      <td>0.451681</td>
      <td>0.416716</td>
      <td>0.024937</td>
      <td>-0.052670</td>
      <td>True</td>
    </tr>
    <tr>
      <th>94</th>
      <td>noise_84_impact_code</td>
      <td>0.412559</td>
      <td>0.376958</td>
      <td>-0.066162</td>
      <td>0.002653</td>
      <td>True</td>
    </tr>
    <tr>
      <th>95</th>
      <td>noise_85_impact_code</td>
      <td>0.452560</td>
      <td>0.426902</td>
      <td>0.018183</td>
      <td>-0.015517</td>
      <td>True</td>
    </tr>
    <tr>
      <th>96</th>
      <td>noise_86_impact_code</td>
      <td>0.444211</td>
      <td>0.407383</td>
      <td>-0.023706</td>
      <td>-0.008768</td>
      <td>True</td>
    </tr>
    <tr>
      <th>97</th>
      <td>noise_87_impact_code</td>
      <td>0.439506</td>
      <td>0.405778</td>
      <td>-0.001028</td>
      <td>0.000972</td>
      <td>True</td>
    </tr>
    <tr>
      <th>98</th>
      <td>noise_88_impact_code</td>
      <td>0.424965</td>
      <td>0.397122</td>
      <td>-0.013679</td>
      <td>-0.019896</td>
      <td>True</td>
    </tr>
    <tr>
      <th>99</th>
      <td>noise_89_impact_code</td>
      <td>0.412794</td>
      <td>0.376263</td>
      <td>-0.057370</td>
      <td>-0.015923</td>
      <td>True</td>
    </tr>
    <tr>
      <th>100</th>
      <td>noise_90_impact_code</td>
      <td>0.435000</td>
      <td>0.401312</td>
      <td>0.003770</td>
      <td>-0.013418</td>
      <td>True</td>
    </tr>
    <tr>
      <th>101</th>
      <td>noise_91_impact_code</td>
      <td>0.430119</td>
      <td>0.394681</td>
      <td>-0.027264</td>
      <td>0.028582</td>
      <td>True</td>
    </tr>
    <tr>
      <th>102</th>
      <td>noise_92_impact_code</td>
      <td>0.449887</td>
      <td>0.423104</td>
      <td>0.003952</td>
      <td>0.022531</td>
      <td>True</td>
    </tr>
    <tr>
      <th>103</th>
      <td>noise_93_impact_code</td>
      <td>0.410867</td>
      <td>0.380193</td>
      <td>-0.021012</td>
      <td>0.006417</td>
      <td>True</td>
    </tr>
    <tr>
      <th>104</th>
      <td>noise_94_impact_code</td>
      <td>0.449876</td>
      <td>0.420410</td>
      <td>0.022913</td>
      <td>-0.022740</td>
      <td>True</td>
    </tr>
    <tr>
      <th>105</th>
      <td>noise_95_impact_code</td>
      <td>0.436371</td>
      <td>0.401349</td>
      <td>-0.013341</td>
      <td>0.006263</td>
      <td>True</td>
    </tr>
    <tr>
      <th>106</th>
      <td>noise_96_impact_code</td>
      <td>0.443538</td>
      <td>0.410439</td>
      <td>0.015183</td>
      <td>0.011904</td>
      <td>True</td>
    </tr>
    <tr>
      <th>107</th>
      <td>noise_97_impact_code</td>
      <td>0.431936</td>
      <td>0.398895</td>
      <td>-0.016899</td>
      <td>0.018587</td>
      <td>True</td>
    </tr>
    <tr>
      <th>108</th>
      <td>noise_98_impact_code</td>
      <td>0.420400</td>
      <td>0.385987</td>
      <td>-0.038401</td>
      <td>0.001377</td>
      <td>True</td>
    </tr>
    <tr>
      <th>109</th>
      <td>noise_99_impact_code</td>
      <td>0.426315</td>
      <td>0.394024</td>
      <td>-0.009541</td>
      <td>-0.029113</td>
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

    (0.8552856382584195, 1.295641205227952e-32)



![png](output_18_1.png)



```python
print(scipy.stats.pearsonr(corr_frame['naive_train_hierarchical_correlation'], corr_frame['test_correlation']))
seaborn.scatterplot(x = "naive_train_hierarchical_correlation", y = "test_correlation", data = corr_frame, hue = "is_noise")
matplotlib.pyplot.plot([-1, 1], [-1, 1], color="red")
matplotlib.pyplot.xlim(-.2,1)
matplotlib.pyplot.ylim(-.2,1)
matplotlib.pyplot.gca().set_aspect('equal', adjustable='box')
```

    (0.85747877102484, 6.0391214806109486e-33)



![png](output_19_1.png)



```python
print(scipy.stats.pearsonr(corr_frame['cross_frame_correlation'], corr_frame['test_correlation']))
seaborn.scatterplot(x = "cross_frame_correlation", y = "test_correlation", data = corr_frame,  hue = "is_noise")
matplotlib.pyplot.plot([-1, 1], [-1, 1], color="red")
matplotlib.pyplot.xlim(-.2,1)
matplotlib.pyplot.ylim(-.2,1)
matplotlib.pyplot.gca().set_aspect('equal', adjustable='box')
```

    (0.8278827854397146, 6.985268378478407e-29)



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
      <td>-0.013341</td>
      <td>0.499015</td>
      <td>110.0</td>
      <td>False</td>
    </tr>
    <tr>
      <th>106</th>
      <td>noise_96_impact_code</td>
      <td>impact_code</td>
      <td>True</td>
      <td>True</td>
      <td>0.015183</td>
      <td>0.441665</td>
      <td>110.0</td>
      <td>False</td>
    </tr>
    <tr>
      <th>107</th>
      <td>noise_97_impact_code</td>
      <td>impact_code</td>
      <td>True</td>
      <td>True</td>
      <td>-0.016899</td>
      <td>0.391818</td>
      <td>110.0</td>
      <td>False</td>
    </tr>
    <tr>
      <th>108</th>
      <td>noise_98_impact_code</td>
      <td>impact_code</td>
      <td>True</td>
      <td>True</td>
      <td>-0.038401</td>
      <td>0.051592</td>
      <td>110.0</td>
      <td>False</td>
    </tr>
    <tr>
      <th>109</th>
      <td>noise_99_impact_code</td>
      <td>impact_code</td>
      <td>True</td>
      <td>True</td>
      <td>-0.009541</td>
      <td>0.628779</td>
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
     'noise_18_impact_code']




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
      <td>-0.026570</td>
      <td>1.494868</td>
      <td>0.810747</td>
      <td>1.532900</td>
      <td>0.778234</td>
    </tr>
    <tr>
      <th>1</th>
      <td>27.258577</td>
      <td>3.548667</td>
      <td>1.247861</td>
      <td>0.085402</td>
      <td>-0.663297</td>
    </tr>
    <tr>
      <th>2</th>
      <td>16.899068</td>
      <td>0.807263</td>
      <td>-0.755245</td>
      <td>-0.520152</td>
      <td>-1.183128</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-0.028523</td>
      <td>10.112324</td>
      <td>2.623343</td>
      <td>2.602956</td>
      <td>4.702430</td>
    </tr>
    <tr>
      <th>4</th>
      <td>23.936118</td>
      <td>-12.158877</td>
      <td>-15.492469</td>
      <td>-7.039041</td>
      <td>-11.418475</td>
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

    3.206479233301278



![png](output_33_1.png)



```python
print(rmse(plot_train["predict_naive_hierarchical_all_vars"], plot_train["y"]))
seaborn.scatterplot(x="predict_naive_hierarchical_all_vars", y ="y", data = plot_train)
plt = matplotlib.pyplot.title("Naive hierarchical prediction on train")
```

    4.651747067091252



![png](output_34_1.png)



```python
print(rmse(plot_train["predict_cross_all_vars"], plot_train["y"]))
seaborn.scatterplot(x="predict_cross_all_vars", y ="y", data = plot_train)
plt = matplotlib.pyplot.title("Cross prediction on (all vars) train")
```

    15.42621122814677



![png](output_35_1.png)



```python
print(rmse(plot_train["predict_cross_recommended_vars"], plot_train["y"]))
seaborn.scatterplot(x="predict_cross_recommended_vars", y ="y", data = plot_train)
plt = matplotlib.pyplot.title("Cross prediction (recommended vars) on train")
```

    15.872204115447927



![png](output_36_1.png)



```python
print(rmse(plot_test["predict_naive_empirical_all_vars"], plot_test["y"]))
seaborn.scatterplot(x="predict_naive_empirical_all_vars", y ="y", data = plot_test)
plt = matplotlib.pyplot.title("Naive empirical prediction on test")
```

    17.5439880598221



![png](output_37_1.png)



```python
print(rmse(plot_test["predict_naive_hierarchical_all_vars"], plot_test["y"]))
seaborn.scatterplot(x="predict_naive_hierarchical_all_vars", y ="y", data = plot_test)
plt = matplotlib.pyplot.title("Naive hierarchical prediction on test")
```

    17.2340552407076



![png](output_38_1.png)



```python
print(rmse(plot_test["predict_cross_all_vars"], plot_test["y"]))
seaborn.scatterplot(x="predict_cross_all_vars", y ="y", data = plot_test)
plt = matplotlib.pyplot.title("Cross prediction on (all vars) test")
```

    15.65079425480502



![png](output_39_1.png)



```python
print(rmse(plot_test["predict_cross_recommended_vars"], plot_test["y"]))
seaborn.scatterplot(x="predict_cross_recommended_vars", y ="y", data = plot_test)
plt = matplotlib.pyplot.title("Cross prediction (recommended vars) on test")
```

    15.262015123521905



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
  <th>Dep. Variable:</th>            <td>y</td>        <th>  R-squared:         </th> <td>   0.969</td> 
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th> <td>   0.968</td> 
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th> <td>   707.8</td> 
</tr>
<tr>
  <th>Date:</th>             <td>Thu, 25 Jul 2019</td> <th>  Prob (F-statistic):</th>  <td>  0.00</td>  
</tr>
<tr>
  <th>Time:</th>                 <td>16:09:07</td>     <th>  Log-Likelihood:    </th> <td> -6641.2</td> 
</tr>
<tr>
  <th>No. Observations:</th>      <td>  2570</td>      <th>  AIC:               </th> <td>1.350e+04</td>
</tr>
<tr>
  <th>Df Residuals:</th>          <td>  2459</td>      <th>  BIC:               </th> <td>1.415e+04</td>
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
  <th>const</th>                <td>   -0.2536</td> <td>    0.065</td> <td>   -3.922</td> <td> 0.000</td> <td>   -0.380</td> <td>   -0.127</td>
</tr>
<tr>
  <th>var_0_impact_code</th>    <td>    0.0489</td> <td>    0.008</td> <td>    6.165</td> <td> 0.000</td> <td>    0.033</td> <td>    0.064</td>
</tr>
<tr>
  <th>var_1_impact_code</th>    <td>    0.0747</td> <td>    0.008</td> <td>    9.475</td> <td> 0.000</td> <td>    0.059</td> <td>    0.090</td>
</tr>
<tr>
  <th>var_2_impact_code</th>    <td>    0.0541</td> <td>    0.008</td> <td>    6.767</td> <td> 0.000</td> <td>    0.038</td> <td>    0.070</td>
</tr>
<tr>
  <th>var_3_impact_code</th>    <td>    0.0800</td> <td>    0.008</td> <td>   10.088</td> <td> 0.000</td> <td>    0.064</td> <td>    0.096</td>
</tr>
<tr>
  <th>var_4_impact_code</th>    <td>    0.0519</td> <td>    0.008</td> <td>    6.609</td> <td> 0.000</td> <td>    0.037</td> <td>    0.067</td>
</tr>
<tr>
  <th>var_5_impact_code</th>    <td>    0.0729</td> <td>    0.008</td> <td>    8.946</td> <td> 0.000</td> <td>    0.057</td> <td>    0.089</td>
</tr>
<tr>
  <th>var_6_impact_code</th>    <td>    0.0570</td> <td>    0.008</td> <td>    6.826</td> <td> 0.000</td> <td>    0.041</td> <td>    0.073</td>
</tr>
<tr>
  <th>var_7_impact_code</th>    <td>    0.0712</td> <td>    0.008</td> <td>    8.917</td> <td> 0.000</td> <td>    0.056</td> <td>    0.087</td>
</tr>
<tr>
  <th>var_8_impact_code</th>    <td>    0.0612</td> <td>    0.008</td> <td>    7.582</td> <td> 0.000</td> <td>    0.045</td> <td>    0.077</td>
</tr>
<tr>
  <th>var_9_impact_code</th>    <td>    0.0697</td> <td>    0.008</td> <td>    8.442</td> <td> 0.000</td> <td>    0.054</td> <td>    0.086</td>
</tr>
<tr>
  <th>noise_0_impact_code</th>  <td>    0.0427</td> <td>    0.009</td> <td>    4.743</td> <td> 0.000</td> <td>    0.025</td> <td>    0.060</td>
</tr>
<tr>
  <th>noise_1_impact_code</th>  <td>    0.0262</td> <td>    0.009</td> <td>    2.841</td> <td> 0.005</td> <td>    0.008</td> <td>    0.044</td>
</tr>
<tr>
  <th>noise_2_impact_code</th>  <td>    0.0469</td> <td>    0.009</td> <td>    5.042</td> <td> 0.000</td> <td>    0.029</td> <td>    0.065</td>
</tr>
<tr>
  <th>noise_3_impact_code</th>  <td>    0.0365</td> <td>    0.009</td> <td>    3.952</td> <td> 0.000</td> <td>    0.018</td> <td>    0.055</td>
</tr>
<tr>
  <th>noise_4_impact_code</th>  <td>    0.0400</td> <td>    0.009</td> <td>    4.415</td> <td> 0.000</td> <td>    0.022</td> <td>    0.058</td>
</tr>
<tr>
  <th>noise_5_impact_code</th>  <td>    0.0528</td> <td>    0.009</td> <td>    5.906</td> <td> 0.000</td> <td>    0.035</td> <td>    0.070</td>
</tr>
<tr>
  <th>noise_6_impact_code</th>  <td>    0.0468</td> <td>    0.009</td> <td>    5.066</td> <td> 0.000</td> <td>    0.029</td> <td>    0.065</td>
</tr>
<tr>
  <th>noise_7_impact_code</th>  <td>    0.0401</td> <td>    0.009</td> <td>    4.498</td> <td> 0.000</td> <td>    0.023</td> <td>    0.058</td>
</tr>
<tr>
  <th>noise_8_impact_code</th>  <td>    0.0407</td> <td>    0.009</td> <td>    4.515</td> <td> 0.000</td> <td>    0.023</td> <td>    0.058</td>
</tr>
<tr>
  <th>noise_9_impact_code</th>  <td>    0.0416</td> <td>    0.009</td> <td>    4.633</td> <td> 0.000</td> <td>    0.024</td> <td>    0.059</td>
</tr>
<tr>
  <th>noise_10_impact_code</th> <td>    0.0449</td> <td>    0.009</td> <td>    4.988</td> <td> 0.000</td> <td>    0.027</td> <td>    0.063</td>
</tr>
<tr>
  <th>noise_11_impact_code</th> <td>    0.0235</td> <td>    0.009</td> <td>    2.580</td> <td> 0.010</td> <td>    0.006</td> <td>    0.041</td>
</tr>
<tr>
  <th>noise_12_impact_code</th> <td>    0.0538</td> <td>    0.009</td> <td>    6.017</td> <td> 0.000</td> <td>    0.036</td> <td>    0.071</td>
</tr>
<tr>
  <th>noise_13_impact_code</th> <td>    0.0353</td> <td>    0.009</td> <td>    3.840</td> <td> 0.000</td> <td>    0.017</td> <td>    0.053</td>
</tr>
<tr>
  <th>noise_14_impact_code</th> <td>    0.0432</td> <td>    0.009</td> <td>    4.700</td> <td> 0.000</td> <td>    0.025</td> <td>    0.061</td>
</tr>
<tr>
  <th>noise_15_impact_code</th> <td>    0.0378</td> <td>    0.009</td> <td>    4.339</td> <td> 0.000</td> <td>    0.021</td> <td>    0.055</td>
</tr>
<tr>
  <th>noise_16_impact_code</th> <td>    0.0209</td> <td>    0.009</td> <td>    2.338</td> <td> 0.019</td> <td>    0.003</td> <td>    0.038</td>
</tr>
<tr>
  <th>noise_17_impact_code</th> <td>    0.0482</td> <td>    0.009</td> <td>    5.289</td> <td> 0.000</td> <td>    0.030</td> <td>    0.066</td>
</tr>
<tr>
  <th>noise_18_impact_code</th> <td>    0.0366</td> <td>    0.009</td> <td>    4.152</td> <td> 0.000</td> <td>    0.019</td> <td>    0.054</td>
</tr>
<tr>
  <th>noise_19_impact_code</th> <td>    0.0325</td> <td>    0.009</td> <td>    3.434</td> <td> 0.001</td> <td>    0.014</td> <td>    0.051</td>
</tr>
<tr>
  <th>noise_20_impact_code</th> <td>    0.0577</td> <td>    0.009</td> <td>    6.282</td> <td> 0.000</td> <td>    0.040</td> <td>    0.076</td>
</tr>
<tr>
  <th>noise_21_impact_code</th> <td>    0.0497</td> <td>    0.009</td> <td>    5.355</td> <td> 0.000</td> <td>    0.031</td> <td>    0.068</td>
</tr>
<tr>
  <th>noise_22_impact_code</th> <td>    0.0384</td> <td>    0.009</td> <td>    4.195</td> <td> 0.000</td> <td>    0.020</td> <td>    0.056</td>
</tr>
<tr>
  <th>noise_23_impact_code</th> <td>    0.0418</td> <td>    0.009</td> <td>    4.657</td> <td> 0.000</td> <td>    0.024</td> <td>    0.059</td>
</tr>
<tr>
  <th>noise_24_impact_code</th> <td>    0.0271</td> <td>    0.009</td> <td>    2.926</td> <td> 0.003</td> <td>    0.009</td> <td>    0.045</td>
</tr>
<tr>
  <th>noise_25_impact_code</th> <td>    0.0462</td> <td>    0.009</td> <td>    5.066</td> <td> 0.000</td> <td>    0.028</td> <td>    0.064</td>
</tr>
<tr>
  <th>noise_26_impact_code</th> <td>    0.0454</td> <td>    0.009</td> <td>    5.205</td> <td> 0.000</td> <td>    0.028</td> <td>    0.062</td>
</tr>
<tr>
  <th>noise_27_impact_code</th> <td>    0.0441</td> <td>    0.009</td> <td>    4.835</td> <td> 0.000</td> <td>    0.026</td> <td>    0.062</td>
</tr>
<tr>
  <th>noise_28_impact_code</th> <td>    0.0365</td> <td>    0.009</td> <td>    4.168</td> <td> 0.000</td> <td>    0.019</td> <td>    0.054</td>
</tr>
<tr>
  <th>noise_29_impact_code</th> <td>    0.0503</td> <td>    0.009</td> <td>    5.758</td> <td> 0.000</td> <td>    0.033</td> <td>    0.067</td>
</tr>
<tr>
  <th>noise_30_impact_code</th> <td>    0.0542</td> <td>    0.009</td> <td>    5.937</td> <td> 0.000</td> <td>    0.036</td> <td>    0.072</td>
</tr>
<tr>
  <th>noise_31_impact_code</th> <td>    0.0514</td> <td>    0.009</td> <td>    5.841</td> <td> 0.000</td> <td>    0.034</td> <td>    0.069</td>
</tr>
<tr>
  <th>noise_32_impact_code</th> <td>    0.0563</td> <td>    0.009</td> <td>    5.948</td> <td> 0.000</td> <td>    0.038</td> <td>    0.075</td>
</tr>
<tr>
  <th>noise_33_impact_code</th> <td>    0.0436</td> <td>    0.009</td> <td>    4.684</td> <td> 0.000</td> <td>    0.025</td> <td>    0.062</td>
</tr>
<tr>
  <th>noise_34_impact_code</th> <td>    0.0423</td> <td>    0.009</td> <td>    4.730</td> <td> 0.000</td> <td>    0.025</td> <td>    0.060</td>
</tr>
<tr>
  <th>noise_35_impact_code</th> <td>    0.0337</td> <td>    0.009</td> <td>    3.720</td> <td> 0.000</td> <td>    0.016</td> <td>    0.051</td>
</tr>
<tr>
  <th>noise_36_impact_code</th> <td>    0.0281</td> <td>    0.009</td> <td>    3.145</td> <td> 0.002</td> <td>    0.011</td> <td>    0.046</td>
</tr>
<tr>
  <th>noise_37_impact_code</th> <td>    0.0259</td> <td>    0.009</td> <td>    2.782</td> <td> 0.005</td> <td>    0.008</td> <td>    0.044</td>
</tr>
<tr>
  <th>noise_38_impact_code</th> <td>    0.0352</td> <td>    0.009</td> <td>    3.800</td> <td> 0.000</td> <td>    0.017</td> <td>    0.053</td>
</tr>
<tr>
  <th>noise_39_impact_code</th> <td>    0.0332</td> <td>    0.009</td> <td>    3.546</td> <td> 0.000</td> <td>    0.015</td> <td>    0.052</td>
</tr>
<tr>
  <th>noise_40_impact_code</th> <td>    0.0426</td> <td>    0.009</td> <td>    4.670</td> <td> 0.000</td> <td>    0.025</td> <td>    0.060</td>
</tr>
<tr>
  <th>noise_41_impact_code</th> <td>    0.0507</td> <td>    0.009</td> <td>    5.664</td> <td> 0.000</td> <td>    0.033</td> <td>    0.068</td>
</tr>
<tr>
  <th>noise_42_impact_code</th> <td>    0.0466</td> <td>    0.009</td> <td>    5.286</td> <td> 0.000</td> <td>    0.029</td> <td>    0.064</td>
</tr>
<tr>
  <th>noise_43_impact_code</th> <td>    0.0282</td> <td>    0.009</td> <td>    3.044</td> <td> 0.002</td> <td>    0.010</td> <td>    0.046</td>
</tr>
<tr>
  <th>noise_44_impact_code</th> <td>    0.0445</td> <td>    0.009</td> <td>    4.941</td> <td> 0.000</td> <td>    0.027</td> <td>    0.062</td>
</tr>
<tr>
  <th>noise_45_impact_code</th> <td>    0.0410</td> <td>    0.009</td> <td>    4.503</td> <td> 0.000</td> <td>    0.023</td> <td>    0.059</td>
</tr>
<tr>
  <th>noise_46_impact_code</th> <td>    0.0411</td> <td>    0.009</td> <td>    4.569</td> <td> 0.000</td> <td>    0.023</td> <td>    0.059</td>
</tr>
<tr>
  <th>noise_47_impact_code</th> <td>    0.0281</td> <td>    0.009</td> <td>    3.112</td> <td> 0.002</td> <td>    0.010</td> <td>    0.046</td>
</tr>
<tr>
  <th>noise_48_impact_code</th> <td>    0.0375</td> <td>    0.009</td> <td>    4.047</td> <td> 0.000</td> <td>    0.019</td> <td>    0.056</td>
</tr>
<tr>
  <th>noise_49_impact_code</th> <td>    0.0332</td> <td>    0.009</td> <td>    3.685</td> <td> 0.000</td> <td>    0.016</td> <td>    0.051</td>
</tr>
<tr>
  <th>noise_50_impact_code</th> <td>    0.0556</td> <td>    0.009</td> <td>    6.164</td> <td> 0.000</td> <td>    0.038</td> <td>    0.073</td>
</tr>
<tr>
  <th>noise_51_impact_code</th> <td>    0.0510</td> <td>    0.009</td> <td>    5.548</td> <td> 0.000</td> <td>    0.033</td> <td>    0.069</td>
</tr>
<tr>
  <th>noise_52_impact_code</th> <td>    0.0394</td> <td>    0.009</td> <td>    4.167</td> <td> 0.000</td> <td>    0.021</td> <td>    0.058</td>
</tr>
<tr>
  <th>noise_53_impact_code</th> <td>    0.0587</td> <td>    0.009</td> <td>    6.523</td> <td> 0.000</td> <td>    0.041</td> <td>    0.076</td>
</tr>
<tr>
  <th>noise_54_impact_code</th> <td>    0.0537</td> <td>    0.010</td> <td>    5.628</td> <td> 0.000</td> <td>    0.035</td> <td>    0.072</td>
</tr>
<tr>
  <th>noise_55_impact_code</th> <td>    0.0402</td> <td>    0.009</td> <td>    4.428</td> <td> 0.000</td> <td>    0.022</td> <td>    0.058</td>
</tr>
<tr>
  <th>noise_56_impact_code</th> <td>    0.0487</td> <td>    0.009</td> <td>    5.448</td> <td> 0.000</td> <td>    0.031</td> <td>    0.066</td>
</tr>
<tr>
  <th>noise_57_impact_code</th> <td>    0.0312</td> <td>    0.009</td> <td>    3.389</td> <td> 0.001</td> <td>    0.013</td> <td>    0.049</td>
</tr>
<tr>
  <th>noise_58_impact_code</th> <td>    0.0532</td> <td>    0.009</td> <td>    5.793</td> <td> 0.000</td> <td>    0.035</td> <td>    0.071</td>
</tr>
<tr>
  <th>noise_59_impact_code</th> <td>    0.0468</td> <td>    0.009</td> <td>    5.291</td> <td> 0.000</td> <td>    0.029</td> <td>    0.064</td>
</tr>
<tr>
  <th>noise_60_impact_code</th> <td>    0.0489</td> <td>    0.009</td> <td>    5.420</td> <td> 0.000</td> <td>    0.031</td> <td>    0.067</td>
</tr>
<tr>
  <th>noise_61_impact_code</th> <td>    0.0301</td> <td>    0.009</td> <td>    3.296</td> <td> 0.001</td> <td>    0.012</td> <td>    0.048</td>
</tr>
<tr>
  <th>noise_62_impact_code</th> <td>    0.0407</td> <td>    0.009</td> <td>    4.367</td> <td> 0.000</td> <td>    0.022</td> <td>    0.059</td>
</tr>
<tr>
  <th>noise_63_impact_code</th> <td>    0.0432</td> <td>    0.009</td> <td>    4.717</td> <td> 0.000</td> <td>    0.025</td> <td>    0.061</td>
</tr>
<tr>
  <th>noise_64_impact_code</th> <td>    0.0159</td> <td>    0.009</td> <td>    1.755</td> <td> 0.079</td> <td>   -0.002</td> <td>    0.034</td>
</tr>
<tr>
  <th>noise_65_impact_code</th> <td>    0.0468</td> <td>    0.009</td> <td>    5.159</td> <td> 0.000</td> <td>    0.029</td> <td>    0.065</td>
</tr>
<tr>
  <th>noise_66_impact_code</th> <td>    0.0362</td> <td>    0.009</td> <td>    4.014</td> <td> 0.000</td> <td>    0.019</td> <td>    0.054</td>
</tr>
<tr>
  <th>noise_67_impact_code</th> <td>    0.0415</td> <td>    0.009</td> <td>    4.412</td> <td> 0.000</td> <td>    0.023</td> <td>    0.060</td>
</tr>
<tr>
  <th>noise_68_impact_code</th> <td>    0.0376</td> <td>    0.009</td> <td>    4.158</td> <td> 0.000</td> <td>    0.020</td> <td>    0.055</td>
</tr>
<tr>
  <th>noise_69_impact_code</th> <td>    0.0517</td> <td>    0.009</td> <td>    5.775</td> <td> 0.000</td> <td>    0.034</td> <td>    0.069</td>
</tr>
<tr>
  <th>noise_70_impact_code</th> <td>    0.0424</td> <td>    0.009</td> <td>    4.603</td> <td> 0.000</td> <td>    0.024</td> <td>    0.060</td>
</tr>
<tr>
  <th>noise_71_impact_code</th> <td>    0.0420</td> <td>    0.009</td> <td>    4.529</td> <td> 0.000</td> <td>    0.024</td> <td>    0.060</td>
</tr>
<tr>
  <th>noise_72_impact_code</th> <td>    0.0246</td> <td>    0.009</td> <td>    2.668</td> <td> 0.008</td> <td>    0.007</td> <td>    0.043</td>
</tr>
<tr>
  <th>noise_73_impact_code</th> <td>    0.0511</td> <td>    0.009</td> <td>    5.666</td> <td> 0.000</td> <td>    0.033</td> <td>    0.069</td>
</tr>
<tr>
  <th>noise_74_impact_code</th> <td>    0.0514</td> <td>    0.009</td> <td>    5.532</td> <td> 0.000</td> <td>    0.033</td> <td>    0.070</td>
</tr>
<tr>
  <th>noise_75_impact_code</th> <td>    0.0281</td> <td>    0.009</td> <td>    3.158</td> <td> 0.002</td> <td>    0.011</td> <td>    0.046</td>
</tr>
<tr>
  <th>noise_76_impact_code</th> <td>    0.0212</td> <td>    0.009</td> <td>    2.249</td> <td> 0.025</td> <td>    0.003</td> <td>    0.040</td>
</tr>
<tr>
  <th>noise_77_impact_code</th> <td>    0.0477</td> <td>    0.009</td> <td>    5.519</td> <td> 0.000</td> <td>    0.031</td> <td>    0.065</td>
</tr>
<tr>
  <th>noise_78_impact_code</th> <td>    0.0434</td> <td>    0.009</td> <td>    4.799</td> <td> 0.000</td> <td>    0.026</td> <td>    0.061</td>
</tr>
<tr>
  <th>noise_79_impact_code</th> <td>    0.0409</td> <td>    0.009</td> <td>    4.528</td> <td> 0.000</td> <td>    0.023</td> <td>    0.059</td>
</tr>
<tr>
  <th>noise_80_impact_code</th> <td>    0.0601</td> <td>    0.009</td> <td>    6.736</td> <td> 0.000</td> <td>    0.043</td> <td>    0.078</td>
</tr>
<tr>
  <th>noise_81_impact_code</th> <td>    0.0267</td> <td>    0.009</td> <td>    2.920</td> <td> 0.004</td> <td>    0.009</td> <td>    0.045</td>
</tr>
<tr>
  <th>noise_82_impact_code</th> <td>    0.0425</td> <td>    0.009</td> <td>    4.720</td> <td> 0.000</td> <td>    0.025</td> <td>    0.060</td>
</tr>
<tr>
  <th>noise_83_impact_code</th> <td>    0.0474</td> <td>    0.009</td> <td>    5.302</td> <td> 0.000</td> <td>    0.030</td> <td>    0.065</td>
</tr>
<tr>
  <th>noise_84_impact_code</th> <td>    0.0344</td> <td>    0.010</td> <td>    3.589</td> <td> 0.000</td> <td>    0.016</td> <td>    0.053</td>
</tr>
<tr>
  <th>noise_85_impact_code</th> <td>    0.0353</td> <td>    0.009</td> <td>    3.960</td> <td> 0.000</td> <td>    0.018</td> <td>    0.053</td>
</tr>
<tr>
  <th>noise_86_impact_code</th> <td>    0.0582</td> <td>    0.009</td> <td>    6.468</td> <td> 0.000</td> <td>    0.041</td> <td>    0.076</td>
</tr>
<tr>
  <th>noise_87_impact_code</th> <td>    0.0257</td> <td>    0.009</td> <td>    2.812</td> <td> 0.005</td> <td>    0.008</td> <td>    0.044</td>
</tr>
<tr>
  <th>noise_88_impact_code</th> <td>    0.0458</td> <td>    0.009</td> <td>    4.914</td> <td> 0.000</td> <td>    0.028</td> <td>    0.064</td>
</tr>
<tr>
  <th>noise_89_impact_code</th> <td>    0.0469</td> <td>    0.010</td> <td>    4.917</td> <td> 0.000</td> <td>    0.028</td> <td>    0.066</td>
</tr>
<tr>
  <th>noise_90_impact_code</th> <td>    0.0451</td> <td>    0.009</td> <td>    4.922</td> <td> 0.000</td> <td>    0.027</td> <td>    0.063</td>
</tr>
<tr>
  <th>noise_91_impact_code</th> <td>    0.0349</td> <td>    0.009</td> <td>    3.739</td> <td> 0.000</td> <td>    0.017</td> <td>    0.053</td>
</tr>
<tr>
  <th>noise_92_impact_code</th> <td>    0.0338</td> <td>    0.009</td> <td>    3.804</td> <td> 0.000</td> <td>    0.016</td> <td>    0.051</td>
</tr>
<tr>
  <th>noise_93_impact_code</th> <td>    0.0300</td> <td>    0.010</td> <td>    3.129</td> <td> 0.002</td> <td>    0.011</td> <td>    0.049</td>
</tr>
<tr>
  <th>noise_94_impact_code</th> <td>    0.0417</td> <td>    0.009</td> <td>    4.661</td> <td> 0.000</td> <td>    0.024</td> <td>    0.059</td>
</tr>
<tr>
  <th>noise_95_impact_code</th> <td>    0.0345</td> <td>    0.009</td> <td>    3.771</td> <td> 0.000</td> <td>    0.017</td> <td>    0.052</td>
</tr>
<tr>
  <th>noise_96_impact_code</th> <td>    0.0529</td> <td>    0.009</td> <td>    5.875</td> <td> 0.000</td> <td>    0.035</td> <td>    0.071</td>
</tr>
<tr>
  <th>noise_97_impact_code</th> <td>    0.0380</td> <td>    0.009</td> <td>    4.089</td> <td> 0.000</td> <td>    0.020</td> <td>    0.056</td>
</tr>
<tr>
  <th>noise_98_impact_code</th> <td>    0.0448</td> <td>    0.009</td> <td>    4.734</td> <td> 0.000</td> <td>    0.026</td> <td>    0.063</td>
</tr>
<tr>
  <th>noise_99_impact_code</th> <td>    0.0384</td> <td>    0.009</td> <td>    4.106</td> <td> 0.000</td> <td>    0.020</td> <td>    0.057</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td> 0.241</td> <th>  Durbin-Watson:     </th> <td>   1.991</td>
</tr>
<tr>
  <th>Prob(Omnibus):</th> <td> 0.886</td> <th>  Jarque-Bera (JB):  </th> <td>   0.214</td>
</tr>
<tr>
  <th>Skew:</th>          <td>-0.021</td> <th>  Prob(JB):          </th> <td>   0.899</td>
</tr>
<tr>
  <th>Kurtosis:</th>      <td> 3.014</td> <th>  Cond. No.          </th> <td>    39.5</td>
</tr>
</table><br/><br/>Warnings:<br/>[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.




```python
sklearn.metrics.r2_score(plot_train["y"],plot_train["predict_naive_empirical_all_vars"])
```




    0.9693835710295952




```python
sklearn.metrics.r2_score(plot_test["y"],plot_test["predict_naive_empirical_all_vars"])
```




    0.05366921439362193




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
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th> <td>   0.260</td> 
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th> <td>   9.192</td> 
</tr>
<tr>
  <th>Date:</th>             <td>Thu, 25 Jul 2019</td> <th>  Prob (F-statistic):</th> <td>1.64e-117</td>
</tr>
<tr>
  <th>Time:</th>                 <td>16:09:08</td>     <th>  Log-Likelihood:    </th> <td> -10678.</td> 
</tr>
<tr>
  <th>No. Observations:</th>      <td>  2570</td>      <th>  AIC:               </th> <td>2.158e+04</td>
</tr>
<tr>
  <th>Df Residuals:</th>          <td>  2459</td>      <th>  BIC:               </th> <td>2.223e+04</td>
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
  <th>const</th>                <td>   -0.1245</td> <td>    0.317</td> <td>   -0.393</td> <td> 0.694</td> <td>   -0.745</td> <td>    0.496</td>
</tr>
<tr>
  <th>var_0_impact_code</th>    <td>    0.4625</td> <td>    0.046</td> <td>   10.045</td> <td> 0.000</td> <td>    0.372</td> <td>    0.553</td>
</tr>
<tr>
  <th>var_1_impact_code</th>    <td>    0.4429</td> <td>    0.047</td> <td>    9.402</td> <td> 0.000</td> <td>    0.351</td> <td>    0.535</td>
</tr>
<tr>
  <th>var_2_impact_code</th>    <td>    0.3366</td> <td>    0.046</td> <td>    7.287</td> <td> 0.000</td> <td>    0.246</td> <td>    0.427</td>
</tr>
<tr>
  <th>var_3_impact_code</th>    <td>    0.4816</td> <td>    0.047</td> <td>   10.290</td> <td> 0.000</td> <td>    0.390</td> <td>    0.573</td>
</tr>
<tr>
  <th>var_4_impact_code</th>    <td>    0.4732</td> <td>    0.047</td> <td>   10.057</td> <td> 0.000</td> <td>    0.381</td> <td>    0.566</td>
</tr>
<tr>
  <th>var_5_impact_code</th>    <td>    0.3946</td> <td>    0.048</td> <td>    8.176</td> <td> 0.000</td> <td>    0.300</td> <td>    0.489</td>
</tr>
<tr>
  <th>var_6_impact_code</th>    <td>    0.3712</td> <td>    0.052</td> <td>    7.079</td> <td> 0.000</td> <td>    0.268</td> <td>    0.474</td>
</tr>
<tr>
  <th>var_7_impact_code</th>    <td>    0.4735</td> <td>    0.047</td> <td>   10.153</td> <td> 0.000</td> <td>    0.382</td> <td>    0.565</td>
</tr>
<tr>
  <th>var_8_impact_code</th>    <td>    0.3970</td> <td>    0.048</td> <td>    8.226</td> <td> 0.000</td> <td>    0.302</td> <td>    0.492</td>
</tr>
<tr>
  <th>var_9_impact_code</th>    <td>    0.3714</td> <td>    0.049</td> <td>    7.651</td> <td> 0.000</td> <td>    0.276</td> <td>    0.467</td>
</tr>
<tr>
  <th>noise_0_impact_code</th>  <td>    0.0663</td> <td>    0.056</td> <td>    1.182</td> <td> 0.237</td> <td>   -0.044</td> <td>    0.176</td>
</tr>
<tr>
  <th>noise_1_impact_code</th>  <td>   -0.0177</td> <td>    0.058</td> <td>   -0.307</td> <td> 0.759</td> <td>   -0.131</td> <td>    0.095</td>
</tr>
<tr>
  <th>noise_2_impact_code</th>  <td>   -0.0862</td> <td>    0.060</td> <td>   -1.445</td> <td> 0.149</td> <td>   -0.203</td> <td>    0.031</td>
</tr>
<tr>
  <th>noise_3_impact_code</th>  <td>   -0.0272</td> <td>    0.058</td> <td>   -0.468</td> <td> 0.640</td> <td>   -0.141</td> <td>    0.087</td>
</tr>
<tr>
  <th>noise_4_impact_code</th>  <td>    0.0094</td> <td>    0.058</td> <td>    0.162</td> <td> 0.871</td> <td>   -0.105</td> <td>    0.123</td>
</tr>
<tr>
  <th>noise_5_impact_code</th>  <td>    0.0078</td> <td>    0.057</td> <td>    0.137</td> <td> 0.891</td> <td>   -0.104</td> <td>    0.119</td>
</tr>
<tr>
  <th>noise_6_impact_code</th>  <td>    0.0567</td> <td>    0.059</td> <td>    0.957</td> <td> 0.339</td> <td>   -0.059</td> <td>    0.173</td>
</tr>
<tr>
  <th>noise_7_impact_code</th>  <td>    0.0005</td> <td>    0.057</td> <td>    0.009</td> <td> 0.993</td> <td>   -0.111</td> <td>    0.112</td>
</tr>
<tr>
  <th>noise_8_impact_code</th>  <td>    0.0857</td> <td>    0.060</td> <td>    1.436</td> <td> 0.151</td> <td>   -0.031</td> <td>    0.203</td>
</tr>
<tr>
  <th>noise_9_impact_code</th>  <td>   -0.0472</td> <td>    0.056</td> <td>   -0.848</td> <td> 0.397</td> <td>   -0.156</td> <td>    0.062</td>
</tr>
<tr>
  <th>noise_10_impact_code</th> <td>    0.0488</td> <td>    0.056</td> <td>    0.871</td> <td> 0.384</td> <td>   -0.061</td> <td>    0.159</td>
</tr>
<tr>
  <th>noise_11_impact_code</th> <td>    0.0319</td> <td>    0.055</td> <td>    0.580</td> <td> 0.562</td> <td>   -0.076</td> <td>    0.140</td>
</tr>
<tr>
  <th>noise_12_impact_code</th> <td>    0.0313</td> <td>    0.055</td> <td>    0.569</td> <td> 0.569</td> <td>   -0.076</td> <td>    0.139</td>
</tr>
<tr>
  <th>noise_13_impact_code</th> <td>    0.0353</td> <td>    0.061</td> <td>    0.577</td> <td> 0.564</td> <td>   -0.085</td> <td>    0.155</td>
</tr>
<tr>
  <th>noise_14_impact_code</th> <td>   -0.0564</td> <td>    0.059</td> <td>   -0.961</td> <td> 0.337</td> <td>   -0.171</td> <td>    0.059</td>
</tr>
<tr>
  <th>noise_15_impact_code</th> <td>    0.0555</td> <td>    0.057</td> <td>    0.975</td> <td> 0.330</td> <td>   -0.056</td> <td>    0.167</td>
</tr>
<tr>
  <th>noise_16_impact_code</th> <td>    0.0396</td> <td>    0.060</td> <td>    0.655</td> <td> 0.512</td> <td>   -0.079</td> <td>    0.158</td>
</tr>
<tr>
  <th>noise_17_impact_code</th> <td>   -0.0097</td> <td>    0.057</td> <td>   -0.171</td> <td> 0.865</td> <td>   -0.122</td> <td>    0.102</td>
</tr>
<tr>
  <th>noise_18_impact_code</th> <td>    0.1035</td> <td>    0.058</td> <td>    1.784</td> <td> 0.075</td> <td>   -0.010</td> <td>    0.217</td>
</tr>
<tr>
  <th>noise_19_impact_code</th> <td>   -0.1945</td> <td>    0.059</td> <td>   -3.302</td> <td> 0.001</td> <td>   -0.310</td> <td>   -0.079</td>
</tr>
<tr>
  <th>noise_20_impact_code</th> <td>    0.0286</td> <td>    0.058</td> <td>    0.494</td> <td> 0.621</td> <td>   -0.085</td> <td>    0.142</td>
</tr>
<tr>
  <th>noise_21_impact_code</th> <td>   -0.0262</td> <td>    0.059</td> <td>   -0.446</td> <td> 0.655</td> <td>   -0.141</td> <td>    0.089</td>
</tr>
<tr>
  <th>noise_22_impact_code</th> <td>   -0.0575</td> <td>    0.060</td> <td>   -0.960</td> <td> 0.337</td> <td>   -0.175</td> <td>    0.060</td>
</tr>
<tr>
  <th>noise_23_impact_code</th> <td>   -0.0126</td> <td>    0.060</td> <td>   -0.209</td> <td> 0.835</td> <td>   -0.131</td> <td>    0.106</td>
</tr>
<tr>
  <th>noise_24_impact_code</th> <td>   -0.0087</td> <td>    0.059</td> <td>   -0.147</td> <td> 0.883</td> <td>   -0.125</td> <td>    0.108</td>
</tr>
<tr>
  <th>noise_25_impact_code</th> <td>   -0.0150</td> <td>    0.059</td> <td>   -0.252</td> <td> 0.801</td> <td>   -0.131</td> <td>    0.101</td>
</tr>
<tr>
  <th>noise_26_impact_code</th> <td>    0.0334</td> <td>    0.054</td> <td>    0.616</td> <td> 0.538</td> <td>   -0.073</td> <td>    0.140</td>
</tr>
<tr>
  <th>noise_27_impact_code</th> <td>   -0.0908</td> <td>    0.060</td> <td>   -1.525</td> <td> 0.127</td> <td>   -0.208</td> <td>    0.026</td>
</tr>
<tr>
  <th>noise_28_impact_code</th> <td>    0.1067</td> <td>    0.054</td> <td>    1.977</td> <td> 0.048</td> <td>    0.001</td> <td>    0.213</td>
</tr>
<tr>
  <th>noise_29_impact_code</th> <td>    0.0923</td> <td>    0.056</td> <td>    1.653</td> <td> 0.098</td> <td>   -0.017</td> <td>    0.202</td>
</tr>
<tr>
  <th>noise_30_impact_code</th> <td>   -0.0312</td> <td>    0.060</td> <td>   -0.518</td> <td> 0.604</td> <td>   -0.149</td> <td>    0.087</td>
</tr>
<tr>
  <th>noise_31_impact_code</th> <td>    0.0839</td> <td>    0.053</td> <td>    1.578</td> <td> 0.115</td> <td>   -0.020</td> <td>    0.188</td>
</tr>
<tr>
  <th>noise_32_impact_code</th> <td>   -0.0808</td> <td>    0.063</td> <td>   -1.290</td> <td> 0.197</td> <td>   -0.204</td> <td>    0.042</td>
</tr>
<tr>
  <th>noise_33_impact_code</th> <td>   -0.1007</td> <td>    0.061</td> <td>   -1.655</td> <td> 0.098</td> <td>   -0.220</td> <td>    0.019</td>
</tr>
<tr>
  <th>noise_34_impact_code</th> <td>   -0.0508</td> <td>    0.059</td> <td>   -0.864</td> <td> 0.387</td> <td>   -0.166</td> <td>    0.064</td>
</tr>
<tr>
  <th>noise_35_impact_code</th> <td>    0.0510</td> <td>    0.057</td> <td>    0.887</td> <td> 0.375</td> <td>   -0.062</td> <td>    0.164</td>
</tr>
<tr>
  <th>noise_36_impact_code</th> <td>   -0.0247</td> <td>    0.059</td> <td>   -0.420</td> <td> 0.675</td> <td>   -0.140</td> <td>    0.090</td>
</tr>
<tr>
  <th>noise_37_impact_code</th> <td>   -0.0668</td> <td>    0.059</td> <td>   -1.132</td> <td> 0.258</td> <td>   -0.182</td> <td>    0.049</td>
</tr>
<tr>
  <th>noise_38_impact_code</th> <td>    0.0370</td> <td>    0.062</td> <td>    0.599</td> <td> 0.549</td> <td>   -0.084</td> <td>    0.158</td>
</tr>
<tr>
  <th>noise_39_impact_code</th> <td>   -0.0184</td> <td>    0.058</td> <td>   -0.316</td> <td> 0.752</td> <td>   -0.132</td> <td>    0.096</td>
</tr>
<tr>
  <th>noise_40_impact_code</th> <td>   -0.0423</td> <td>    0.058</td> <td>   -0.727</td> <td> 0.467</td> <td>   -0.156</td> <td>    0.072</td>
</tr>
<tr>
  <th>noise_41_impact_code</th> <td>    0.0137</td> <td>    0.056</td> <td>    0.246</td> <td> 0.806</td> <td>   -0.096</td> <td>    0.123</td>
</tr>
<tr>
  <th>noise_42_impact_code</th> <td>    0.1485</td> <td>    0.056</td> <td>    2.661</td> <td> 0.008</td> <td>    0.039</td> <td>    0.258</td>
</tr>
<tr>
  <th>noise_43_impact_code</th> <td>    0.0372</td> <td>    0.059</td> <td>    0.629</td> <td> 0.530</td> <td>   -0.079</td> <td>    0.153</td>
</tr>
<tr>
  <th>noise_44_impact_code</th> <td>   -0.0086</td> <td>    0.055</td> <td>   -0.155</td> <td> 0.877</td> <td>   -0.117</td> <td>    0.100</td>
</tr>
<tr>
  <th>noise_45_impact_code</th> <td>    0.0057</td> <td>    0.058</td> <td>    0.099</td> <td> 0.921</td> <td>   -0.108</td> <td>    0.119</td>
</tr>
<tr>
  <th>noise_46_impact_code</th> <td>    0.0065</td> <td>    0.059</td> <td>    0.110</td> <td> 0.913</td> <td>   -0.109</td> <td>    0.122</td>
</tr>
<tr>
  <th>noise_47_impact_code</th> <td>    0.0485</td> <td>    0.059</td> <td>    0.820</td> <td> 0.412</td> <td>   -0.067</td> <td>    0.164</td>
</tr>
<tr>
  <th>noise_48_impact_code</th> <td>   -0.0024</td> <td>    0.058</td> <td>   -0.040</td> <td> 0.968</td> <td>   -0.117</td> <td>    0.112</td>
</tr>
<tr>
  <th>noise_49_impact_code</th> <td>    0.0595</td> <td>    0.056</td> <td>    1.069</td> <td> 0.285</td> <td>   -0.050</td> <td>    0.169</td>
</tr>
<tr>
  <th>noise_50_impact_code</th> <td>   -0.0688</td> <td>    0.058</td> <td>   -1.187</td> <td> 0.235</td> <td>   -0.183</td> <td>    0.045</td>
</tr>
<tr>
  <th>noise_51_impact_code</th> <td>   -0.1236</td> <td>    0.060</td> <td>   -2.047</td> <td> 0.041</td> <td>   -0.242</td> <td>   -0.005</td>
</tr>
<tr>
  <th>noise_52_impact_code</th> <td>   -0.0884</td> <td>    0.061</td> <td>   -1.461</td> <td> 0.144</td> <td>   -0.207</td> <td>    0.030</td>
</tr>
<tr>
  <th>noise_53_impact_code</th> <td>   -0.0098</td> <td>    0.055</td> <td>   -0.180</td> <td> 0.857</td> <td>   -0.117</td> <td>    0.097</td>
</tr>
<tr>
  <th>noise_54_impact_code</th> <td>   -0.1709</td> <td>    0.065</td> <td>   -2.610</td> <td> 0.009</td> <td>   -0.299</td> <td>   -0.043</td>
</tr>
<tr>
  <th>noise_55_impact_code</th> <td>   -0.0643</td> <td>    0.059</td> <td>   -1.091</td> <td> 0.275</td> <td>   -0.180</td> <td>    0.051</td>
</tr>
<tr>
  <th>noise_56_impact_code</th> <td>    0.0626</td> <td>    0.058</td> <td>    1.089</td> <td> 0.276</td> <td>   -0.050</td> <td>    0.175</td>
</tr>
<tr>
  <th>noise_57_impact_code</th> <td>   -0.0231</td> <td>    0.057</td> <td>   -0.403</td> <td> 0.687</td> <td>   -0.136</td> <td>    0.089</td>
</tr>
<tr>
  <th>noise_58_impact_code</th> <td>   -0.1318</td> <td>    0.062</td> <td>   -2.138</td> <td> 0.033</td> <td>   -0.253</td> <td>   -0.011</td>
</tr>
<tr>
  <th>noise_59_impact_code</th> <td>    0.0577</td> <td>    0.056</td> <td>    1.034</td> <td> 0.301</td> <td>   -0.052</td> <td>    0.167</td>
</tr>
<tr>
  <th>noise_60_impact_code</th> <td>    0.0962</td> <td>    0.057</td> <td>    1.676</td> <td> 0.094</td> <td>   -0.016</td> <td>    0.209</td>
</tr>
<tr>
  <th>noise_61_impact_code</th> <td>   -0.0667</td> <td>    0.057</td> <td>   -1.163</td> <td> 0.245</td> <td>   -0.179</td> <td>    0.046</td>
</tr>
<tr>
  <th>noise_62_impact_code</th> <td>   -0.0862</td> <td>    0.062</td> <td>   -1.383</td> <td> 0.167</td> <td>   -0.208</td> <td>    0.036</td>
</tr>
<tr>
  <th>noise_63_impact_code</th> <td>   -0.0815</td> <td>    0.061</td> <td>   -1.327</td> <td> 0.185</td> <td>   -0.202</td> <td>    0.039</td>
</tr>
<tr>
  <th>noise_64_impact_code</th> <td>    0.0526</td> <td>    0.058</td> <td>    0.900</td> <td> 0.368</td> <td>   -0.062</td> <td>    0.167</td>
</tr>
<tr>
  <th>noise_65_impact_code</th> <td>   -0.0122</td> <td>    0.057</td> <td>   -0.213</td> <td> 0.831</td> <td>   -0.124</td> <td>    0.100</td>
</tr>
<tr>
  <th>noise_66_impact_code</th> <td>    0.0533</td> <td>    0.056</td> <td>    0.958</td> <td> 0.338</td> <td>   -0.056</td> <td>    0.162</td>
</tr>
<tr>
  <th>noise_67_impact_code</th> <td>   -0.0371</td> <td>    0.062</td> <td>   -0.603</td> <td> 0.546</td> <td>   -0.158</td> <td>    0.084</td>
</tr>
<tr>
  <th>noise_68_impact_code</th> <td>    0.0023</td> <td>    0.060</td> <td>    0.038</td> <td> 0.969</td> <td>   -0.115</td> <td>    0.119</td>
</tr>
<tr>
  <th>noise_69_impact_code</th> <td>    0.0178</td> <td>    0.059</td> <td>    0.303</td> <td> 0.762</td> <td>   -0.098</td> <td>    0.133</td>
</tr>
<tr>
  <th>noise_70_impact_code</th> <td>   -0.0203</td> <td>    0.058</td> <td>   -0.351</td> <td> 0.726</td> <td>   -0.134</td> <td>    0.093</td>
</tr>
<tr>
  <th>noise_71_impact_code</th> <td>   -0.0419</td> <td>    0.060</td> <td>   -0.696</td> <td> 0.487</td> <td>   -0.160</td> <td>    0.076</td>
</tr>
<tr>
  <th>noise_72_impact_code</th> <td>    0.0116</td> <td>    0.060</td> <td>    0.194</td> <td> 0.846</td> <td>   -0.106</td> <td>    0.129</td>
</tr>
<tr>
  <th>noise_73_impact_code</th> <td>   -0.0396</td> <td>    0.056</td> <td>   -0.703</td> <td> 0.482</td> <td>   -0.150</td> <td>    0.071</td>
</tr>
<tr>
  <th>noise_74_impact_code</th> <td>    0.0025</td> <td>    0.062</td> <td>    0.041</td> <td> 0.967</td> <td>   -0.119</td> <td>    0.124</td>
</tr>
<tr>
  <th>noise_75_impact_code</th> <td>   -0.0304</td> <td>    0.056</td> <td>   -0.544</td> <td> 0.586</td> <td>   -0.140</td> <td>    0.079</td>
</tr>
<tr>
  <th>noise_76_impact_code</th> <td>   -0.1302</td> <td>    0.063</td> <td>   -2.061</td> <td> 0.039</td> <td>   -0.254</td> <td>   -0.006</td>
</tr>
<tr>
  <th>noise_77_impact_code</th> <td>    0.1423</td> <td>    0.055</td> <td>    2.597</td> <td> 0.009</td> <td>    0.035</td> <td>    0.250</td>
</tr>
<tr>
  <th>noise_78_impact_code</th> <td>    0.0214</td> <td>    0.056</td> <td>    0.385</td> <td> 0.701</td> <td>   -0.088</td> <td>    0.131</td>
</tr>
<tr>
  <th>noise_79_impact_code</th> <td>    0.0396</td> <td>    0.057</td> <td>    0.693</td> <td> 0.488</td> <td>   -0.072</td> <td>    0.151</td>
</tr>
<tr>
  <th>noise_80_impact_code</th> <td>    0.1420</td> <td>    0.057</td> <td>    2.476</td> <td> 0.013</td> <td>    0.030</td> <td>    0.254</td>
</tr>
<tr>
  <th>noise_81_impact_code</th> <td>   -0.0695</td> <td>    0.058</td> <td>   -1.190</td> <td> 0.234</td> <td>   -0.184</td> <td>    0.045</td>
</tr>
<tr>
  <th>noise_82_impact_code</th> <td>   -0.0410</td> <td>    0.058</td> <td>   -0.712</td> <td> 0.477</td> <td>   -0.154</td> <td>    0.072</td>
</tr>
<tr>
  <th>noise_83_impact_code</th> <td>    0.0930</td> <td>    0.055</td> <td>    1.705</td> <td> 0.088</td> <td>   -0.014</td> <td>    0.200</td>
</tr>
<tr>
  <th>noise_84_impact_code</th> <td>   -0.1590</td> <td>    0.062</td> <td>   -2.553</td> <td> 0.011</td> <td>   -0.281</td> <td>   -0.037</td>
</tr>
<tr>
  <th>noise_85_impact_code</th> <td>    0.0275</td> <td>    0.059</td> <td>    0.469</td> <td> 0.639</td> <td>   -0.088</td> <td>    0.143</td>
</tr>
<tr>
  <th>noise_86_impact_code</th> <td>   -0.0613</td> <td>    0.056</td> <td>   -1.094</td> <td> 0.274</td> <td>   -0.171</td> <td>    0.049</td>
</tr>
<tr>
  <th>noise_87_impact_code</th> <td>    0.0188</td> <td>    0.057</td> <td>    0.329</td> <td> 0.742</td> <td>   -0.093</td> <td>    0.131</td>
</tr>
<tr>
  <th>noise_88_impact_code</th> <td>   -0.0702</td> <td>    0.060</td> <td>   -1.162</td> <td> 0.245</td> <td>   -0.189</td> <td>    0.048</td>
</tr>
<tr>
  <th>noise_89_impact_code</th> <td>   -0.1737</td> <td>    0.063</td> <td>   -2.764</td> <td> 0.006</td> <td>   -0.297</td> <td>   -0.050</td>
</tr>
<tr>
  <th>noise_90_impact_code</th> <td>    0.0134</td> <td>    0.057</td> <td>    0.234</td> <td> 0.815</td> <td>   -0.099</td> <td>    0.126</td>
</tr>
<tr>
  <th>noise_91_impact_code</th> <td>   -0.0043</td> <td>    0.058</td> <td>   -0.074</td> <td> 0.941</td> <td>   -0.118</td> <td>    0.109</td>
</tr>
<tr>
  <th>noise_92_impact_code</th> <td>   -0.0076</td> <td>    0.058</td> <td>   -0.130</td> <td> 0.896</td> <td>   -0.121</td> <td>    0.106</td>
</tr>
<tr>
  <th>noise_93_impact_code</th> <td>   -0.1267</td> <td>    0.064</td> <td>   -1.992</td> <td> 0.047</td> <td>   -0.251</td> <td>   -0.002</td>
</tr>
<tr>
  <th>noise_94_impact_code</th> <td>   -0.0232</td> <td>    0.055</td> <td>   -0.419</td> <td> 0.675</td> <td>   -0.132</td> <td>    0.085</td>
</tr>
<tr>
  <th>noise_95_impact_code</th> <td>   -0.0425</td> <td>    0.058</td> <td>   -0.728</td> <td> 0.466</td> <td>   -0.157</td> <td>    0.072</td>
</tr>
<tr>
  <th>noise_96_impact_code</th> <td>    0.0225</td> <td>    0.057</td> <td>    0.393</td> <td> 0.694</td> <td>   -0.090</td> <td>    0.135</td>
</tr>
<tr>
  <th>noise_97_impact_code</th> <td>   -0.0171</td> <td>    0.059</td> <td>   -0.292</td> <td> 0.770</td> <td>   -0.132</td> <td>    0.098</td>
</tr>
<tr>
  <th>noise_98_impact_code</th> <td>   -0.1715</td> <td>    0.060</td> <td>   -2.863</td> <td> 0.004</td> <td>   -0.289</td> <td>   -0.054</td>
</tr>
<tr>
  <th>noise_99_impact_code</th> <td>   -0.0210</td> <td>    0.060</td> <td>   -0.348</td> <td> 0.728</td> <td>   -0.139</td> <td>    0.097</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td> 3.684</td> <th>  Durbin-Watson:     </th> <td>   1.996</td>
</tr>
<tr>
  <th>Prob(Omnibus):</th> <td> 0.159</td> <th>  Jarque-Bera (JB):  </th> <td>   3.337</td>
</tr>
<tr>
  <th>Skew:</th>          <td> 0.027</td> <th>  Prob(JB):          </th> <td>   0.189</td>
</tr>
<tr>
  <th>Kurtosis:</th>      <td> 2.832</td> <th>  Cond. No.          </th> <td>    7.47</td>
</tr>
</table><br/><br/>Warnings:<br/>[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.




```python
sklearn.metrics.r2_score(plot_train["y"],plot_train["predict_cross_all_vars"])
```




    0.29137541013309876




```python
sklearn.metrics.r2_score(plot_test["y"],plot_test["predict_cross_all_vars"])
```




    0.2468888336794015




```python
smf3 = statsmodels.api.OLS(
    y_train, 
    statsmodels.api.add_constant(cross_frame[recommended_vars])).fit()
smf3.summary()
```




<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>            <td>y</td>        <th>  R-squared:         </th> <td>   0.250</td> 
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th> <td>   0.247</td> 
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th> <td>   77.44</td> 
</tr>
<tr>
  <th>Date:</th>             <td>Thu, 25 Jul 2019</td> <th>  Prob (F-statistic):</th> <td>8.04e-151</td>
</tr>
<tr>
  <th>Time:</th>                 <td>16:09:08</td>     <th>  Log-Likelihood:    </th> <td> -10752.</td> 
</tr>
<tr>
  <th>No. Observations:</th>      <td>  2570</td>      <th>  AIC:               </th> <td>2.153e+04</td>
</tr>
<tr>
  <th>Df Residuals:</th>          <td>  2558</td>      <th>  BIC:               </th> <td>2.160e+04</td>
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
  <th>const</th>                <td>   -0.0939</td> <td>    0.314</td> <td>   -0.299</td> <td> 0.765</td> <td>   -0.710</td> <td>    0.522</td>
</tr>
<tr>
  <th>var_0_impact_code</th>    <td>    0.4651</td> <td>    0.045</td> <td>   10.244</td> <td> 0.000</td> <td>    0.376</td> <td>    0.554</td>
</tr>
<tr>
  <th>var_1_impact_code</th>    <td>    0.4441</td> <td>    0.046</td> <td>    9.600</td> <td> 0.000</td> <td>    0.353</td> <td>    0.535</td>
</tr>
<tr>
  <th>var_2_impact_code</th>    <td>    0.3367</td> <td>    0.046</td> <td>    7.353</td> <td> 0.000</td> <td>    0.247</td> <td>    0.426</td>
</tr>
<tr>
  <th>var_3_impact_code</th>    <td>    0.4906</td> <td>    0.046</td> <td>   10.638</td> <td> 0.000</td> <td>    0.400</td> <td>    0.581</td>
</tr>
<tr>
  <th>var_4_impact_code</th>    <td>    0.4761</td> <td>    0.047</td> <td>   10.212</td> <td> 0.000</td> <td>    0.385</td> <td>    0.568</td>
</tr>
<tr>
  <th>var_5_impact_code</th>    <td>    0.4259</td> <td>    0.048</td> <td>    8.924</td> <td> 0.000</td> <td>    0.332</td> <td>    0.520</td>
</tr>
<tr>
  <th>var_6_impact_code</th>    <td>    0.3729</td> <td>    0.052</td> <td>    7.205</td> <td> 0.000</td> <td>    0.271</td> <td>    0.474</td>
</tr>
<tr>
  <th>var_7_impact_code</th>    <td>    0.4522</td> <td>    0.046</td> <td>    9.820</td> <td> 0.000</td> <td>    0.362</td> <td>    0.543</td>
</tr>
<tr>
  <th>var_8_impact_code</th>    <td>    0.3966</td> <td>    0.048</td> <td>    8.296</td> <td> 0.000</td> <td>    0.303</td> <td>    0.490</td>
</tr>
<tr>
  <th>var_9_impact_code</th>    <td>    0.3869</td> <td>    0.048</td> <td>    8.022</td> <td> 0.000</td> <td>    0.292</td> <td>    0.482</td>
</tr>
<tr>
  <th>noise_18_impact_code</th> <td>    0.1083</td> <td>    0.057</td> <td>    1.894</td> <td> 0.058</td> <td>   -0.004</td> <td>    0.221</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td> 4.406</td> <th>  Durbin-Watson:     </th> <td>   2.011</td>
</tr>
<tr>
  <th>Prob(Omnibus):</th> <td> 0.110</td> <th>  Jarque-Bera (JB):  </th> <td>   3.836</td>
</tr>
<tr>
  <th>Skew:</th>          <td>-0.009</td> <th>  Prob(JB):          </th> <td>   0.147</td>
</tr>
<tr>
  <th>Kurtosis:</th>      <td> 2.812</td> <th>  Cond. No.          </th> <td>    7.11</td>
</tr>
</table><br/><br/>Warnings:<br/>[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.




```python
sklearn.metrics.r2_score(plot_train["y"],plot_train["predict_cross_recommended_vars"])
```




    0.24980847983124788




```python
sklearn.metrics.r2_score(plot_test["y"],plot_test["predict_cross_recommended_vars"])
```




    0.28383996531300304




```python

```
