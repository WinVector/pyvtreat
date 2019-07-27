

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
      <td>21.609390</td>
      <td>level_423</td>
      <td>level_466</td>
      <td>level_66</td>
      <td>level_471</td>
      <td>level_21</td>
      <td>level_293</td>
      <td>level_298</td>
      <td>level_392</td>
      <td>level_120</td>
      <td>...</td>
      <td>level_311</td>
      <td>level_90</td>
      <td>level_64</td>
      <td>level_363</td>
      <td>level_442</td>
      <td>level_228</td>
      <td>level_103</td>
      <td>level_240</td>
      <td>level_387</td>
      <td>level_216</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-34.679568</td>
      <td>level_255</td>
      <td>level_319</td>
      <td>level_152</td>
      <td>level_102</td>
      <td>level_278</td>
      <td>level_6</td>
      <td>level_280</td>
      <td>level_335</td>
      <td>level_412</td>
      <td>...</td>
      <td>level_5</td>
      <td>level_275</td>
      <td>level_395</td>
      <td>level_341</td>
      <td>level_284</td>
      <td>level_253</td>
      <td>level_158</td>
      <td>level_414</td>
      <td>level_419</td>
      <td>level_55</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-11.430131</td>
      <td>level_413</td>
      <td>level_347</td>
      <td>level_90</td>
      <td>level_101</td>
      <td>level_115</td>
      <td>level_156</td>
      <td>level_383</td>
      <td>level_140</td>
      <td>level_275</td>
      <td>...</td>
      <td>level_489</td>
      <td>level_53</td>
      <td>level_404</td>
      <td>level_215</td>
      <td>level_456</td>
      <td>level_493</td>
      <td>level_67</td>
      <td>level_146</td>
      <td>level_324</td>
      <td>level_393</td>
    </tr>
    <tr>
      <th>3</th>
      <td>29.711234</td>
      <td>level_60</td>
      <td>level_108</td>
      <td>level_458</td>
      <td>level_183</td>
      <td>level_43</td>
      <td>level_233</td>
      <td>level_98</td>
      <td>level_186</td>
      <td>level_101</td>
      <td>...</td>
      <td>level_216</td>
      <td>level_463</td>
      <td>level_432</td>
      <td>level_41</td>
      <td>level_308</td>
      <td>level_139</td>
      <td>level_220</td>
      <td>level_136</td>
      <td>level_448</td>
      <td>level_32</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-15.009132</td>
      <td>level_98</td>
      <td>level_140</td>
      <td>level_493</td>
      <td>level_344</td>
      <td>level_2</td>
      <td>level_223</td>
      <td>level_405</td>
      <td>level_111</td>
      <td>level_438</td>
      <td>...</td>
      <td>level_79</td>
      <td>level_78</td>
      <td>level_291</td>
      <td>level_230</td>
      <td>level_225</td>
      <td>level_9</td>
      <td>level_107</td>
      <td>level_495</td>
      <td>level_284</td>
      <td>level_455</td>
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
      <td>-8.679132</td>
      <td>-12.946137</td>
      <td>-10.540995</td>
      <td>-13.802363</td>
      <td>-2.423945</td>
      <td>-2.190078</td>
      <td>-27.101848</td>
      <td>14.417548</td>
      <td>-14.502177</td>
      <td>-22.410318</td>
      <td>...</td>
      <td>-10.967147</td>
      <td>-2.380420</td>
      <td>-10.436120</td>
      <td>-13.803981</td>
      <td>-2.003209</td>
      <td>-4.355741</td>
      <td>-0.747413</td>
      <td>9.070266</td>
      <td>-2.595415</td>
      <td>1.803060</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-9.547589</td>
      <td>1.185870</td>
      <td>-5.059223</td>
      <td>-11.519371</td>
      <td>-9.783299</td>
      <td>-11.861813</td>
      <td>-13.021508</td>
      <td>-2.905935</td>
      <td>-9.991621</td>
      <td>9.410818</td>
      <td>...</td>
      <td>1.591085</td>
      <td>8.329865</td>
      <td>-5.047402</td>
      <td>-11.882084</td>
      <td>7.264065</td>
      <td>1.667959</td>
      <td>-10.928701</td>
      <td>-7.383560</td>
      <td>-9.432686</td>
      <td>-2.118444</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-3.888768</td>
      <td>8.959155</td>
      <td>5.026165</td>
      <td>2.543915</td>
      <td>12.577426</td>
      <td>5.217121</td>
      <td>4.682948</td>
      <td>-1.366168</td>
      <td>0.179817</td>
      <td>17.171127</td>
      <td>...</td>
      <td>10.043307</td>
      <td>-2.805734</td>
      <td>-3.094489</td>
      <td>-5.973781</td>
      <td>-4.441704</td>
      <td>10.806701</td>
      <td>9.085976</td>
      <td>-1.319056</td>
      <td>-1.361479</td>
      <td>11.152599</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-2.551488</td>
      <td>19.544376</td>
      <td>16.623793</td>
      <td>20.029344</td>
      <td>-5.913443</td>
      <td>13.356561</td>
      <td>7.973948</td>
      <td>5.691288</td>
      <td>-4.220708</td>
      <td>-4.934892</td>
      <td>...</td>
      <td>-5.519360</td>
      <td>0.523508</td>
      <td>7.640596</td>
      <td>11.276150</td>
      <td>3.007670</td>
      <td>5.478667</td>
      <td>3.651329</td>
      <td>2.865009</td>
      <td>9.951669</td>
      <td>7.248498</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-11.190103</td>
      <td>0.700021</td>
      <td>6.473456</td>
      <td>5.746871</td>
      <td>0.214304</td>
      <td>-20.888046</td>
      <td>-13.152050</td>
      <td>-7.141453</td>
      <td>-8.515137</td>
      <td>-3.972293</td>
      <td>...</td>
      <td>-2.847291</td>
      <td>-9.850933</td>
      <td>-2.008301</td>
      <td>-7.821950</td>
      <td>4.399075</td>
      <td>-14.265884</td>
      <td>-1.601099</td>
      <td>-7.323682</td>
      <td>-6.379215</td>
      <td>-1.343901</td>
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
      <td>-5.409051</td>
      <td>-9.534771</td>
      <td>-5.165008</td>
      <td>-8.025075</td>
      <td>-1.767962</td>
      <td>-0.914594</td>
      <td>-7.807669</td>
      <td>4.713296</td>
      <td>-11.325538</td>
      <td>-14.301553</td>
      <td>...</td>
      <td>-6.783303</td>
      <td>-1.708832</td>
      <td>-6.966434</td>
      <td>-9.270277</td>
      <td>-1.067556</td>
      <td>-2.705914</td>
      <td>-0.531356</td>
      <td>3.622207</td>
      <td>-1.198465</td>
      <td>1.033777</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-8.813362</td>
      <td>0.886794</td>
      <td>-2.835199</td>
      <td>-10.507145</td>
      <td>-7.201350</td>
      <td>-10.046687</td>
      <td>-6.575432</td>
      <td>-2.096325</td>
      <td>-6.857580</td>
      <td>4.123817</td>
      <td>...</td>
      <td>0.650271</td>
      <td>2.872083</td>
      <td>-3.914992</td>
      <td>-4.689325</td>
      <td>4.323584</td>
      <td>0.846375</td>
      <td>-8.963390</td>
      <td>-5.109565</td>
      <td>-7.899942</td>
      <td>-1.377017</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-3.062155</td>
      <td>7.122420</td>
      <td>3.298227</td>
      <td>2.258247</td>
      <td>5.997278</td>
      <td>3.605126</td>
      <td>3.960861</td>
      <td>-0.610069</td>
      <td>0.148842</td>
      <td>10.793259</td>
      <td>...</td>
      <td>5.553129</td>
      <td>-1.792656</td>
      <td>-1.923821</td>
      <td>-4.084548</td>
      <td>-3.053155</td>
      <td>7.541425</td>
      <td>4.143451</td>
      <td>-1.144251</td>
      <td>-0.642274</td>
      <td>8.787491</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-1.898779</td>
      <td>12.613384</td>
      <td>14.999617</td>
      <td>18.263842</td>
      <td>-4.578783</td>
      <td>12.814313</td>
      <td>7.188919</td>
      <td>3.285220</td>
      <td>-2.726931</td>
      <td>-0.751065</td>
      <td>...</td>
      <td>-2.183322</td>
      <td>0.307034</td>
      <td>5.524847</td>
      <td>5.023735</td>
      <td>2.057327</td>
      <td>4.542621</td>
      <td>2.952050</td>
      <td>1.673224</td>
      <td>9.103564</td>
      <td>6.066340</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-7.304249</td>
      <td>0.455404</td>
      <td>4.054707</td>
      <td>4.689491</td>
      <td>0.131177</td>
      <td>-11.722100</td>
      <td>-10.791469</td>
      <td>-5.111008</td>
      <td>-7.170465</td>
      <td>-2.934171</td>
      <td>...</td>
      <td>-1.289193</td>
      <td>-3.926041</td>
      <td>-0.667166</td>
      <td>-5.587118</td>
      <td>3.205342</td>
      <td>0.000000</td>
      <td>-0.943923</td>
      <td>-4.648872</td>
      <td>-4.826035</td>
      <td>-0.982326</td>
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
      <td>-1.577828</td>
      <td>-7.227700</td>
      <td>-3.855967</td>
      <td>-3.886909e+00</td>
      <td>-4.257620</td>
      <td>-0.761945</td>
      <td>-2.651281</td>
      <td>12.682504</td>
      <td>-7.761752</td>
      <td>-11.772113</td>
      <td>...</td>
      <td>-4.853190</td>
      <td>-0.079950</td>
      <td>-7.657039</td>
      <td>-7.560701</td>
      <td>3.454321</td>
      <td>-0.755195</td>
      <td>1.257315</td>
      <td>10.785771</td>
      <td>-2.642234</td>
      <td>0.464302</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-8.002641</td>
      <td>1.322067</td>
      <td>-3.134132</td>
      <td>-9.644490e+00</td>
      <td>-6.639595</td>
      <td>-9.270213</td>
      <td>-5.744349</td>
      <td>-4.082217</td>
      <td>-5.927746</td>
      <td>6.273442</td>
      <td>...</td>
      <td>3.033130</td>
      <td>2.538090</td>
      <td>-0.903240</td>
      <td>-3.756326</td>
      <td>5.267615</td>
      <td>13.009085</td>
      <td>-6.455530</td>
      <td>-2.353809</td>
      <td>-8.586595</td>
      <td>6.642490</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-0.243802</td>
      <td>7.508896</td>
      <td>3.167852</td>
      <td>-1.110223e-16</td>
      <td>5.862297</td>
      <td>7.347908</td>
      <td>1.197554</td>
      <td>-0.939724</td>
      <td>-0.999242</td>
      <td>13.469074</td>
      <td>...</td>
      <td>1.575999</td>
      <td>-5.978235</td>
      <td>-7.944025</td>
      <td>-6.010237</td>
      <td>-4.399605</td>
      <td>6.958157</td>
      <td>3.346978</td>
      <td>0.524751</td>
      <td>0.000000</td>
      <td>9.401659</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-4.638295</td>
      <td>9.099892</td>
      <td>11.242872</td>
      <td>1.950372e+01</td>
      <td>-7.945239</td>
      <td>0.000000</td>
      <td>5.215806</td>
      <td>2.647548</td>
      <td>-4.593982</td>
      <td>0.000000</td>
      <td>...</td>
      <td>-4.334630</td>
      <td>3.117263</td>
      <td>0.000000</td>
      <td>3.710264</td>
      <td>0.901831</td>
      <td>0.384036</td>
      <td>1.944163</td>
      <td>0.009096</td>
      <td>0.000000</td>
      <td>7.475121</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.000000</td>
      <td>1.957648</td>
      <td>3.428543</td>
      <td>4.586005e+00</td>
      <td>2.236838</td>
      <td>0.000000</td>
      <td>-10.795208</td>
      <td>-2.199459</td>
      <td>-2.569341</td>
      <td>-0.157058</td>
      <td>...</td>
      <td>1.047682</td>
      <td>-2.373316</td>
      <td>0.760367</td>
      <td>-5.079112</td>
      <td>6.617227</td>
      <td>0.000000</td>
      <td>4.177583</td>
      <td>-1.846081</td>
      <td>-4.848367</td>
      <td>0.000000</td>
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
      <td>0.537538</td>
      <td>0.512498</td>
      <td>0.173024</td>
      <td>0.189492</td>
      <td>False</td>
    </tr>
    <tr>
      <th>1</th>
      <td>var_1_impact_code</td>
      <td>0.517274</td>
      <td>0.490760</td>
      <td>0.143846</td>
      <td>0.174671</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2</th>
      <td>var_2_impact_code</td>
      <td>0.494306</td>
      <td>0.462618</td>
      <td>0.106144</td>
      <td>0.160621</td>
      <td>False</td>
    </tr>
    <tr>
      <th>3</th>
      <td>var_3_impact_code</td>
      <td>0.540377</td>
      <td>0.517965</td>
      <td>0.196954</td>
      <td>0.185956</td>
      <td>False</td>
    </tr>
    <tr>
      <th>4</th>
      <td>var_4_impact_code</td>
      <td>0.537606</td>
      <td>0.514735</td>
      <td>0.176158</td>
      <td>0.172490</td>
      <td>False</td>
    </tr>
    <tr>
      <th>5</th>
      <td>var_5_impact_code</td>
      <td>0.536109</td>
      <td>0.508992</td>
      <td>0.169973</td>
      <td>0.158436</td>
      <td>False</td>
    </tr>
    <tr>
      <th>6</th>
      <td>var_6_impact_code</td>
      <td>0.532761</td>
      <td>0.507913</td>
      <td>0.154856</td>
      <td>0.214377</td>
      <td>False</td>
    </tr>
    <tr>
      <th>7</th>
      <td>var_7_impact_code</td>
      <td>0.522554</td>
      <td>0.496471</td>
      <td>0.166342</td>
      <td>0.146731</td>
      <td>False</td>
    </tr>
    <tr>
      <th>8</th>
      <td>var_8_impact_code</td>
      <td>0.518686</td>
      <td>0.494362</td>
      <td>0.136878</td>
      <td>0.146182</td>
      <td>False</td>
    </tr>
    <tr>
      <th>9</th>
      <td>var_9_impact_code</td>
      <td>0.515479</td>
      <td>0.487364</td>
      <td>0.133546</td>
      <td>0.129562</td>
      <td>False</td>
    </tr>
    <tr>
      <th>10</th>
      <td>noise_0_impact_code</td>
      <td>0.445585</td>
      <td>0.414996</td>
      <td>-0.000244</td>
      <td>-0.006766</td>
      <td>True</td>
    </tr>
    <tr>
      <th>11</th>
      <td>noise_1_impact_code</td>
      <td>0.437374</td>
      <td>0.409657</td>
      <td>-0.001348</td>
      <td>-0.000008</td>
      <td>True</td>
    </tr>
    <tr>
      <th>12</th>
      <td>noise_2_impact_code</td>
      <td>0.439760</td>
      <td>0.408043</td>
      <td>-0.022775</td>
      <td>0.010506</td>
      <td>True</td>
    </tr>
    <tr>
      <th>13</th>
      <td>noise_3_impact_code</td>
      <td>0.445398</td>
      <td>0.419427</td>
      <td>0.011538</td>
      <td>-0.021490</td>
      <td>True</td>
    </tr>
    <tr>
      <th>14</th>
      <td>noise_4_impact_code</td>
      <td>0.437810</td>
      <td>0.404932</td>
      <td>0.012626</td>
      <td>0.001318</td>
      <td>True</td>
    </tr>
    <tr>
      <th>15</th>
      <td>noise_5_impact_code</td>
      <td>0.428542</td>
      <td>0.399424</td>
      <td>-0.018017</td>
      <td>0.007516</td>
      <td>True</td>
    </tr>
    <tr>
      <th>16</th>
      <td>noise_6_impact_code</td>
      <td>0.444305</td>
      <td>0.413066</td>
      <td>0.013665</td>
      <td>-0.029029</td>
      <td>True</td>
    </tr>
    <tr>
      <th>17</th>
      <td>noise_7_impact_code</td>
      <td>0.449492</td>
      <td>0.417463</td>
      <td>0.017376</td>
      <td>0.001439</td>
      <td>True</td>
    </tr>
    <tr>
      <th>18</th>
      <td>noise_8_impact_code</td>
      <td>0.421637</td>
      <td>0.386197</td>
      <td>-0.024192</td>
      <td>-0.018093</td>
      <td>True</td>
    </tr>
    <tr>
      <th>19</th>
      <td>noise_9_impact_code</td>
      <td>0.445379</td>
      <td>0.411923</td>
      <td>0.010800</td>
      <td>-0.005291</td>
      <td>True</td>
    </tr>
    <tr>
      <th>20</th>
      <td>noise_10_impact_code</td>
      <td>0.438248</td>
      <td>0.405460</td>
      <td>0.012719</td>
      <td>0.026924</td>
      <td>True</td>
    </tr>
    <tr>
      <th>21</th>
      <td>noise_11_impact_code</td>
      <td>0.478902</td>
      <td>0.449365</td>
      <td>0.054864</td>
      <td>0.016743</td>
      <td>True</td>
    </tr>
    <tr>
      <th>22</th>
      <td>noise_12_impact_code</td>
      <td>0.437313</td>
      <td>0.411621</td>
      <td>0.000433</td>
      <td>-0.005145</td>
      <td>True</td>
    </tr>
    <tr>
      <th>23</th>
      <td>noise_13_impact_code</td>
      <td>0.431276</td>
      <td>0.403635</td>
      <td>0.007586</td>
      <td>0.005632</td>
      <td>True</td>
    </tr>
    <tr>
      <th>24</th>
      <td>noise_14_impact_code</td>
      <td>0.432003</td>
      <td>0.401580</td>
      <td>-0.018066</td>
      <td>0.004108</td>
      <td>True</td>
    </tr>
    <tr>
      <th>25</th>
      <td>noise_15_impact_code</td>
      <td>0.432230</td>
      <td>0.398577</td>
      <td>-0.028103</td>
      <td>0.014775</td>
      <td>True</td>
    </tr>
    <tr>
      <th>26</th>
      <td>noise_16_impact_code</td>
      <td>0.455284</td>
      <td>0.424922</td>
      <td>0.011684</td>
      <td>0.012503</td>
      <td>True</td>
    </tr>
    <tr>
      <th>27</th>
      <td>noise_17_impact_code</td>
      <td>0.441482</td>
      <td>0.406383</td>
      <td>-0.007502</td>
      <td>0.039947</td>
      <td>True</td>
    </tr>
    <tr>
      <th>28</th>
      <td>noise_18_impact_code</td>
      <td>0.430074</td>
      <td>0.395139</td>
      <td>-0.031971</td>
      <td>0.005061</td>
      <td>True</td>
    </tr>
    <tr>
      <th>29</th>
      <td>noise_19_impact_code</td>
      <td>0.470364</td>
      <td>0.438032</td>
      <td>0.053334</td>
      <td>0.016688</td>
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
      <td>0.430104</td>
      <td>0.392580</td>
      <td>-0.008812</td>
      <td>-0.027274</td>
      <td>True</td>
    </tr>
    <tr>
      <th>81</th>
      <td>noise_71_impact_code</td>
      <td>0.446653</td>
      <td>0.417490</td>
      <td>-0.009932</td>
      <td>-0.017198</td>
      <td>True</td>
    </tr>
    <tr>
      <th>82</th>
      <td>noise_72_impact_code</td>
      <td>0.448942</td>
      <td>0.416806</td>
      <td>0.035325</td>
      <td>-0.032601</td>
      <td>True</td>
    </tr>
    <tr>
      <th>83</th>
      <td>noise_73_impact_code</td>
      <td>0.446203</td>
      <td>0.419958</td>
      <td>0.029622</td>
      <td>0.013920</td>
      <td>True</td>
    </tr>
    <tr>
      <th>84</th>
      <td>noise_74_impact_code</td>
      <td>0.466410</td>
      <td>0.432077</td>
      <td>0.046331</td>
      <td>-0.006496</td>
      <td>True</td>
    </tr>
    <tr>
      <th>85</th>
      <td>noise_75_impact_code</td>
      <td>0.417365</td>
      <td>0.386518</td>
      <td>-0.022967</td>
      <td>0.016579</td>
      <td>True</td>
    </tr>
    <tr>
      <th>86</th>
      <td>noise_76_impact_code</td>
      <td>0.432446</td>
      <td>0.402527</td>
      <td>-0.049504</td>
      <td>-0.027383</td>
      <td>True</td>
    </tr>
    <tr>
      <th>87</th>
      <td>noise_77_impact_code</td>
      <td>0.450455</td>
      <td>0.414845</td>
      <td>0.010461</td>
      <td>0.025775</td>
      <td>True</td>
    </tr>
    <tr>
      <th>88</th>
      <td>noise_78_impact_code</td>
      <td>0.462161</td>
      <td>0.430833</td>
      <td>0.023506</td>
      <td>-0.027556</td>
      <td>True</td>
    </tr>
    <tr>
      <th>89</th>
      <td>noise_79_impact_code</td>
      <td>0.443624</td>
      <td>0.414319</td>
      <td>0.014495</td>
      <td>-0.051524</td>
      <td>True</td>
    </tr>
    <tr>
      <th>90</th>
      <td>noise_80_impact_code</td>
      <td>0.436850</td>
      <td>0.405676</td>
      <td>-0.019893</td>
      <td>-0.001293</td>
      <td>True</td>
    </tr>
    <tr>
      <th>91</th>
      <td>noise_81_impact_code</td>
      <td>0.462291</td>
      <td>0.436176</td>
      <td>0.061565</td>
      <td>0.009543</td>
      <td>True</td>
    </tr>
    <tr>
      <th>92</th>
      <td>noise_82_impact_code</td>
      <td>0.464126</td>
      <td>0.433793</td>
      <td>0.048270</td>
      <td>0.010405</td>
      <td>True</td>
    </tr>
    <tr>
      <th>93</th>
      <td>noise_83_impact_code</td>
      <td>0.449701</td>
      <td>0.417996</td>
      <td>0.007772</td>
      <td>-0.004550</td>
      <td>True</td>
    </tr>
    <tr>
      <th>94</th>
      <td>noise_84_impact_code</td>
      <td>0.441646</td>
      <td>0.411434</td>
      <td>0.026074</td>
      <td>-0.001416</td>
      <td>True</td>
    </tr>
    <tr>
      <th>95</th>
      <td>noise_85_impact_code</td>
      <td>0.455337</td>
      <td>0.425115</td>
      <td>0.044889</td>
      <td>0.034777</td>
      <td>True</td>
    </tr>
    <tr>
      <th>96</th>
      <td>noise_86_impact_code</td>
      <td>0.433902</td>
      <td>0.398871</td>
      <td>-0.016003</td>
      <td>-0.031865</td>
      <td>True</td>
    </tr>
    <tr>
      <th>97</th>
      <td>noise_87_impact_code</td>
      <td>0.456591</td>
      <td>0.423223</td>
      <td>0.013064</td>
      <td>0.005138</td>
      <td>True</td>
    </tr>
    <tr>
      <th>98</th>
      <td>noise_88_impact_code</td>
      <td>0.428930</td>
      <td>0.391945</td>
      <td>-0.030786</td>
      <td>-0.017892</td>
      <td>True</td>
    </tr>
    <tr>
      <th>99</th>
      <td>noise_89_impact_code</td>
      <td>0.451824</td>
      <td>0.415203</td>
      <td>0.049153</td>
      <td>0.006819</td>
      <td>True</td>
    </tr>
    <tr>
      <th>100</th>
      <td>noise_90_impact_code</td>
      <td>0.423548</td>
      <td>0.394017</td>
      <td>-0.005003</td>
      <td>-0.011891</td>
      <td>True</td>
    </tr>
    <tr>
      <th>101</th>
      <td>noise_91_impact_code</td>
      <td>0.445310</td>
      <td>0.412975</td>
      <td>0.048420</td>
      <td>0.022945</td>
      <td>True</td>
    </tr>
    <tr>
      <th>102</th>
      <td>noise_92_impact_code</td>
      <td>0.446973</td>
      <td>0.416775</td>
      <td>0.040607</td>
      <td>-0.053256</td>
      <td>True</td>
    </tr>
    <tr>
      <th>103</th>
      <td>noise_93_impact_code</td>
      <td>0.418886</td>
      <td>0.381906</td>
      <td>-0.041899</td>
      <td>0.016745</td>
      <td>True</td>
    </tr>
    <tr>
      <th>104</th>
      <td>noise_94_impact_code</td>
      <td>0.441350</td>
      <td>0.408764</td>
      <td>0.010847</td>
      <td>-0.020402</td>
      <td>True</td>
    </tr>
    <tr>
      <th>105</th>
      <td>noise_95_impact_code</td>
      <td>0.443247</td>
      <td>0.415308</td>
      <td>0.010373</td>
      <td>0.008060</td>
      <td>True</td>
    </tr>
    <tr>
      <th>106</th>
      <td>noise_96_impact_code</td>
      <td>0.457919</td>
      <td>0.428634</td>
      <td>0.033622</td>
      <td>-0.019921</td>
      <td>True</td>
    </tr>
    <tr>
      <th>107</th>
      <td>noise_97_impact_code</td>
      <td>0.453013</td>
      <td>0.415536</td>
      <td>0.005381</td>
      <td>0.027916</td>
      <td>True</td>
    </tr>
    <tr>
      <th>108</th>
      <td>noise_98_impact_code</td>
      <td>0.432367</td>
      <td>0.404039</td>
      <td>-0.024979</td>
      <td>-0.005237</td>
      <td>True</td>
    </tr>
    <tr>
      <th>109</th>
      <td>noise_99_impact_code</td>
      <td>0.452095</td>
      <td>0.415402</td>
      <td>-0.018061</td>
      <td>-0.016240</td>
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

    (0.8216353775825459, 4.0110002843803767e-28)



![png](output_18_1.png)



```python
print(scipy.stats.pearsonr(corr_frame['naive_train_hierarchical_correlation'], corr_frame['test_correlation']))
seaborn.scatterplot(x = "naive_train_hierarchical_correlation", y = "test_correlation", data = corr_frame, hue = "is_noise")
matplotlib.pyplot.plot([-1, 1], [-1, 1], color="red")
matplotlib.pyplot.xlim(-.2,1)
matplotlib.pyplot.ylim(-.2,1)
matplotlib.pyplot.gca().set_aspect('equal', adjustable='box')
```

    (0.8144826467457034, 2.7338656166809485e-27)



![png](output_19_1.png)



```python
print(scipy.stats.pearsonr(corr_frame['cross_frame_correlation'], corr_frame['test_correlation']))
seaborn.scatterplot(x = "cross_frame_correlation", y = "test_correlation", data = corr_frame,  hue = "is_noise")
matplotlib.pyplot.plot([-1, 1], [-1, 1], color="red")
matplotlib.pyplot.xlim(-.2,1)
matplotlib.pyplot.ylim(-.2,1)
matplotlib.pyplot.gca().set_aspect('equal', adjustable='box')
```

    (0.7931790631040361, 5.252120745943376e-25)



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
      <th>orig_variable</th>
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
      <td>noise_95</td>
      <td>impact_code</td>
      <td>True</td>
      <td>True</td>
      <td>0.010373</td>
      <td>0.600721</td>
      <td>110.0</td>
      <td>False</td>
    </tr>
    <tr>
      <th>106</th>
      <td>noise_96_impact_code</td>
      <td>noise_96</td>
      <td>impact_code</td>
      <td>True</td>
      <td>True</td>
      <td>0.033622</td>
      <td>0.089736</td>
      <td>110.0</td>
      <td>False</td>
    </tr>
    <tr>
      <th>107</th>
      <td>noise_97_impact_code</td>
      <td>noise_97</td>
      <td>impact_code</td>
      <td>True</td>
      <td>True</td>
      <td>0.005381</td>
      <td>0.786011</td>
      <td>110.0</td>
      <td>False</td>
    </tr>
    <tr>
      <th>108</th>
      <td>noise_98_impact_code</td>
      <td>noise_98</td>
      <td>impact_code</td>
      <td>True</td>
      <td>True</td>
      <td>-0.024979</td>
      <td>0.207501</td>
      <td>110.0</td>
      <td>False</td>
    </tr>
    <tr>
      <th>109</th>
      <td>noise_99_impact_code</td>
      <td>noise_99</td>
      <td>impact_code</td>
      <td>True</td>
      <td>True</td>
      <td>-0.018061</td>
      <td>0.362136</td>
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
     'noise_11_impact_code',
     'noise_19_impact_code',
     'noise_81_impact_code']




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
      <td>21.609390</td>
      <td>-5.276038</td>
      <td>0.525848</td>
      <td>0.686556</td>
      <td>2.688831</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-11.430131</td>
      <td>-0.744079</td>
      <td>-3.162960</td>
      <td>-2.935260</td>
      <td>-5.072990</td>
    </tr>
    <tr>
      <th>2</th>
      <td>29.711234</td>
      <td>7.972856</td>
      <td>12.452825</td>
      <td>0.039935</td>
      <td>0.772492</td>
    </tr>
    <tr>
      <th>3</th>
      <td>13.826420</td>
      <td>-3.398400</td>
      <td>-4.491066</td>
      <td>-0.955996</td>
      <td>-1.191377</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2.787255</td>
      <td>-9.724369</td>
      <td>-3.755357</td>
      <td>-1.106058</td>
      <td>-1.172688</td>
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

    3.148009362822994



![png](output_33_1.png)



```python
print(rmse(plot_train["predict_naive_hierarchical_all_vars"], plot_train["y"]))
seaborn.scatterplot(x="predict_naive_hierarchical_all_vars", y ="y", data = plot_train)
plt = matplotlib.pyplot.title("Naive hierarchical prediction on train")
```

    4.642603781688982



![png](output_34_1.png)



```python
print(rmse(plot_train["predict_cross_all_vars"], plot_train["y"]))
seaborn.scatterplot(x="predict_cross_all_vars", y ="y", data = plot_train)
plt = matplotlib.pyplot.title("Cross prediction on (all vars) train")
```

    14.837879538021449



![png](output_35_1.png)



```python
print(rmse(plot_train["predict_cross_recommended_vars"], plot_train["y"]))
seaborn.scatterplot(x="predict_cross_recommended_vars", y ="y", data = plot_train)
plt = matplotlib.pyplot.title("Cross prediction (recommended vars) on train")
```

    15.21001841587104



![png](output_36_1.png)



```python
print(rmse(plot_test["predict_naive_empirical_all_vars"], plot_test["y"]))
seaborn.scatterplot(x="predict_naive_empirical_all_vars", y ="y", data = plot_test)
plt = matplotlib.pyplot.title("Naive empirical prediction on test")
```

    17.945143602822135



![png](output_37_1.png)



```python
print(rmse(plot_test["predict_naive_hierarchical_all_vars"], plot_test["y"]))
seaborn.scatterplot(x="predict_naive_hierarchical_all_vars", y ="y", data = plot_test)
plt = matplotlib.pyplot.title("Naive hierarchical prediction on test")
```

    17.56539958000269



![png](output_38_1.png)



```python
print(rmse(plot_test["predict_cross_all_vars"], plot_test["y"]))
seaborn.scatterplot(x="predict_cross_all_vars", y ="y", data = plot_test)
plt = matplotlib.pyplot.title("Cross prediction on (all vars) test")
```

    15.9764331468181



![png](output_39_1.png)



```python
print(rmse(plot_test["predict_cross_recommended_vars"], plot_test["y"]))
seaborn.scatterplot(x="predict_cross_recommended_vars", y ="y", data = plot_test)
plt = matplotlib.pyplot.title("Cross prediction (recommended vars) on test")
```

    15.658332158893971



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
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th> <td>   0.967</td> 
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th> <td>   680.3</td> 
</tr>
<tr>
  <th>Date:</th>             <td>Fri, 26 Jul 2019</td> <th>  Prob (F-statistic):</th>  <td>  0.00</td>  
</tr>
<tr>
  <th>Time:</th>                 <td>18:50:54</td>     <th>  Log-Likelihood:    </th> <td> -6537.4</td> 
</tr>
<tr>
  <th>No. Observations:</th>      <td>  2548</td>      <th>  AIC:               </th> <td>1.330e+04</td>
</tr>
<tr>
  <th>Df Residuals:</th>          <td>  2437</td>      <th>  BIC:               </th> <td>1.395e+04</td>
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
  <th>const</th>                <td>   -0.7378</td> <td>    0.064</td> <td>  -11.570</td> <td> 0.000</td> <td>   -0.863</td> <td>   -0.613</td>
</tr>
<tr>
  <th>var_0_impact_code</th>    <td>    0.0655</td> <td>    0.008</td> <td>    8.113</td> <td> 0.000</td> <td>    0.050</td> <td>    0.081</td>
</tr>
<tr>
  <th>var_1_impact_code</th>    <td>    0.0676</td> <td>    0.008</td> <td>    8.228</td> <td> 0.000</td> <td>    0.051</td> <td>    0.084</td>
</tr>
<tr>
  <th>var_2_impact_code</th>    <td>    0.0535</td> <td>    0.009</td> <td>    6.286</td> <td> 0.000</td> <td>    0.037</td> <td>    0.070</td>
</tr>
<tr>
  <th>var_3_impact_code</th>    <td>    0.0681</td> <td>    0.008</td> <td>    8.453</td> <td> 0.000</td> <td>    0.052</td> <td>    0.084</td>
</tr>
<tr>
  <th>var_4_impact_code</th>    <td>    0.0765</td> <td>    0.008</td> <td>    9.468</td> <td> 0.000</td> <td>    0.061</td> <td>    0.092</td>
</tr>
<tr>
  <th>var_5_impact_code</th>    <td>    0.0647</td> <td>    0.008</td> <td>    8.024</td> <td> 0.000</td> <td>    0.049</td> <td>    0.081</td>
</tr>
<tr>
  <th>var_6_impact_code</th>    <td>    0.0682</td> <td>    0.008</td> <td>    8.418</td> <td> 0.000</td> <td>    0.052</td> <td>    0.084</td>
</tr>
<tr>
  <th>var_7_impact_code</th>    <td>    0.0727</td> <td>    0.008</td> <td>    8.918</td> <td> 0.000</td> <td>    0.057</td> <td>    0.089</td>
</tr>
<tr>
  <th>var_8_impact_code</th>    <td>    0.0564</td> <td>    0.008</td> <td>    6.819</td> <td> 0.000</td> <td>    0.040</td> <td>    0.073</td>
</tr>
<tr>
  <th>var_9_impact_code</th>    <td>    0.0530</td> <td>    0.008</td> <td>    6.434</td> <td> 0.000</td> <td>    0.037</td> <td>    0.069</td>
</tr>
<tr>
  <th>noise_0_impact_code</th>  <td>    0.0481</td> <td>    0.009</td> <td>    5.254</td> <td> 0.000</td> <td>    0.030</td> <td>    0.066</td>
</tr>
<tr>
  <th>noise_1_impact_code</th>  <td>    0.0403</td> <td>    0.009</td> <td>    4.341</td> <td> 0.000</td> <td>    0.022</td> <td>    0.059</td>
</tr>
<tr>
  <th>noise_2_impact_code</th>  <td>    0.0305</td> <td>    0.009</td> <td>    3.291</td> <td> 0.001</td> <td>    0.012</td> <td>    0.049</td>
</tr>
<tr>
  <th>noise_3_impact_code</th>  <td>    0.0343</td> <td>    0.009</td> <td>    3.733</td> <td> 0.000</td> <td>    0.016</td> <td>    0.052</td>
</tr>
<tr>
  <th>noise_4_impact_code</th>  <td>    0.0517</td> <td>    0.009</td> <td>    5.562</td> <td> 0.000</td> <td>    0.033</td> <td>    0.070</td>
</tr>
<tr>
  <th>noise_5_impact_code</th>  <td>    0.0409</td> <td>    0.010</td> <td>    4.278</td> <td> 0.000</td> <td>    0.022</td> <td>    0.060</td>
</tr>
<tr>
  <th>noise_6_impact_code</th>  <td>    0.0562</td> <td>    0.009</td> <td>    6.100</td> <td> 0.000</td> <td>    0.038</td> <td>    0.074</td>
</tr>
<tr>
  <th>noise_7_impact_code</th>  <td>    0.0352</td> <td>    0.009</td> <td>    3.849</td> <td> 0.000</td> <td>    0.017</td> <td>    0.053</td>
</tr>
<tr>
  <th>noise_8_impact_code</th>  <td>    0.0406</td> <td>    0.010</td> <td>    4.257</td> <td> 0.000</td> <td>    0.022</td> <td>    0.059</td>
</tr>
<tr>
  <th>noise_9_impact_code</th>  <td>    0.0282</td> <td>    0.009</td> <td>    3.070</td> <td> 0.002</td> <td>    0.010</td> <td>    0.046</td>
</tr>
<tr>
  <th>noise_10_impact_code</th> <td>    0.0237</td> <td>    0.009</td> <td>    2.542</td> <td> 0.011</td> <td>    0.005</td> <td>    0.042</td>
</tr>
<tr>
  <th>noise_11_impact_code</th> <td>    0.0379</td> <td>    0.009</td> <td>    4.367</td> <td> 0.000</td> <td>    0.021</td> <td>    0.055</td>
</tr>
<tr>
  <th>noise_12_impact_code</th> <td>    0.0485</td> <td>    0.009</td> <td>    5.222</td> <td> 0.000</td> <td>    0.030</td> <td>    0.067</td>
</tr>
<tr>
  <th>noise_13_impact_code</th> <td>    0.0450</td> <td>    0.009</td> <td>    4.762</td> <td> 0.000</td> <td>    0.026</td> <td>    0.064</td>
</tr>
<tr>
  <th>noise_14_impact_code</th> <td>    0.0399</td> <td>    0.009</td> <td>    4.247</td> <td> 0.000</td> <td>    0.021</td> <td>    0.058</td>
</tr>
<tr>
  <th>noise_15_impact_code</th> <td>    0.0477</td> <td>    0.009</td> <td>    5.113</td> <td> 0.000</td> <td>    0.029</td> <td>    0.066</td>
</tr>
<tr>
  <th>noise_16_impact_code</th> <td>    0.0401</td> <td>    0.009</td> <td>    4.412</td> <td> 0.000</td> <td>    0.022</td> <td>    0.058</td>
</tr>
<tr>
  <th>noise_17_impact_code</th> <td>    0.0344</td> <td>    0.009</td> <td>    3.709</td> <td> 0.000</td> <td>    0.016</td> <td>    0.053</td>
</tr>
<tr>
  <th>noise_18_impact_code</th> <td>    0.0413</td> <td>    0.009</td> <td>    4.367</td> <td> 0.000</td> <td>    0.023</td> <td>    0.060</td>
</tr>
<tr>
  <th>noise_19_impact_code</th> <td>    0.0531</td> <td>    0.009</td> <td>    6.048</td> <td> 0.000</td> <td>    0.036</td> <td>    0.070</td>
</tr>
<tr>
  <th>noise_20_impact_code</th> <td>    0.0514</td> <td>    0.009</td> <td>    5.568</td> <td> 0.000</td> <td>    0.033</td> <td>    0.069</td>
</tr>
<tr>
  <th>noise_21_impact_code</th> <td>    0.0479</td> <td>    0.009</td> <td>    5.298</td> <td> 0.000</td> <td>    0.030</td> <td>    0.066</td>
</tr>
<tr>
  <th>noise_22_impact_code</th> <td>    0.0370</td> <td>    0.009</td> <td>    4.115</td> <td> 0.000</td> <td>    0.019</td> <td>    0.055</td>
</tr>
<tr>
  <th>noise_23_impact_code</th> <td>    0.0248</td> <td>    0.010</td> <td>    2.604</td> <td> 0.009</td> <td>    0.006</td> <td>    0.043</td>
</tr>
<tr>
  <th>noise_24_impact_code</th> <td>    0.0356</td> <td>    0.010</td> <td>    3.686</td> <td> 0.000</td> <td>    0.017</td> <td>    0.055</td>
</tr>
<tr>
  <th>noise_25_impact_code</th> <td>    0.0484</td> <td>    0.009</td> <td>    5.129</td> <td> 0.000</td> <td>    0.030</td> <td>    0.067</td>
</tr>
<tr>
  <th>noise_26_impact_code</th> <td>    0.0446</td> <td>    0.009</td> <td>    4.916</td> <td> 0.000</td> <td>    0.027</td> <td>    0.062</td>
</tr>
<tr>
  <th>noise_27_impact_code</th> <td>    0.0285</td> <td>    0.010</td> <td>    2.970</td> <td> 0.003</td> <td>    0.010</td> <td>    0.047</td>
</tr>
<tr>
  <th>noise_28_impact_code</th> <td>    0.0322</td> <td>    0.010</td> <td>    3.354</td> <td> 0.001</td> <td>    0.013</td> <td>    0.051</td>
</tr>
<tr>
  <th>noise_29_impact_code</th> <td>    0.0309</td> <td>    0.009</td> <td>    3.422</td> <td> 0.001</td> <td>    0.013</td> <td>    0.049</td>
</tr>
<tr>
  <th>noise_30_impact_code</th> <td>    0.0500</td> <td>    0.009</td> <td>    5.347</td> <td> 0.000</td> <td>    0.032</td> <td>    0.068</td>
</tr>
<tr>
  <th>noise_31_impact_code</th> <td>    0.0310</td> <td>    0.009</td> <td>    3.334</td> <td> 0.001</td> <td>    0.013</td> <td>    0.049</td>
</tr>
<tr>
  <th>noise_32_impact_code</th> <td>    0.0528</td> <td>    0.009</td> <td>    5.740</td> <td> 0.000</td> <td>    0.035</td> <td>    0.071</td>
</tr>
<tr>
  <th>noise_33_impact_code</th> <td>    0.0517</td> <td>    0.009</td> <td>    5.639</td> <td> 0.000</td> <td>    0.034</td> <td>    0.070</td>
</tr>
<tr>
  <th>noise_34_impact_code</th> <td>    0.0462</td> <td>    0.009</td> <td>    5.036</td> <td> 0.000</td> <td>    0.028</td> <td>    0.064</td>
</tr>
<tr>
  <th>noise_35_impact_code</th> <td>    0.0404</td> <td>    0.009</td> <td>    4.384</td> <td> 0.000</td> <td>    0.022</td> <td>    0.058</td>
</tr>
<tr>
  <th>noise_36_impact_code</th> <td>    0.0454</td> <td>    0.009</td> <td>    5.040</td> <td> 0.000</td> <td>    0.028</td> <td>    0.063</td>
</tr>
<tr>
  <th>noise_37_impact_code</th> <td>    0.0480</td> <td>    0.010</td> <td>    5.024</td> <td> 0.000</td> <td>    0.029</td> <td>    0.067</td>
</tr>
<tr>
  <th>noise_38_impact_code</th> <td>    0.0558</td> <td>    0.009</td> <td>    6.256</td> <td> 0.000</td> <td>    0.038</td> <td>    0.073</td>
</tr>
<tr>
  <th>noise_39_impact_code</th> <td>    0.0276</td> <td>    0.009</td> <td>    3.006</td> <td> 0.003</td> <td>    0.010</td> <td>    0.046</td>
</tr>
<tr>
  <th>noise_40_impact_code</th> <td>    0.0221</td> <td>    0.010</td> <td>    2.258</td> <td> 0.024</td> <td>    0.003</td> <td>    0.041</td>
</tr>
<tr>
  <th>noise_41_impact_code</th> <td>    0.0451</td> <td>    0.009</td> <td>    5.099</td> <td> 0.000</td> <td>    0.028</td> <td>    0.062</td>
</tr>
<tr>
  <th>noise_42_impact_code</th> <td>    0.0431</td> <td>    0.010</td> <td>    4.501</td> <td> 0.000</td> <td>    0.024</td> <td>    0.062</td>
</tr>
<tr>
  <th>noise_43_impact_code</th> <td>    0.0504</td> <td>    0.009</td> <td>    5.368</td> <td> 0.000</td> <td>    0.032</td> <td>    0.069</td>
</tr>
<tr>
  <th>noise_44_impact_code</th> <td>    0.0435</td> <td>    0.009</td> <td>    4.756</td> <td> 0.000</td> <td>    0.026</td> <td>    0.061</td>
</tr>
<tr>
  <th>noise_45_impact_code</th> <td>    0.0393</td> <td>    0.009</td> <td>    4.237</td> <td> 0.000</td> <td>    0.021</td> <td>    0.058</td>
</tr>
<tr>
  <th>noise_46_impact_code</th> <td>    0.0345</td> <td>    0.009</td> <td>    3.684</td> <td> 0.000</td> <td>    0.016</td> <td>    0.053</td>
</tr>
<tr>
  <th>noise_47_impact_code</th> <td>    0.0452</td> <td>    0.009</td> <td>    4.931</td> <td> 0.000</td> <td>    0.027</td> <td>    0.063</td>
</tr>
<tr>
  <th>noise_48_impact_code</th> <td>    0.0197</td> <td>    0.009</td> <td>    2.143</td> <td> 0.032</td> <td>    0.002</td> <td>    0.038</td>
</tr>
<tr>
  <th>noise_49_impact_code</th> <td>    0.0467</td> <td>    0.009</td> <td>    5.071</td> <td> 0.000</td> <td>    0.029</td> <td>    0.065</td>
</tr>
<tr>
  <th>noise_50_impact_code</th> <td>    0.0433</td> <td>    0.009</td> <td>    4.775</td> <td> 0.000</td> <td>    0.026</td> <td>    0.061</td>
</tr>
<tr>
  <th>noise_51_impact_code</th> <td>    0.0341</td> <td>    0.009</td> <td>    3.637</td> <td> 0.000</td> <td>    0.016</td> <td>    0.053</td>
</tr>
<tr>
  <th>noise_52_impact_code</th> <td>    0.0428</td> <td>    0.009</td> <td>    4.783</td> <td> 0.000</td> <td>    0.025</td> <td>    0.060</td>
</tr>
<tr>
  <th>noise_53_impact_code</th> <td>    0.0426</td> <td>    0.009</td> <td>    4.601</td> <td> 0.000</td> <td>    0.024</td> <td>    0.061</td>
</tr>
<tr>
  <th>noise_54_impact_code</th> <td>    0.0514</td> <td>    0.009</td> <td>    5.488</td> <td> 0.000</td> <td>    0.033</td> <td>    0.070</td>
</tr>
<tr>
  <th>noise_55_impact_code</th> <td>    0.0362</td> <td>    0.009</td> <td>    3.819</td> <td> 0.000</td> <td>    0.018</td> <td>    0.055</td>
</tr>
<tr>
  <th>noise_56_impact_code</th> <td>    0.0447</td> <td>    0.009</td> <td>    4.946</td> <td> 0.000</td> <td>    0.027</td> <td>    0.062</td>
</tr>
<tr>
  <th>noise_57_impact_code</th> <td>    0.0363</td> <td>    0.009</td> <td>    3.838</td> <td> 0.000</td> <td>    0.018</td> <td>    0.055</td>
</tr>
<tr>
  <th>noise_58_impact_code</th> <td>    0.0274</td> <td>    0.009</td> <td>    2.923</td> <td> 0.003</td> <td>    0.009</td> <td>    0.046</td>
</tr>
<tr>
  <th>noise_59_impact_code</th> <td>    0.0411</td> <td>    0.009</td> <td>    4.515</td> <td> 0.000</td> <td>    0.023</td> <td>    0.059</td>
</tr>
<tr>
  <th>noise_60_impact_code</th> <td>    0.0395</td> <td>    0.009</td> <td>    4.250</td> <td> 0.000</td> <td>    0.021</td> <td>    0.058</td>
</tr>
<tr>
  <th>noise_61_impact_code</th> <td>    0.0368</td> <td>    0.009</td> <td>    3.978</td> <td> 0.000</td> <td>    0.019</td> <td>    0.055</td>
</tr>
<tr>
  <th>noise_62_impact_code</th> <td>    0.0405</td> <td>    0.009</td> <td>    4.310</td> <td> 0.000</td> <td>    0.022</td> <td>    0.059</td>
</tr>
<tr>
  <th>noise_63_impact_code</th> <td>    0.0433</td> <td>    0.009</td> <td>    4.633</td> <td> 0.000</td> <td>    0.025</td> <td>    0.062</td>
</tr>
<tr>
  <th>noise_64_impact_code</th> <td>    0.0147</td> <td>    0.009</td> <td>    1.557</td> <td> 0.119</td> <td>   -0.004</td> <td>    0.033</td>
</tr>
<tr>
  <th>noise_65_impact_code</th> <td>    0.0319</td> <td>    0.009</td> <td>    3.549</td> <td> 0.000</td> <td>    0.014</td> <td>    0.050</td>
</tr>
<tr>
  <th>noise_66_impact_code</th> <td>    0.0383</td> <td>    0.010</td> <td>    4.000</td> <td> 0.000</td> <td>    0.020</td> <td>    0.057</td>
</tr>
<tr>
  <th>noise_67_impact_code</th> <td>    0.0461</td> <td>    0.009</td> <td>    5.043</td> <td> 0.000</td> <td>    0.028</td> <td>    0.064</td>
</tr>
<tr>
  <th>noise_68_impact_code</th> <td>    0.0420</td> <td>    0.009</td> <td>    4.445</td> <td> 0.000</td> <td>    0.023</td> <td>    0.061</td>
</tr>
<tr>
  <th>noise_69_impact_code</th> <td>    0.0534</td> <td>    0.009</td> <td>    5.664</td> <td> 0.000</td> <td>    0.035</td> <td>    0.072</td>
</tr>
<tr>
  <th>noise_70_impact_code</th> <td>    0.0388</td> <td>    0.010</td> <td>    4.079</td> <td> 0.000</td> <td>    0.020</td> <td>    0.057</td>
</tr>
<tr>
  <th>noise_71_impact_code</th> <td>    0.0401</td> <td>    0.009</td> <td>    4.373</td> <td> 0.000</td> <td>    0.022</td> <td>    0.058</td>
</tr>
<tr>
  <th>noise_72_impact_code</th> <td>    0.0519</td> <td>    0.009</td> <td>    5.682</td> <td> 0.000</td> <td>    0.034</td> <td>    0.070</td>
</tr>
<tr>
  <th>noise_73_impact_code</th> <td>    0.0446</td> <td>    0.009</td> <td>    4.883</td> <td> 0.000</td> <td>    0.027</td> <td>    0.063</td>
</tr>
<tr>
  <th>noise_74_impact_code</th> <td>    0.0562</td> <td>    0.009</td> <td>    6.359</td> <td> 0.000</td> <td>    0.039</td> <td>    0.073</td>
</tr>
<tr>
  <th>noise_75_impact_code</th> <td>    0.0532</td> <td>    0.010</td> <td>    5.520</td> <td> 0.000</td> <td>    0.034</td> <td>    0.072</td>
</tr>
<tr>
  <th>noise_76_impact_code</th> <td>    0.0121</td> <td>    0.009</td> <td>    1.274</td> <td> 0.203</td> <td>   -0.007</td> <td>    0.031</td>
</tr>
<tr>
  <th>noise_77_impact_code</th> <td>    0.0380</td> <td>    0.009</td> <td>    4.164</td> <td> 0.000</td> <td>    0.020</td> <td>    0.056</td>
</tr>
<tr>
  <th>noise_78_impact_code</th> <td>    0.0505</td> <td>    0.009</td> <td>    5.667</td> <td> 0.000</td> <td>    0.033</td> <td>    0.068</td>
</tr>
<tr>
  <th>noise_79_impact_code</th> <td>    0.0491</td> <td>    0.009</td> <td>    5.337</td> <td> 0.000</td> <td>    0.031</td> <td>    0.067</td>
</tr>
<tr>
  <th>noise_80_impact_code</th> <td>    0.0358</td> <td>    0.009</td> <td>    3.819</td> <td> 0.000</td> <td>    0.017</td> <td>    0.054</td>
</tr>
<tr>
  <th>noise_81_impact_code</th> <td>    0.0473</td> <td>    0.009</td> <td>    5.330</td> <td> 0.000</td> <td>    0.030</td> <td>    0.065</td>
</tr>
<tr>
  <th>noise_82_impact_code</th> <td>    0.0419</td> <td>    0.009</td> <td>    4.727</td> <td> 0.000</td> <td>    0.025</td> <td>    0.059</td>
</tr>
<tr>
  <th>noise_83_impact_code</th> <td>    0.0262</td> <td>    0.009</td> <td>    2.856</td> <td> 0.004</td> <td>    0.008</td> <td>    0.044</td>
</tr>
<tr>
  <th>noise_84_impact_code</th> <td>    0.0330</td> <td>    0.009</td> <td>    3.564</td> <td> 0.000</td> <td>    0.015</td> <td>    0.051</td>
</tr>
<tr>
  <th>noise_85_impact_code</th> <td>    0.0451</td> <td>    0.009</td> <td>    5.005</td> <td> 0.000</td> <td>    0.027</td> <td>    0.063</td>
</tr>
<tr>
  <th>noise_86_impact_code</th> <td>    0.0329</td> <td>    0.009</td> <td>    3.486</td> <td> 0.000</td> <td>    0.014</td> <td>    0.051</td>
</tr>
<tr>
  <th>noise_87_impact_code</th> <td>    0.0416</td> <td>    0.009</td> <td>    4.627</td> <td> 0.000</td> <td>    0.024</td> <td>    0.059</td>
</tr>
<tr>
  <th>noise_88_impact_code</th> <td>    0.0342</td> <td>    0.009</td> <td>    3.633</td> <td> 0.000</td> <td>    0.016</td> <td>    0.053</td>
</tr>
<tr>
  <th>noise_89_impact_code</th> <td>    0.0169</td> <td>    0.009</td> <td>    1.854</td> <td> 0.064</td> <td>   -0.001</td> <td>    0.035</td>
</tr>
<tr>
  <th>noise_90_impact_code</th> <td>    0.0460</td> <td>    0.010</td> <td>    4.840</td> <td> 0.000</td> <td>    0.027</td> <td>    0.065</td>
</tr>
<tr>
  <th>noise_91_impact_code</th> <td>    0.0497</td> <td>    0.009</td> <td>    5.445</td> <td> 0.000</td> <td>    0.032</td> <td>    0.068</td>
</tr>
<tr>
  <th>noise_92_impact_code</th> <td>    0.0387</td> <td>    0.009</td> <td>    4.243</td> <td> 0.000</td> <td>    0.021</td> <td>    0.057</td>
</tr>
<tr>
  <th>noise_93_impact_code</th> <td>    0.0448</td> <td>    0.010</td> <td>    4.676</td> <td> 0.000</td> <td>    0.026</td> <td>    0.064</td>
</tr>
<tr>
  <th>noise_94_impact_code</th> <td>    0.0277</td> <td>    0.009</td> <td>    2.971</td> <td> 0.003</td> <td>    0.009</td> <td>    0.046</td>
</tr>
<tr>
  <th>noise_95_impact_code</th> <td>    0.0433</td> <td>    0.009</td> <td>    4.679</td> <td> 0.000</td> <td>    0.025</td> <td>    0.061</td>
</tr>
<tr>
  <th>noise_96_impact_code</th> <td>    0.0495</td> <td>    0.009</td> <td>    5.522</td> <td> 0.000</td> <td>    0.032</td> <td>    0.067</td>
</tr>
<tr>
  <th>noise_97_impact_code</th> <td>    0.0508</td> <td>    0.009</td> <td>    5.609</td> <td> 0.000</td> <td>    0.033</td> <td>    0.069</td>
</tr>
<tr>
  <th>noise_98_impact_code</th> <td>    0.0334</td> <td>    0.009</td> <td>    3.556</td> <td> 0.000</td> <td>    0.015</td> <td>    0.052</td>
</tr>
<tr>
  <th>noise_99_impact_code</th> <td>    0.0337</td> <td>    0.009</td> <td>    3.713</td> <td> 0.000</td> <td>    0.016</td> <td>    0.052</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td> 3.295</td> <th>  Durbin-Watson:     </th> <td>   2.031</td>
</tr>
<tr>
  <th>Prob(Omnibus):</th> <td> 0.193</td> <th>  Jarque-Bera (JB):  </th> <td>   3.502</td>
</tr>
<tr>
  <th>Skew:</th>          <td>-0.028</td> <th>  Prob(JB):          </th> <td>   0.174</td>
</tr>
<tr>
  <th>Kurtosis:</th>      <td> 3.173</td> <th>  Cond. No.          </th> <td>    38.6</td>
</tr>
</table><br/><br/>Warnings:<br/>[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.




```python
sklearn.metrics.r2_score(plot_train["y"],plot_train["predict_naive_empirical_all_vars"])
```




    0.968462555157565




```python
sklearn.metrics.r2_score(plot_test["y"],plot_test["predict_naive_empirical_all_vars"])
```




    0.054959350522135186




```python
smf2 = statsmodels.api.OLS(
    y_train, 
    statsmodels.api.add_constant(cross_frame[all_vars])).fit()
smf2.summary()
```




<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>            <td>y</td>        <th>  R-squared:         </th> <td>   0.299</td> 
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th> <td>   0.268</td> 
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th> <td>   9.466</td> 
</tr>
<tr>
  <th>Date:</th>             <td>Fri, 26 Jul 2019</td> <th>  Prob (F-statistic):</th> <td>1.97e-121</td>
</tr>
<tr>
  <th>Time:</th>                 <td>18:50:54</td>     <th>  Log-Likelihood:    </th> <td> -10488.</td> 
</tr>
<tr>
  <th>No. Observations:</th>      <td>  2548</td>      <th>  AIC:               </th> <td>2.120e+04</td>
</tr>
<tr>
  <th>Df Residuals:</th>          <td>  2437</td>      <th>  BIC:               </th> <td>2.185e+04</td>
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
  <th>const</th>                <td>   -0.6033</td> <td>    0.306</td> <td>   -1.973</td> <td> 0.049</td> <td>   -1.203</td> <td>   -0.004</td>
</tr>
<tr>
  <th>var_0_impact_code</th>    <td>    0.4912</td> <td>    0.047</td> <td>   10.458</td> <td> 0.000</td> <td>    0.399</td> <td>    0.583</td>
</tr>
<tr>
  <th>var_1_impact_code</th>    <td>    0.4072</td> <td>    0.048</td> <td>    8.445</td> <td> 0.000</td> <td>    0.313</td> <td>    0.502</td>
</tr>
<tr>
  <th>var_2_impact_code</th>    <td>    0.2776</td> <td>    0.052</td> <td>    5.369</td> <td> 0.000</td> <td>    0.176</td> <td>    0.379</td>
</tr>
<tr>
  <th>var_3_impact_code</th>    <td>    0.5005</td> <td>    0.046</td> <td>   10.832</td> <td> 0.000</td> <td>    0.410</td> <td>    0.591</td>
</tr>
<tr>
  <th>var_4_impact_code</th>    <td>    0.5155</td> <td>    0.047</td> <td>   10.962</td> <td> 0.000</td> <td>    0.423</td> <td>    0.608</td>
</tr>
<tr>
  <th>var_5_impact_code</th>    <td>    0.4462</td> <td>    0.045</td> <td>    9.942</td> <td> 0.000</td> <td>    0.358</td> <td>    0.534</td>
</tr>
<tr>
  <th>var_6_impact_code</th>    <td>    0.4556</td> <td>    0.049</td> <td>    9.368</td> <td> 0.000</td> <td>    0.360</td> <td>    0.551</td>
</tr>
<tr>
  <th>var_7_impact_code</th>    <td>    0.4614</td> <td>    0.046</td> <td>   10.060</td> <td> 0.000</td> <td>    0.371</td> <td>    0.551</td>
</tr>
<tr>
  <th>var_8_impact_code</th>    <td>    0.3792</td> <td>    0.049</td> <td>    7.789</td> <td> 0.000</td> <td>    0.284</td> <td>    0.475</td>
</tr>
<tr>
  <th>var_9_impact_code</th>    <td>    0.3201</td> <td>    0.048</td> <td>    6.685</td> <td> 0.000</td> <td>    0.226</td> <td>    0.414</td>
</tr>
<tr>
  <th>noise_0_impact_code</th>  <td>    0.0464</td> <td>    0.061</td> <td>    0.759</td> <td> 0.448</td> <td>   -0.073</td> <td>    0.166</td>
</tr>
<tr>
  <th>noise_1_impact_code</th>  <td>   -0.0058</td> <td>    0.062</td> <td>   -0.093</td> <td> 0.926</td> <td>   -0.128</td> <td>    0.116</td>
</tr>
<tr>
  <th>noise_2_impact_code</th>  <td>   -0.0414</td> <td>    0.058</td> <td>   -0.711</td> <td> 0.477</td> <td>   -0.156</td> <td>    0.073</td>
</tr>
<tr>
  <th>noise_3_impact_code</th>  <td>   -0.0308</td> <td>    0.057</td> <td>   -0.542</td> <td> 0.588</td> <td>   -0.142</td> <td>    0.081</td>
</tr>
<tr>
  <th>noise_4_impact_code</th>  <td>    0.0780</td> <td>    0.059</td> <td>    1.330</td> <td> 0.184</td> <td>   -0.037</td> <td>    0.193</td>
</tr>
<tr>
  <th>noise_5_impact_code</th>  <td>   -0.0494</td> <td>    0.063</td> <td>   -0.790</td> <td> 0.430</td> <td>   -0.172</td> <td>    0.073</td>
</tr>
<tr>
  <th>noise_6_impact_code</th>  <td>    0.1061</td> <td>    0.058</td> <td>    1.825</td> <td> 0.068</td> <td>   -0.008</td> <td>    0.220</td>
</tr>
<tr>
  <th>noise_7_impact_code</th>  <td>    0.0147</td> <td>    0.056</td> <td>    0.262</td> <td> 0.793</td> <td>   -0.095</td> <td>    0.124</td>
</tr>
<tr>
  <th>noise_8_impact_code</th>  <td>   -0.0473</td> <td>    0.061</td> <td>   -0.780</td> <td> 0.436</td> <td>   -0.166</td> <td>    0.072</td>
</tr>
<tr>
  <th>noise_9_impact_code</th>  <td>    0.0491</td> <td>    0.059</td> <td>    0.839</td> <td> 0.402</td> <td>   -0.066</td> <td>    0.164</td>
</tr>
<tr>
  <th>noise_10_impact_code</th> <td>    0.0237</td> <td>    0.059</td> <td>    0.400</td> <td> 0.689</td> <td>   -0.092</td> <td>    0.140</td>
</tr>
<tr>
  <th>noise_11_impact_code</th> <td>    0.1177</td> <td>    0.054</td> <td>    2.172</td> <td> 0.030</td> <td>    0.011</td> <td>    0.224</td>
</tr>
<tr>
  <th>noise_12_impact_code</th> <td>    0.0657</td> <td>    0.059</td> <td>    1.120</td> <td> 0.263</td> <td>   -0.049</td> <td>    0.181</td>
</tr>
<tr>
  <th>noise_13_impact_code</th> <td>    0.0104</td> <td>    0.061</td> <td>    0.169</td> <td> 0.866</td> <td>   -0.110</td> <td>    0.131</td>
</tr>
<tr>
  <th>noise_14_impact_code</th> <td>   -0.0246</td> <td>    0.060</td> <td>   -0.413</td> <td> 0.680</td> <td>   -0.141</td> <td>    0.092</td>
</tr>
<tr>
  <th>noise_15_impact_code</th> <td>   -0.0717</td> <td>    0.060</td> <td>   -1.192</td> <td> 0.234</td> <td>   -0.190</td> <td>    0.046</td>
</tr>
<tr>
  <th>noise_16_impact_code</th> <td>    0.0480</td> <td>    0.056</td> <td>    0.852</td> <td> 0.394</td> <td>   -0.063</td> <td>    0.159</td>
</tr>
<tr>
  <th>noise_17_impact_code</th> <td>   -0.0283</td> <td>    0.055</td> <td>   -0.512</td> <td> 0.609</td> <td>   -0.136</td> <td>    0.080</td>
</tr>
<tr>
  <th>noise_18_impact_code</th> <td>   -0.0556</td> <td>    0.060</td> <td>   -0.921</td> <td> 0.357</td> <td>   -0.174</td> <td>    0.063</td>
</tr>
<tr>
  <th>noise_19_impact_code</th> <td>    0.0857</td> <td>    0.055</td> <td>    1.571</td> <td> 0.116</td> <td>   -0.021</td> <td>    0.193</td>
</tr>
<tr>
  <th>noise_20_impact_code</th> <td>    0.0754</td> <td>    0.058</td> <td>    1.298</td> <td> 0.194</td> <td>   -0.039</td> <td>    0.189</td>
</tr>
<tr>
  <th>noise_21_impact_code</th> <td>    0.0744</td> <td>    0.057</td> <td>    1.307</td> <td> 0.191</td> <td>   -0.037</td> <td>    0.186</td>
</tr>
<tr>
  <th>noise_22_impact_code</th> <td>    0.0479</td> <td>    0.054</td> <td>    0.889</td> <td> 0.374</td> <td>   -0.058</td> <td>    0.154</td>
</tr>
<tr>
  <th>noise_23_impact_code</th> <td>   -0.0524</td> <td>    0.061</td> <td>   -0.866</td> <td> 0.386</td> <td>   -0.171</td> <td>    0.066</td>
</tr>
<tr>
  <th>noise_24_impact_code</th> <td>   -0.0763</td> <td>    0.066</td> <td>   -1.157</td> <td> 0.248</td> <td>   -0.206</td> <td>    0.053</td>
</tr>
<tr>
  <th>noise_25_impact_code</th> <td>    0.0036</td> <td>    0.059</td> <td>    0.061</td> <td> 0.951</td> <td>   -0.111</td> <td>    0.118</td>
</tr>
<tr>
  <th>noise_26_impact_code</th> <td>    0.0851</td> <td>    0.054</td> <td>    1.587</td> <td> 0.113</td> <td>   -0.020</td> <td>    0.190</td>
</tr>
<tr>
  <th>noise_27_impact_code</th> <td>   -0.1658</td> <td>    0.058</td> <td>   -2.845</td> <td> 0.004</td> <td>   -0.280</td> <td>   -0.052</td>
</tr>
<tr>
  <th>noise_28_impact_code</th> <td>   -0.1395</td> <td>    0.061</td> <td>   -2.274</td> <td> 0.023</td> <td>   -0.260</td> <td>   -0.019</td>
</tr>
<tr>
  <th>noise_29_impact_code</th> <td>    0.0223</td> <td>    0.054</td> <td>    0.410</td> <td> 0.682</td> <td>   -0.084</td> <td>    0.129</td>
</tr>
<tr>
  <th>noise_30_impact_code</th> <td>    0.0343</td> <td>    0.059</td> <td>    0.583</td> <td> 0.560</td> <td>   -0.081</td> <td>    0.150</td>
</tr>
<tr>
  <th>noise_31_impact_code</th> <td>    0.0369</td> <td>    0.059</td> <td>    0.626</td> <td> 0.531</td> <td>   -0.079</td> <td>    0.152</td>
</tr>
<tr>
  <th>noise_32_impact_code</th> <td>   -0.0161</td> <td>    0.056</td> <td>   -0.287</td> <td> 0.774</td> <td>   -0.126</td> <td>    0.094</td>
</tr>
<tr>
  <th>noise_33_impact_code</th> <td>    0.0063</td> <td>    0.059</td> <td>    0.107</td> <td> 0.915</td> <td>   -0.109</td> <td>    0.122</td>
</tr>
<tr>
  <th>noise_34_impact_code</th> <td>    0.0384</td> <td>    0.055</td> <td>    0.695</td> <td> 0.487</td> <td>   -0.070</td> <td>    0.147</td>
</tr>
<tr>
  <th>noise_35_impact_code</th> <td>    0.0052</td> <td>    0.058</td> <td>    0.090</td> <td> 0.928</td> <td>   -0.108</td> <td>    0.119</td>
</tr>
<tr>
  <th>noise_36_impact_code</th> <td>    0.1045</td> <td>    0.057</td> <td>    1.832</td> <td> 0.067</td> <td>   -0.007</td> <td>    0.216</td>
</tr>
<tr>
  <th>noise_37_impact_code</th> <td>   -0.0787</td> <td>    0.057</td> <td>   -1.376</td> <td> 0.169</td> <td>   -0.191</td> <td>    0.034</td>
</tr>
<tr>
  <th>noise_38_impact_code</th> <td>    0.0842</td> <td>    0.055</td> <td>    1.520</td> <td> 0.129</td> <td>   -0.024</td> <td>    0.193</td>
</tr>
<tr>
  <th>noise_39_impact_code</th> <td>   -0.0209</td> <td>    0.057</td> <td>   -0.364</td> <td> 0.716</td> <td>   -0.133</td> <td>    0.092</td>
</tr>
<tr>
  <th>noise_40_impact_code</th> <td>   -0.1129</td> <td>    0.066</td> <td>   -1.724</td> <td> 0.085</td> <td>   -0.241</td> <td>    0.016</td>
</tr>
<tr>
  <th>noise_41_impact_code</th> <td>    0.0568</td> <td>    0.054</td> <td>    1.057</td> <td> 0.291</td> <td>   -0.049</td> <td>    0.162</td>
</tr>
<tr>
  <th>noise_42_impact_code</th> <td>   -0.0706</td> <td>    0.060</td> <td>   -1.169</td> <td> 0.242</td> <td>   -0.189</td> <td>    0.048</td>
</tr>
<tr>
  <th>noise_43_impact_code</th> <td>   -0.0226</td> <td>    0.060</td> <td>   -0.379</td> <td> 0.704</td> <td>   -0.140</td> <td>    0.094</td>
</tr>
<tr>
  <th>noise_44_impact_code</th> <td>   -0.0064</td> <td>    0.058</td> <td>   -0.110</td> <td> 0.912</td> <td>   -0.121</td> <td>    0.108</td>
</tr>
<tr>
  <th>noise_45_impact_code</th> <td>   -0.0302</td> <td>    0.061</td> <td>   -0.496</td> <td> 0.620</td> <td>   -0.150</td> <td>    0.089</td>
</tr>
<tr>
  <th>noise_46_impact_code</th> <td>   -0.1051</td> <td>    0.059</td> <td>   -1.775</td> <td> 0.076</td> <td>   -0.221</td> <td>    0.011</td>
</tr>
<tr>
  <th>noise_47_impact_code</th> <td>   -0.0469</td> <td>    0.055</td> <td>   -0.849</td> <td> 0.396</td> <td>   -0.155</td> <td>    0.062</td>
</tr>
<tr>
  <th>noise_48_impact_code</th> <td>   -0.0023</td> <td>    0.058</td> <td>   -0.040</td> <td> 0.968</td> <td>   -0.116</td> <td>    0.111</td>
</tr>
<tr>
  <th>noise_49_impact_code</th> <td>    0.0440</td> <td>    0.056</td> <td>    0.786</td> <td> 0.432</td> <td>   -0.066</td> <td>    0.154</td>
</tr>
<tr>
  <th>noise_50_impact_code</th> <td>    0.0031</td> <td>    0.058</td> <td>    0.053</td> <td> 0.958</td> <td>   -0.110</td> <td>    0.116</td>
</tr>
<tr>
  <th>noise_51_impact_code</th> <td>   -0.0713</td> <td>    0.058</td> <td>   -1.228</td> <td> 0.220</td> <td>   -0.185</td> <td>    0.043</td>
</tr>
<tr>
  <th>noise_52_impact_code</th> <td>    0.0470</td> <td>    0.054</td> <td>    0.873</td> <td> 0.383</td> <td>   -0.058</td> <td>    0.152</td>
</tr>
<tr>
  <th>noise_53_impact_code</th> <td>    0.0959</td> <td>    0.060</td> <td>    1.596</td> <td> 0.111</td> <td>   -0.022</td> <td>    0.214</td>
</tr>
<tr>
  <th>noise_54_impact_code</th> <td>    0.0454</td> <td>    0.059</td> <td>    0.775</td> <td> 0.439</td> <td>   -0.069</td> <td>    0.160</td>
</tr>
<tr>
  <th>noise_55_impact_code</th> <td>   -0.0426</td> <td>    0.058</td> <td>   -0.731</td> <td> 0.465</td> <td>   -0.157</td> <td>    0.072</td>
</tr>
<tr>
  <th>noise_56_impact_code</th> <td>    0.1417</td> <td>    0.056</td> <td>    2.513</td> <td> 0.012</td> <td>    0.031</td> <td>    0.252</td>
</tr>
<tr>
  <th>noise_57_impact_code</th> <td>   -0.0255</td> <td>    0.060</td> <td>   -0.424</td> <td> 0.671</td> <td>   -0.143</td> <td>    0.092</td>
</tr>
<tr>
  <th>noise_58_impact_code</th> <td>    0.0147</td> <td>    0.057</td> <td>    0.257</td> <td> 0.797</td> <td>   -0.097</td> <td>    0.127</td>
</tr>
<tr>
  <th>noise_59_impact_code</th> <td>    0.0398</td> <td>    0.053</td> <td>    0.756</td> <td> 0.450</td> <td>   -0.063</td> <td>    0.143</td>
</tr>
<tr>
  <th>noise_60_impact_code</th> <td>   -0.0179</td> <td>    0.056</td> <td>   -0.317</td> <td> 0.751</td> <td>   -0.129</td> <td>    0.093</td>
</tr>
<tr>
  <th>noise_61_impact_code</th> <td>   -0.0672</td> <td>    0.057</td> <td>   -1.185</td> <td> 0.236</td> <td>   -0.179</td> <td>    0.044</td>
</tr>
<tr>
  <th>noise_62_impact_code</th> <td>   -0.0658</td> <td>    0.061</td> <td>   -1.072</td> <td> 0.284</td> <td>   -0.186</td> <td>    0.055</td>
</tr>
<tr>
  <th>noise_63_impact_code</th> <td>   -0.0067</td> <td>    0.058</td> <td>   -0.115</td> <td> 0.908</td> <td>   -0.120</td> <td>    0.107</td>
</tr>
<tr>
  <th>noise_64_impact_code</th> <td>   -0.0613</td> <td>    0.059</td> <td>   -1.045</td> <td> 0.296</td> <td>   -0.176</td> <td>    0.054</td>
</tr>
<tr>
  <th>noise_65_impact_code</th> <td>    0.0813</td> <td>    0.056</td> <td>    1.441</td> <td> 0.150</td> <td>   -0.029</td> <td>    0.192</td>
</tr>
<tr>
  <th>noise_66_impact_code</th> <td>   -0.1028</td> <td>    0.059</td> <td>   -1.744</td> <td> 0.081</td> <td>   -0.218</td> <td>    0.013</td>
</tr>
<tr>
  <th>noise_67_impact_code</th> <td>    0.0877</td> <td>    0.060</td> <td>    1.472</td> <td> 0.141</td> <td>   -0.029</td> <td>    0.204</td>
</tr>
<tr>
  <th>noise_68_impact_code</th> <td>    0.0214</td> <td>    0.059</td> <td>    0.361</td> <td> 0.718</td> <td>   -0.095</td> <td>    0.137</td>
</tr>
<tr>
  <th>noise_69_impact_code</th> <td>   -0.0639</td> <td>    0.059</td> <td>   -1.078</td> <td> 0.281</td> <td>   -0.180</td> <td>    0.052</td>
</tr>
<tr>
  <th>noise_70_impact_code</th> <td>   -0.0135</td> <td>    0.062</td> <td>   -0.218</td> <td> 0.827</td> <td>   -0.135</td> <td>    0.108</td>
</tr>
<tr>
  <th>noise_71_impact_code</th> <td>   -0.0413</td> <td>    0.056</td> <td>   -0.743</td> <td> 0.458</td> <td>   -0.150</td> <td>    0.068</td>
</tr>
<tr>
  <th>noise_72_impact_code</th> <td>    0.1000</td> <td>    0.058</td> <td>    1.725</td> <td> 0.085</td> <td>   -0.014</td> <td>    0.214</td>
</tr>
<tr>
  <th>noise_73_impact_code</th> <td>    0.0633</td> <td>    0.056</td> <td>    1.122</td> <td> 0.262</td> <td>   -0.047</td> <td>    0.174</td>
</tr>
<tr>
  <th>noise_74_impact_code</th> <td>    0.1108</td> <td>    0.055</td> <td>    2.011</td> <td> 0.044</td> <td>    0.003</td> <td>    0.219</td>
</tr>
<tr>
  <th>noise_75_impact_code</th> <td>   -0.0238</td> <td>    0.059</td> <td>   -0.402</td> <td> 0.688</td> <td>   -0.140</td> <td>    0.092</td>
</tr>
<tr>
  <th>noise_76_impact_code</th> <td>   -0.1181</td> <td>    0.061</td> <td>   -1.935</td> <td> 0.053</td> <td>   -0.238</td> <td>    0.002</td>
</tr>
<tr>
  <th>noise_77_impact_code</th> <td>   -0.0042</td> <td>    0.057</td> <td>   -0.073</td> <td> 0.941</td> <td>   -0.115</td> <td>    0.107</td>
</tr>
<tr>
  <th>noise_78_impact_code</th> <td>    0.0711</td> <td>    0.057</td> <td>    1.238</td> <td> 0.216</td> <td>   -0.042</td> <td>    0.184</td>
</tr>
<tr>
  <th>noise_79_impact_code</th> <td>    0.0360</td> <td>    0.060</td> <td>    0.597</td> <td> 0.550</td> <td>   -0.082</td> <td>    0.154</td>
</tr>
<tr>
  <th>noise_80_impact_code</th> <td>   -0.0244</td> <td>    0.059</td> <td>   -0.413</td> <td> 0.680</td> <td>   -0.140</td> <td>    0.091</td>
</tr>
<tr>
  <th>noise_81_impact_code</th> <td>    0.1284</td> <td>    0.055</td> <td>    2.317</td> <td> 0.021</td> <td>    0.020</td> <td>    0.237</td>
</tr>
<tr>
  <th>noise_82_impact_code</th> <td>    0.1219</td> <td>    0.054</td> <td>    2.240</td> <td> 0.025</td> <td>    0.015</td> <td>    0.229</td>
</tr>
<tr>
  <th>noise_83_impact_code</th> <td>    0.0240</td> <td>    0.057</td> <td>    0.419</td> <td> 0.675</td> <td>   -0.088</td> <td>    0.136</td>
</tr>
<tr>
  <th>noise_84_impact_code</th> <td>    0.0909</td> <td>    0.060</td> <td>    1.513</td> <td> 0.130</td> <td>   -0.027</td> <td>    0.209</td>
</tr>
<tr>
  <th>noise_85_impact_code</th> <td>    0.0730</td> <td>    0.059</td> <td>    1.245</td> <td> 0.213</td> <td>   -0.042</td> <td>    0.188</td>
</tr>
<tr>
  <th>noise_86_impact_code</th> <td>    0.0018</td> <td>    0.058</td> <td>    0.031</td> <td> 0.975</td> <td>   -0.112</td> <td>    0.116</td>
</tr>
<tr>
  <th>noise_87_impact_code</th> <td>    0.0142</td> <td>    0.059</td> <td>    0.239</td> <td> 0.811</td> <td>   -0.102</td> <td>    0.131</td>
</tr>
<tr>
  <th>noise_88_impact_code</th> <td>   -0.0213</td> <td>    0.059</td> <td>   -0.360</td> <td> 0.719</td> <td>   -0.137</td> <td>    0.095</td>
</tr>
<tr>
  <th>noise_89_impact_code</th> <td>    0.0658</td> <td>    0.058</td> <td>    1.129</td> <td> 0.259</td> <td>   -0.049</td> <td>    0.180</td>
</tr>
<tr>
  <th>noise_90_impact_code</th> <td>    0.0104</td> <td>    0.061</td> <td>    0.169</td> <td> 0.866</td> <td>   -0.110</td> <td>    0.131</td>
</tr>
<tr>
  <th>noise_91_impact_code</th> <td>    0.0967</td> <td>    0.059</td> <td>    1.631</td> <td> 0.103</td> <td>   -0.020</td> <td>    0.213</td>
</tr>
<tr>
  <th>noise_92_impact_code</th> <td>    0.0988</td> <td>    0.057</td> <td>    1.720</td> <td> 0.086</td> <td>   -0.014</td> <td>    0.211</td>
</tr>
<tr>
  <th>noise_93_impact_code</th> <td>   -0.0561</td> <td>    0.062</td> <td>   -0.910</td> <td> 0.363</td> <td>   -0.177</td> <td>    0.065</td>
</tr>
<tr>
  <th>noise_94_impact_code</th> <td>    0.0147</td> <td>    0.056</td> <td>    0.265</td> <td> 0.791</td> <td>   -0.094</td> <td>    0.124</td>
</tr>
<tr>
  <th>noise_95_impact_code</th> <td>    0.0377</td> <td>    0.059</td> <td>    0.640</td> <td> 0.522</td> <td>   -0.078</td> <td>    0.153</td>
</tr>
<tr>
  <th>noise_96_impact_code</th> <td>    0.1024</td> <td>    0.055</td> <td>    1.875</td> <td> 0.061</td> <td>   -0.005</td> <td>    0.210</td>
</tr>
<tr>
  <th>noise_97_impact_code</th> <td>    0.0068</td> <td>    0.055</td> <td>    0.123</td> <td> 0.902</td> <td>   -0.101</td> <td>    0.115</td>
</tr>
<tr>
  <th>noise_98_impact_code</th> <td>   -0.0980</td> <td>    0.060</td> <td>   -1.643</td> <td> 0.100</td> <td>   -0.215</td> <td>    0.019</td>
</tr>
<tr>
  <th>noise_99_impact_code</th> <td>   -0.0661</td> <td>    0.056</td> <td>   -1.177</td> <td> 0.239</td> <td>   -0.176</td> <td>    0.044</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td> 3.749</td> <th>  Durbin-Watson:     </th> <td>   2.055</td>
</tr>
<tr>
  <th>Prob(Omnibus):</th> <td> 0.153</td> <th>  Jarque-Bera (JB):  </th> <td>   3.752</td>
</tr>
<tr>
  <th>Skew:</th>          <td>-0.094</td> <th>  Prob(JB):          </th> <td>   0.153</td>
</tr>
<tr>
  <th>Kurtosis:</th>      <td> 2.991</td> <th>  Cond. No.          </th> <td>    7.22</td>
</tr>
</table><br/><br/>Warnings:<br/>[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.




```python
sklearn.metrics.r2_score(plot_train["y"],plot_train["predict_cross_all_vars"])
```




    0.29935479356295913




```python
sklearn.metrics.r2_score(plot_test["y"],plot_test["predict_cross_all_vars"])
```




    0.2509405864802372




```python
smf3 = statsmodels.api.OLS(
    y_train, 
    statsmodels.api.add_constant(cross_frame[recommended_vars])).fit()
smf3.summary()
```




<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>            <td>y</td>        <th>  R-squared:         </th> <td>   0.264</td> 
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th> <td>   0.260</td> 
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th> <td>   69.84</td> 
</tr>
<tr>
  <th>Date:</th>             <td>Fri, 26 Jul 2019</td> <th>  Prob (F-statistic):</th> <td>8.73e-158</td>
</tr>
<tr>
  <th>Time:</th>                 <td>18:50:54</td>     <th>  Log-Likelihood:    </th> <td> -10551.</td> 
</tr>
<tr>
  <th>No. Observations:</th>      <td>  2548</td>      <th>  AIC:               </th> <td>2.113e+04</td>
</tr>
<tr>
  <th>Df Residuals:</th>          <td>  2534</td>      <th>  BIC:               </th> <td>2.121e+04</td>
</tr>
<tr>
  <th>Df Model:</th>              <td>    13</td>      <th>                     </th>     <td> </td>    
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
  <th>const</th>                <td>   -0.6580</td> <td>    0.303</td> <td>   -2.174</td> <td> 0.030</td> <td>   -1.252</td> <td>   -0.064</td>
</tr>
<tr>
  <th>var_0_impact_code</th>    <td>    0.5021</td> <td>    0.046</td> <td>   10.841</td> <td> 0.000</td> <td>    0.411</td> <td>    0.593</td>
</tr>
<tr>
  <th>var_1_impact_code</th>    <td>    0.4137</td> <td>    0.048</td> <td>    8.709</td> <td> 0.000</td> <td>    0.321</td> <td>    0.507</td>
</tr>
<tr>
  <th>var_2_impact_code</th>    <td>    0.3071</td> <td>    0.051</td> <td>    6.024</td> <td> 0.000</td> <td>    0.207</td> <td>    0.407</td>
</tr>
<tr>
  <th>var_3_impact_code</th>    <td>    0.5248</td> <td>    0.046</td> <td>   11.506</td> <td> 0.000</td> <td>    0.435</td> <td>    0.614</td>
</tr>
<tr>
  <th>var_4_impact_code</th>    <td>    0.5113</td> <td>    0.046</td> <td>   11.040</td> <td> 0.000</td> <td>    0.421</td> <td>    0.602</td>
</tr>
<tr>
  <th>var_5_impact_code</th>    <td>    0.4550</td> <td>    0.044</td> <td>   10.267</td> <td> 0.000</td> <td>    0.368</td> <td>    0.542</td>
</tr>
<tr>
  <th>var_6_impact_code</th>    <td>    0.4714</td> <td>    0.048</td> <td>    9.874</td> <td> 0.000</td> <td>    0.378</td> <td>    0.565</td>
</tr>
<tr>
  <th>var_7_impact_code</th>    <td>    0.4736</td> <td>    0.045</td> <td>   10.425</td> <td> 0.000</td> <td>    0.385</td> <td>    0.563</td>
</tr>
<tr>
  <th>var_8_impact_code</th>    <td>    0.3941</td> <td>    0.048</td> <td>    8.260</td> <td> 0.000</td> <td>    0.301</td> <td>    0.488</td>
</tr>
<tr>
  <th>var_9_impact_code</th>    <td>    0.3393</td> <td>    0.047</td> <td>    7.183</td> <td> 0.000</td> <td>    0.247</td> <td>    0.432</td>
</tr>
<tr>
  <th>noise_11_impact_code</th> <td>    0.1268</td> <td>    0.054</td> <td>    2.365</td> <td> 0.018</td> <td>    0.022</td> <td>    0.232</td>
</tr>
<tr>
  <th>noise_19_impact_code</th> <td>    0.0973</td> <td>    0.054</td> <td>    1.804</td> <td> 0.071</td> <td>   -0.008</td> <td>    0.203</td>
</tr>
<tr>
  <th>noise_81_impact_code</th> <td>    0.1240</td> <td>    0.055</td> <td>    2.264</td> <td> 0.024</td> <td>    0.017</td> <td>    0.231</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td> 3.832</td> <th>  Durbin-Watson:     </th> <td>   2.053</td>
</tr>
<tr>
  <th>Prob(Omnibus):</th> <td> 0.147</td> <th>  Jarque-Bera (JB):  </th> <td>   3.886</td>
</tr>
<tr>
  <th>Skew:</th>          <td>-0.089</td> <th>  Prob(JB):          </th> <td>   0.143</td>
</tr>
<tr>
  <th>Kurtosis:</th>      <td> 2.931</td> <th>  Cond. No.          </th> <td>    6.94</td>
</tr>
</table><br/><br/>Warnings:<br/>[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.




```python
sklearn.metrics.r2_score(plot_train["y"],plot_train["predict_cross_recommended_vars"])
```




    0.26376924929240864




```python
sklearn.metrics.r2_score(plot_test["y"],plot_test["predict_cross_recommended_vars"])
```




    0.28047213688877426




```python

```
