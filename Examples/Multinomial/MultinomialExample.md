

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
good_new_variables = numpy.asarray(score_frame.variable[score_frame.recommended])
good_new_variables
```




    array(['var_n_0', 'var_n_1', 'var_c_0_logit_code_a',
           'var_c_0_prevalence_code', 'var_c_0_lev_b', 'var_c_0_lev_a',
           'var_c_0_lev_c', 'var_c_1_logit_code_a', 'var_c_1_prevalence_code',
           'var_c_1_lev_a', 'var_n_0', 'var_n_1', 'var_c_0_logit_code_b',
           'var_c_0_prevalence_code', 'var_c_0_lev_b', 'var_c_0_lev_a',
           'var_c_0_lev_c', 'var_c_1_logit_code_b', 'var_c_1_prevalence_code',
           'var_c_1_lev_b', 'var_c_1_lev_c', 'var_n_0', 'var_n_1',
           'var_c_0_logit_code_c', 'var_c_0_prevalence_code', 'var_c_0_lev_b',
           'var_c_0_lev_c', 'var_c_1_logit_code_c', 'var_c_1_prevalence_code',
           'var_c_1_lev_a', 'var_c_1_lev_b', 'var_c_1_lev_c'], dtype=object)



Or in terms of original variables as follows.


```python
good_original_variables = score_frame.orig_variable[score_frame.recommended].unique()
good_original_variables
```




    array(['var_n_0', 'var_n_1', 'var_c_0', 'var_c_1'], dtype=object)



The cross frame and score frame look like the following.


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
      <th>y</th>
      <th>var_n_0</th>
      <th>var_n_1</th>
      <th>var_c_0_logit_code_c</th>
      <th>var_c_0_logit_code_a</th>
      <th>var_c_0_logit_code_b</th>
      <th>var_c_0_prevalence_code</th>
      <th>var_c_0_lev_b</th>
      <th>var_c_0_lev_a</th>
      <th>var_c_0_lev_c</th>
      <th>var_c_1_logit_code_c</th>
      <th>var_c_1_logit_code_a</th>
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
      <td>-0.170159</td>
      <td>-0.269680</td>
      <td>0.334343</td>
      <td>0.352</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>-0.157131</td>
      <td>0.157304</td>
      <td>-0.029161</td>
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
      <td>-0.159685</td>
      <td>0.291708</td>
      <td>-0.193574</td>
      <td>0.328</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>-0.278599</td>
      <td>-0.101950</td>
      <td>0.316892</td>
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
      <td>-0.213218</td>
      <td>-0.164782</td>
      <td>0.304510</td>
      <td>0.352</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>-0.211409</td>
      <td>0.143157</td>
      <td>0.031278</td>
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
      <td>0.305725</td>
      <td>-0.203392</td>
      <td>-0.245761</td>
      <td>0.320</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>-0.278599</td>
      <td>-0.101950</td>
      <td>0.316892</td>
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
      <td>0.248363</td>
      <td>-0.169911</td>
      <td>-0.190688</td>
      <td>0.320</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.343053</td>
      <td>-0.093987</td>
      <td>-0.438143</td>
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
      <td>var_c_0_logit_code_c</td>
      <td>var_c_0</td>
      <td>logit_code</td>
      <td>True</td>
      <td>True</td>
      <td>-0.044396</td>
      <td>1.606541e-01</td>
      <td>12.0</td>
      <td>0.020833</td>
      <td>False</td>
      <td>a</td>
    </tr>
    <tr>
      <th>5</th>
      <td>var_c_0_logit_code_a</td>
      <td>var_c_0</td>
      <td>logit_code</td>
      <td>True</td>
      <td>True</td>
      <td>0.171044</td>
      <td>5.260217e-08</td>
      <td>12.0</td>
      <td>0.020833</td>
      <td>True</td>
      <td>a</td>
    </tr>
    <tr>
      <th>6</th>
      <td>var_c_0_logit_code_b</td>
      <td>var_c_0</td>
      <td>logit_code</td>
      <td>True</td>
      <td>True</td>
      <td>-0.107505</td>
      <td>6.610536e-04</td>
      <td>12.0</td>
      <td>0.020833</td>
      <td>False</td>
      <td>a</td>
    </tr>
    <tr>
      <th>7</th>
      <td>var_c_0_prevalence_code</td>
      <td>var_c_0</td>
      <td>prevalence_code</td>
      <td>False</td>
      <td>True</td>
      <td>-0.060871</td>
      <td>5.431808e-02</td>
      <td>4.0</td>
      <td>0.062500</td>
      <td>True</td>
      <td>a</td>
    </tr>
    <tr>
      <th>8</th>
      <td>var_c_0_lev_b</td>
      <td>var_c_0</td>
      <td>indicator_code</td>
      <td>False</td>
      <td>True</td>
      <td>-0.100126</td>
      <td>1.522917e-03</td>
      <td>12.0</td>
      <td>0.020833</td>
      <td>True</td>
      <td>a</td>
    </tr>
    <tr>
      <th>9</th>
      <td>var_c_0_lev_a</td>
      <td>var_c_0</td>
      <td>indicator_code</td>
      <td>False</td>
      <td>True</td>
      <td>0.184872</td>
      <td>3.868879e-09</td>
      <td>12.0</td>
      <td>0.020833</td>
      <td>True</td>
      <td>a</td>
    </tr>
    <tr>
      <th>10</th>
      <td>var_c_0_lev_c</td>
      <td>var_c_0</td>
      <td>indicator_code</td>
      <td>False</td>
      <td>True</td>
      <td>-0.083552</td>
      <td>8.206181e-03</td>
      <td>12.0</td>
      <td>0.020833</td>
      <td>True</td>
      <td>a</td>
    </tr>
    <tr>
      <th>11</th>
      <td>var_c_1_logit_code_c</td>
      <td>var_c_1</td>
      <td>logit_code</td>
      <td>True</td>
      <td>True</td>
      <td>-0.035772</td>
      <td>2.584166e-01</td>
      <td>12.0</td>
      <td>0.020833</td>
      <td>False</td>
      <td>a</td>
    </tr>
    <tr>
      <th>12</th>
      <td>var_c_1_logit_code_a</td>
      <td>var_c_1</td>
      <td>logit_code</td>
      <td>True</td>
      <td>True</td>
      <td>0.100804</td>
      <td>1.413720e-03</td>
      <td>12.0</td>
      <td>0.020833</td>
      <td>True</td>
      <td>a</td>
    </tr>
    <tr>
      <th>13</th>
      <td>var_c_1_logit_code_b</td>
      <td>var_c_1</td>
      <td>logit_code</td>
      <td>True</td>
      <td>True</td>
      <td>0.006520</td>
      <td>8.368533e-01</td>
      <td>12.0</td>
      <td>0.020833</td>
      <td>False</td>
      <td>a</td>
    </tr>
    <tr>
      <th>14</th>
      <td>var_c_1_prevalence_code</td>
      <td>var_c_1</td>
      <td>prevalence_code</td>
      <td>False</td>
      <td>True</td>
      <td>0.096485</td>
      <td>2.254798e-03</td>
      <td>4.0</td>
      <td>0.062500</td>
      <td>True</td>
      <td>a</td>
    </tr>
    <tr>
      <th>15</th>
      <td>var_c_1_lev_a</td>
      <td>var_c_1</td>
      <td>indicator_code</td>
      <td>False</td>
      <td>True</td>
      <td>0.120567</td>
      <td>1.324653e-04</td>
      <td>12.0</td>
      <td>0.020833</td>
      <td>True</td>
      <td>a</td>
    </tr>
    <tr>
      <th>16</th>
      <td>var_c_1_lev_b</td>
      <td>var_c_1</td>
      <td>indicator_code</td>
      <td>False</td>
      <td>True</td>
      <td>-0.070889</td>
      <td>2.497855e-02</td>
      <td>12.0</td>
      <td>0.020833</td>
      <td>False</td>
      <td>a</td>
    </tr>
    <tr>
      <th>17</th>
      <td>var_c_1_lev_c</td>
      <td>var_c_1</td>
      <td>indicator_code</td>
      <td>False</td>
      <td>True</td>
      <td>-0.051275</td>
      <td>1.051273e-01</td>
      <td>12.0</td>
      <td>0.020833</td>
      <td>False</td>
      <td>a</td>
    </tr>
    <tr>
      <th>18</th>
      <td>noise_c_0_logit_code_c</td>
      <td>noise_c_0</td>
      <td>logit_code</td>
      <td>True</td>
      <td>True</td>
      <td>0.021457</td>
      <td>4.979168e-01</td>
      <td>12.0</td>
      <td>0.020833</td>
      <td>False</td>
      <td>a</td>
    </tr>
    <tr>
      <th>19</th>
      <td>noise_c_0_logit_code_a</td>
      <td>noise_c_0</td>
      <td>logit_code</td>
      <td>True</td>
      <td>True</td>
      <td>-0.074075</td>
      <td>1.914267e-02</td>
      <td>12.0</td>
      <td>0.020833</td>
      <td>False</td>
      <td>a</td>
    </tr>
    <tr>
      <th>20</th>
      <td>noise_c_0_logit_code_b</td>
      <td>noise_c_0</td>
      <td>logit_code</td>
      <td>True</td>
      <td>True</td>
      <td>0.006383</td>
      <td>8.402249e-01</td>
      <td>12.0</td>
      <td>0.020833</td>
      <td>False</td>
      <td>a</td>
    </tr>
    <tr>
      <th>21</th>
      <td>noise_c_0_prevalence_code</td>
      <td>noise_c_0</td>
      <td>prevalence_code</td>
      <td>False</td>
      <td>True</td>
      <td>-0.003217</td>
      <td>9.190732e-01</td>
      <td>4.0</td>
      <td>0.062500</td>
      <td>False</td>
      <td>a</td>
    </tr>
    <tr>
      <th>22</th>
      <td>noise_c_0_lev_b</td>
      <td>noise_c_0</td>
      <td>indicator_code</td>
      <td>False</td>
      <td>True</td>
      <td>-0.000326</td>
      <td>9.917733e-01</td>
      <td>12.0</td>
      <td>0.020833</td>
      <td>False</td>
      <td>a</td>
    </tr>
    <tr>
      <th>23</th>
      <td>noise_c_0_lev_a</td>
      <td>noise_c_0</td>
      <td>indicator_code</td>
      <td>False</td>
      <td>True</td>
      <td>-0.003919</td>
      <td>9.014892e-01</td>
      <td>12.0</td>
      <td>0.020833</td>
      <td>False</td>
      <td>a</td>
    </tr>
    <tr>
      <th>24</th>
      <td>noise_c_0_lev_c</td>
      <td>noise_c_0</td>
      <td>indicator_code</td>
      <td>False</td>
      <td>True</td>
      <td>0.004287</td>
      <td>8.922984e-01</td>
      <td>12.0</td>
      <td>0.020833</td>
      <td>False</td>
      <td>a</td>
    </tr>
    <tr>
      <th>25</th>
      <td>noise_c_1_logit_code_c</td>
      <td>noise_c_1</td>
      <td>logit_code</td>
      <td>True</td>
      <td>True</td>
      <td>-0.014232</td>
      <td>6.530656e-01</td>
      <td>12.0</td>
      <td>0.020833</td>
      <td>False</td>
      <td>a</td>
    </tr>
    <tr>
      <th>26</th>
      <td>noise_c_1_logit_code_a</td>
      <td>noise_c_1</td>
      <td>logit_code</td>
      <td>True</td>
      <td>True</td>
      <td>0.013643</td>
      <td>6.665379e-01</td>
      <td>12.0</td>
      <td>0.020833</td>
      <td>False</td>
      <td>a</td>
    </tr>
    <tr>
      <th>27</th>
      <td>noise_c_1_logit_code_b</td>
      <td>noise_c_1</td>
      <td>logit_code</td>
      <td>True</td>
      <td>True</td>
      <td>0.004169</td>
      <td>8.952364e-01</td>
      <td>12.0</td>
      <td>0.020833</td>
      <td>False</td>
      <td>a</td>
    </tr>
    <tr>
      <th>28</th>
      <td>noise_c_1_prevalence_code</td>
      <td>noise_c_1</td>
      <td>prevalence_code</td>
      <td>False</td>
      <td>True</td>
      <td>0.002054</td>
      <td>9.482694e-01</td>
      <td>4.0</td>
      <td>0.062500</td>
      <td>False</td>
      <td>a</td>
    </tr>
    <tr>
      <th>29</th>
      <td>noise_c_1_lev_c</td>
      <td>noise_c_1</td>
      <td>indicator_code</td>
      <td>False</td>
      <td>True</td>
      <td>-0.013034</td>
      <td>6.805904e-01</td>
      <td>12.0</td>
      <td>0.020833</td>
      <td>False</td>
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
      <th>66</th>
      <td>noise_n_0</td>
      <td>noise_n_0</td>
      <td>clean_copy</td>
      <td>False</td>
      <td>True</td>
      <td>-0.005021</td>
      <td>8.739920e-01</td>
      <td>4.0</td>
      <td>0.062500</td>
      <td>False</td>
      <td>c</td>
    </tr>
    <tr>
      <th>67</th>
      <td>noise_n_1</td>
      <td>noise_n_1</td>
      <td>clean_copy</td>
      <td>False</td>
      <td>True</td>
      <td>0.022103</td>
      <td>4.850783e-01</td>
      <td>4.0</td>
      <td>0.062500</td>
      <td>False</td>
      <td>c</td>
    </tr>
    <tr>
      <th>68</th>
      <td>var_c_0_logit_code_c</td>
      <td>var_c_0</td>
      <td>logit_code</td>
      <td>True</td>
      <td>True</td>
      <td>0.155713</td>
      <td>7.492672e-07</td>
      <td>12.0</td>
      <td>0.020833</td>
      <td>True</td>
      <td>c</td>
    </tr>
    <tr>
      <th>69</th>
      <td>var_c_0_logit_code_a</td>
      <td>var_c_0</td>
      <td>logit_code</td>
      <td>True</td>
      <td>True</td>
      <td>-0.046828</td>
      <td>1.389282e-01</td>
      <td>12.0</td>
      <td>0.020833</td>
      <td>False</td>
      <td>c</td>
    </tr>
    <tr>
      <th>70</th>
      <td>var_c_0_logit_code_b</td>
      <td>var_c_0</td>
      <td>logit_code</td>
      <td>True</td>
      <td>True</td>
      <td>-0.089015</td>
      <td>4.847630e-03</td>
      <td>12.0</td>
      <td>0.020833</td>
      <td>False</td>
      <td>c</td>
    </tr>
    <tr>
      <th>71</th>
      <td>var_c_0_prevalence_code</td>
      <td>var_c_0</td>
      <td>prevalence_code</td>
      <td>False</td>
      <td>True</td>
      <td>-0.139135</td>
      <td>1.006197e-05</td>
      <td>4.0</td>
      <td>0.062500</td>
      <td>True</td>
      <td>c</td>
    </tr>
    <tr>
      <th>72</th>
      <td>var_c_0_lev_b</td>
      <td>var_c_0</td>
      <td>indicator_code</td>
      <td>False</td>
      <td>True</td>
      <td>-0.109217</td>
      <td>5.405943e-04</td>
      <td>12.0</td>
      <td>0.020833</td>
      <td>True</td>
      <td>c</td>
    </tr>
    <tr>
      <th>73</th>
      <td>var_c_0_lev_a</td>
      <td>var_c_0</td>
      <td>indicator_code</td>
      <td>False</td>
      <td>True</td>
      <td>-0.064276</td>
      <td>4.213767e-02</td>
      <td>12.0</td>
      <td>0.020833</td>
      <td>False</td>
      <td>c</td>
    </tr>
    <tr>
      <th>74</th>
      <td>var_c_0_lev_c</td>
      <td>var_c_0</td>
      <td>indicator_code</td>
      <td>False</td>
      <td>True</td>
      <td>0.176511</td>
      <td>1.920949e-08</td>
      <td>12.0</td>
      <td>0.020833</td>
      <td>True</td>
      <td>c</td>
    </tr>
    <tr>
      <th>75</th>
      <td>var_c_1_logit_code_c</td>
      <td>var_c_1</td>
      <td>logit_code</td>
      <td>True</td>
      <td>True</td>
      <td>0.227716</td>
      <td>3.146820e-13</td>
      <td>12.0</td>
      <td>0.020833</td>
      <td>True</td>
      <td>c</td>
    </tr>
    <tr>
      <th>76</th>
      <td>var_c_1_logit_code_a</td>
      <td>var_c_1</td>
      <td>logit_code</td>
      <td>True</td>
      <td>True</td>
      <td>-0.075404</td>
      <td>1.708388e-02</td>
      <td>12.0</td>
      <td>0.020833</td>
      <td>False</td>
      <td>c</td>
    </tr>
    <tr>
      <th>77</th>
      <td>var_c_1_logit_code_b</td>
      <td>var_c_1</td>
      <td>logit_code</td>
      <td>True</td>
      <td>True</td>
      <td>-0.218394</td>
      <td>2.909484e-12</td>
      <td>12.0</td>
      <td>0.020833</td>
      <td>False</td>
      <td>c</td>
    </tr>
    <tr>
      <th>78</th>
      <td>var_c_1_prevalence_code</td>
      <td>var_c_1</td>
      <td>prevalence_code</td>
      <td>False</td>
      <td>True</td>
      <td>-0.211188</td>
      <td>1.517424e-11</td>
      <td>4.0</td>
      <td>0.062500</td>
      <td>True</td>
      <td>c</td>
    </tr>
    <tr>
      <th>79</th>
      <td>var_c_1_lev_a</td>
      <td>var_c_1</td>
      <td>indicator_code</td>
      <td>False</td>
      <td>True</td>
      <td>-0.113476</td>
      <td>3.237033e-04</td>
      <td>12.0</td>
      <td>0.020833</td>
      <td>True</td>
      <td>c</td>
    </tr>
    <tr>
      <th>80</th>
      <td>var_c_1_lev_b</td>
      <td>var_c_1</td>
      <td>indicator_code</td>
      <td>False</td>
      <td>True</td>
      <td>-0.126592</td>
      <td>5.961234e-05</td>
      <td>12.0</td>
      <td>0.020833</td>
      <td>True</td>
      <td>c</td>
    </tr>
    <tr>
      <th>81</th>
      <td>var_c_1_lev_c</td>
      <td>var_c_1</td>
      <td>indicator_code</td>
      <td>False</td>
      <td>True</td>
      <td>0.243711</td>
      <td>5.482387e-15</td>
      <td>12.0</td>
      <td>0.020833</td>
      <td>True</td>
      <td>c</td>
    </tr>
    <tr>
      <th>82</th>
      <td>noise_c_0_logit_code_c</td>
      <td>noise_c_0</td>
      <td>logit_code</td>
      <td>True</td>
      <td>True</td>
      <td>-0.023104</td>
      <td>4.655240e-01</td>
      <td>12.0</td>
      <td>0.020833</td>
      <td>False</td>
      <td>c</td>
    </tr>
    <tr>
      <th>83</th>
      <td>noise_c_0_logit_code_a</td>
      <td>noise_c_0</td>
      <td>logit_code</td>
      <td>True</td>
      <td>True</td>
      <td>0.058404</td>
      <td>6.486682e-02</td>
      <td>12.0</td>
      <td>0.020833</td>
      <td>False</td>
      <td>c</td>
    </tr>
    <tr>
      <th>84</th>
      <td>noise_c_0_logit_code_b</td>
      <td>noise_c_0</td>
      <td>logit_code</td>
      <td>True</td>
      <td>True</td>
      <td>0.003788</td>
      <td>9.047818e-01</td>
      <td>12.0</td>
      <td>0.020833</td>
      <td>False</td>
      <td>c</td>
    </tr>
    <tr>
      <th>85</th>
      <td>noise_c_0_prevalence_code</td>
      <td>noise_c_0</td>
      <td>prevalence_code</td>
      <td>False</td>
      <td>True</td>
      <td>0.039606</td>
      <td>2.107948e-01</td>
      <td>4.0</td>
      <td>0.062500</td>
      <td>False</td>
      <td>c</td>
    </tr>
    <tr>
      <th>86</th>
      <td>noise_c_0_lev_b</td>
      <td>noise_c_0</td>
      <td>indicator_code</td>
      <td>False</td>
      <td>True</td>
      <td>0.020221</td>
      <td>5.230213e-01</td>
      <td>12.0</td>
      <td>0.020833</td>
      <td>False</td>
      <td>c</td>
    </tr>
    <tr>
      <th>87</th>
      <td>noise_c_0_lev_a</td>
      <td>noise_c_0</td>
      <td>indicator_code</td>
      <td>False</td>
      <td>True</td>
      <td>0.022470</td>
      <td>4.778443e-01</td>
      <td>12.0</td>
      <td>0.020833</td>
      <td>False</td>
      <td>c</td>
    </tr>
    <tr>
      <th>88</th>
      <td>noise_c_0_lev_c</td>
      <td>noise_c_0</td>
      <td>indicator_code</td>
      <td>False</td>
      <td>True</td>
      <td>-0.043193</td>
      <td>1.723135e-01</td>
      <td>12.0</td>
      <td>0.020833</td>
      <td>False</td>
      <td>c</td>
    </tr>
    <tr>
      <th>89</th>
      <td>noise_c_1_logit_code_c</td>
      <td>noise_c_1</td>
      <td>logit_code</td>
      <td>True</td>
      <td>True</td>
      <td>-0.023665</td>
      <td>4.547524e-01</td>
      <td>12.0</td>
      <td>0.020833</td>
      <td>False</td>
      <td>c</td>
    </tr>
    <tr>
      <th>90</th>
      <td>noise_c_1_logit_code_a</td>
      <td>noise_c_1</td>
      <td>logit_code</td>
      <td>True</td>
      <td>True</td>
      <td>-0.015457</td>
      <td>6.254064e-01</td>
      <td>12.0</td>
      <td>0.020833</td>
      <td>False</td>
      <td>c</td>
    </tr>
    <tr>
      <th>91</th>
      <td>noise_c_1_logit_code_b</td>
      <td>noise_c_1</td>
      <td>logit_code</td>
      <td>True</td>
      <td>True</td>
      <td>0.055366</td>
      <td>8.012076e-02</td>
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

