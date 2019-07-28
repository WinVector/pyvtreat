

```python

```


```python
# To install:
!pip install /Users/johnmount/Documents/work/pyvtreat/pkg/dist/vtreat-0.1.tar.gz
#!pip install https://github.com/WinVector/pyvtreat/raw/master/pkg/dist/vtreat-0.1.tar.gz
```

    Processing /Users/johnmount/Documents/work/pyvtreat/pkg/dist/vtreat-0.1.tar.gz
    Requirement already satisfied: numpy in /Users/johnmount/anaconda3/envs/aiAcademy/lib/python3.7/site-packages (from vtreat==0.1) (1.16.4)
    Requirement already satisfied: pandas in /Users/johnmount/anaconda3/envs/aiAcademy/lib/python3.7/site-packages (from vtreat==0.1) (0.24.2)
    Requirement already satisfied: statistics in /Users/johnmount/anaconda3/envs/aiAcademy/lib/python3.7/site-packages (from vtreat==0.1) (1.0.3.5)
    Requirement already satisfied: scipy in /Users/johnmount/anaconda3/envs/aiAcademy/lib/python3.7/site-packages (from vtreat==0.1) (1.2.1)
    Requirement already satisfied: python-dateutil>=2.5.0 in /Users/johnmount/anaconda3/envs/aiAcademy/lib/python3.7/site-packages (from pandas->vtreat==0.1) (2.8.0)
    Requirement already satisfied: pytz>=2011k in /Users/johnmount/anaconda3/envs/aiAcademy/lib/python3.7/site-packages (from pandas->vtreat==0.1) (2019.1)
    Requirement already satisfied: docutils>=0.3 in /Users/johnmount/anaconda3/envs/aiAcademy/lib/python3.7/site-packages (from statistics->vtreat==0.1) (0.14)
    Requirement already satisfied: six>=1.5 in /Users/johnmount/anaconda3/envs/aiAcademy/lib/python3.7/site-packages (from python-dateutil>=2.5.0->pandas->vtreat==0.1) (1.12.0)
    Building wheels for collected packages: vtreat
      Building wheel for vtreat (setup.py) ... [?25ldone
    [?25h  Stored in directory: /Users/johnmount/Library/Caches/pip/wheels/cf/06/fc/6b2552717486fb6401f19308eec24381555e456e3bd9cfb103
    Successfully built vtreat
    Installing collected packages: vtreat
      Found existing installation: vtreat 0.1
        Uninstalling vtreat-0.1:
          Successfully uninstalled vtreat-0.1
    Successfully installed vtreat-0.1



```python
import numpy.random
import pandas
import seaborn
import vtreat # https://github.com/WinVector/pyvtreat


numpy.random.seed(235)
zip = ['z' + str(i+1).zfill(5) for i in range(15)]
d = pandas.DataFrame({'zip':numpy.random.choice(zip, size=1000)})
d["const"] = 1
d["const2"]= "b"
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
      <th>zip</th>
      <th>const</th>
      <th>const2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>z00009</td>
      <td>1</td>
      <td>b</td>
    </tr>
    <tr>
      <th>1</th>
      <td>z00015</td>
      <td>1</td>
      <td>b</td>
    </tr>
    <tr>
      <th>2</th>
      <td>z00002</td>
      <td>1</td>
      <td>b</td>
    </tr>
    <tr>
      <th>3</th>
      <td>z00006</td>
      <td>1</td>
      <td>b</td>
    </tr>
    <tr>
      <th>4</th>
      <td>z00013</td>
      <td>1</td>
      <td>b</td>
    </tr>
  </tbody>
</table>
</div>




```python
transform = vtreat.UnsupervisedTreatment(
    params=vtreat.vtreat_parameters({
        'indicator_min_fracton': 0.01,
    }))
transform.params_
```




    {'use_hierarchical_estimate': True,
     'coders': {'clean_copy',
      'deviance_code',
      'impact_code',
      'indicator_code',
      'logit_code',
      'missing_indicator',
      'prevalence_code'},
     'filter_to_recommended': True,
     'indicator_min_fracton': 0.01,
     'cross_validation_plan': <vtreat.KWayCrossPlan at 0x104b2a2b0>,
     'cross_validation_k': 5}




```python
d_treated = transform.fit_transform(d)
d_treated.head()
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
      <th>zip_prevalence_code</th>
      <th>zip_lev_z00009</th>
      <th>zip_lev_z00007</th>
      <th>zip_lev_z00011</th>
      <th>zip_lev_z00013</th>
      <th>zip_lev_z00003</th>
      <th>zip_lev_z00008</th>
      <th>zip_lev_z00015</th>
      <th>zip_lev_z00004</th>
      <th>zip_lev_z00005</th>
      <th>zip_lev_z00001</th>
      <th>zip_lev_z00014</th>
      <th>zip_lev_z00012</th>
      <th>zip_lev_z00006</th>
      <th>zip_lev_z00002</th>
      <th>zip_lev_z00010</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.083</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.064</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.057</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.060</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.071</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
transform.score_frame_
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
      <th>recommended</th>
      <th>vcount</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>zip_prevalence_code</td>
      <td>zip</td>
      <td>prevalence_code</td>
      <td>False</td>
      <td>True</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>True</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>zip_lev_z00009</td>
      <td>zip</td>
      <td>indicator_code</td>
      <td>False</td>
      <td>True</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>True</td>
      <td>16.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>zip_lev_z00007</td>
      <td>zip</td>
      <td>indicator_code</td>
      <td>False</td>
      <td>True</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>True</td>
      <td>16.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>zip_lev_z00011</td>
      <td>zip</td>
      <td>indicator_code</td>
      <td>False</td>
      <td>True</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>True</td>
      <td>16.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>zip_lev_z00013</td>
      <td>zip</td>
      <td>indicator_code</td>
      <td>False</td>
      <td>True</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>True</td>
      <td>16.0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>zip_lev_z00003</td>
      <td>zip</td>
      <td>indicator_code</td>
      <td>False</td>
      <td>True</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>True</td>
      <td>16.0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>zip_lev_z00008</td>
      <td>zip</td>
      <td>indicator_code</td>
      <td>False</td>
      <td>True</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>True</td>
      <td>16.0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>zip_lev_z00015</td>
      <td>zip</td>
      <td>indicator_code</td>
      <td>False</td>
      <td>True</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>True</td>
      <td>16.0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>zip_lev_z00004</td>
      <td>zip</td>
      <td>indicator_code</td>
      <td>False</td>
      <td>True</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>True</td>
      <td>16.0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>zip_lev_z00005</td>
      <td>zip</td>
      <td>indicator_code</td>
      <td>False</td>
      <td>True</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>True</td>
      <td>16.0</td>
    </tr>
    <tr>
      <th>10</th>
      <td>zip_lev_z00001</td>
      <td>zip</td>
      <td>indicator_code</td>
      <td>False</td>
      <td>True</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>True</td>
      <td>16.0</td>
    </tr>
    <tr>
      <th>11</th>
      <td>zip_lev_z00014</td>
      <td>zip</td>
      <td>indicator_code</td>
      <td>False</td>
      <td>True</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>True</td>
      <td>16.0</td>
    </tr>
    <tr>
      <th>12</th>
      <td>zip_lev_z00012</td>
      <td>zip</td>
      <td>indicator_code</td>
      <td>False</td>
      <td>True</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>True</td>
      <td>16.0</td>
    </tr>
    <tr>
      <th>13</th>
      <td>zip_lev_z00006</td>
      <td>zip</td>
      <td>indicator_code</td>
      <td>False</td>
      <td>True</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>True</td>
      <td>16.0</td>
    </tr>
    <tr>
      <th>14</th>
      <td>zip_lev_z00002</td>
      <td>zip</td>
      <td>indicator_code</td>
      <td>False</td>
      <td>True</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>True</td>
      <td>16.0</td>
    </tr>
    <tr>
      <th>15</th>
      <td>zip_lev_z00010</td>
      <td>zip</td>
      <td>indicator_code</td>
      <td>False</td>
      <td>True</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>True</td>
      <td>16.0</td>
    </tr>
    <tr>
      <th>16</th>
      <td>const2_prevalence_code</td>
      <td>const2</td>
      <td>prevalence_code</td>
      <td>False</td>
      <td>False</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>False</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>17</th>
      <td>const2_lev_b</td>
      <td>const2</td>
      <td>indicator_code</td>
      <td>False</td>
      <td>False</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>False</td>
      <td>16.0</td>
    </tr>
  </tbody>
</table>
</div>




```python

```
