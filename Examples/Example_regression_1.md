

```python
!pip install /Users/johnmount/Documents/work/pyvtreat/dist/vtreat-0.1.tar.gz
#!pip install https://github.com/WinVector/pyvtreat/raw/master/dist/vtreat-0.1.tar.gz
```

    Processing /Users/johnmount/Documents/work/pyvtreat/dist/vtreat-0.1.tar.gz
    Requirement already satisfied: numpy in /Users/johnmount/anaconda3/envs/aiAcademy/lib/python3.7/site-packages (from vtreat==0.1) (1.16.4)
    Requirement already satisfied: pandas in /Users/johnmount/anaconda3/envs/aiAcademy/lib/python3.7/site-packages (from vtreat==0.1) (0.24.2)
    Requirement already satisfied: statistics in /Users/johnmount/anaconda3/envs/aiAcademy/lib/python3.7/site-packages (from vtreat==0.1) (1.0.3.5)
    Requirement already satisfied: scipy in /Users/johnmount/anaconda3/envs/aiAcademy/lib/python3.7/site-packages (from vtreat==0.1) (1.2.1)
    Requirement already satisfied: pytz>=2011k in /Users/johnmount/anaconda3/envs/aiAcademy/lib/python3.7/site-packages (from pandas->vtreat==0.1) (2019.1)
    Requirement already satisfied: python-dateutil>=2.5.0 in /Users/johnmount/anaconda3/envs/aiAcademy/lib/python3.7/site-packages (from pandas->vtreat==0.1) (2.8.0)
    Requirement already satisfied: docutils>=0.3 in /Users/johnmount/anaconda3/envs/aiAcademy/lib/python3.7/site-packages (from statistics->vtreat==0.1) (0.14)
    Requirement already satisfied: six>=1.5 in /Users/johnmount/anaconda3/envs/aiAcademy/lib/python3.7/site-packages (from python-dateutil>=2.5.0->pandas->vtreat==0.1) (1.12.0)
    Building wheels for collected packages: vtreat
      Building wheel for vtreat (setup.py) ... [?25ldone
    [?25h  Stored in directory: /Users/johnmount/Library/Caches/pip/wheels/28/d1/8a/f8f4ee7c515a6c18d95d64f4d49327fe498b9e6e23d04c7159
    Successfully built vtreat
    Installing collected packages: vtreat
      Found existing installation: vtreat 0.1
        Uninstalling vtreat-0.1:
          Successfully uninstalled vtreat-0.1
    Successfully installed vtreat-0.1



```python
import vtreat.util
```


```python
vtreat.util.k_way_cross_plan(10,4)
```




    [{'train': [0, 1, 2, 4, 6, 7, 9], 'app': [3, 5, 8]},
     {'train': [1, 2, 3, 4, 5, 7, 8], 'app': [0, 6, 9]},
     {'train': [0, 1, 2, 3, 5, 6, 8, 9], 'app': [4, 7]},
     {'train': [0, 3, 4, 5, 6, 7, 8, 9], 'app': [1, 2]}]




```python
import vtreat
```


```python
plan = vtreat.numeric_outcome_treatment(outcomename="y")
```


```python
import pandas
import numpy
```


```python
# from https://pandas.pydata.org/pandas-docs/stable/user_guide/missing_data.html
df = pandas.DataFrame(numpy.random.randn(5, 3), index=['a', 'c', 'e', 'f', 'h'],
                      columns=['one', 'two', 'three'])
df['four'] = 'foo'
df['five'] = df['one'] > 0
df2 = df.reindex(['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'])
df2.reset_index(inplace=True, drop=True)
df2["y"] = range(df2.shape[0])
df2.loc[3, "four"] = "blog"
```


```python
df2
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
      <th>one</th>
      <th>two</th>
      <th>three</th>
      <th>four</th>
      <th>five</th>
      <th>y</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-1.286413</td>
      <td>0.436261</td>
      <td>-0.123267</td>
      <td>foo</td>
      <td>False</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-1.304813</td>
      <td>0.406402</td>
      <td>0.441027</td>
      <td>foo</td>
      <td>False</td>
      <td>2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>blog</td>
      <td>NaN</td>
      <td>3</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.785464</td>
      <td>0.904092</td>
      <td>0.920157</td>
      <td>foo</td>
      <td>True</td>
      <td>4</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.794746</td>
      <td>-0.866337</td>
      <td>0.223271</td>
      <td>foo</td>
      <td>True</td>
      <td>5</td>
    </tr>
    <tr>
      <th>6</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>6</td>
    </tr>
    <tr>
      <th>7</th>
      <td>-0.261195</td>
      <td>-1.108357</td>
      <td>-0.646967</td>
      <td>foo</td>
      <td>False</td>
      <td>7</td>
    </tr>
  </tbody>
</table>
</div>




```python
plan.fit(df2, df2["y"])
```




    <vtreat.numeric_outcome_treatment at 0x1a14773358>




```python
res = plan.transform(df2)
res
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
      <th>one_is_bad</th>
      <th>two_is_bad</th>
      <th>three_is_bad</th>
      <th>four_is_bad</th>
      <th>five_is_bad</th>
      <th>one</th>
      <th>two</th>
      <th>three</th>
      <th>five</th>
      <th>four_impact_code</th>
      <th>four_prevalence_code</th>
      <th>four_lev_foo</th>
      <th>four_lev__NA_</th>
      <th>four_lev_blog</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>-1.286413</td>
      <td>0.436261</td>
      <td>-0.123267</td>
      <td>0.0</td>
      <td>0.007412</td>
      <td>0.500</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>-0.254442</td>
      <td>-0.045588</td>
      <td>0.162844</td>
      <td>0.4</td>
      <td>0.000000</td>
      <td>0.125</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>-1.304813</td>
      <td>0.406402</td>
      <td>0.441027</td>
      <td>0.0</td>
      <td>0.007412</td>
      <td>0.500</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>-0.254442</td>
      <td>-0.045588</td>
      <td>0.162844</td>
      <td>0.4</td>
      <td>0.000000</td>
      <td>0.000</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.785464</td>
      <td>0.904092</td>
      <td>0.920157</td>
      <td>1.0</td>
      <td>0.007412</td>
      <td>0.500</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>5</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.794746</td>
      <td>-0.866337</td>
      <td>0.223271</td>
      <td>1.0</td>
      <td>0.007412</td>
      <td>0.500</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>6</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>-0.254442</td>
      <td>-0.045588</td>
      <td>0.162844</td>
      <td>0.4</td>
      <td>0.000000</td>
      <td>0.125</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>7</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>-0.261195</td>
      <td>-1.108357</td>
      <td>-0.646967</td>
      <td>0.0</td>
      <td>0.007412</td>
      <td>0.500</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
res2 = plan.fit_transform(df2, df2["y"])
res2
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
      <th>one_is_bad</th>
      <th>two_is_bad</th>
      <th>three_is_bad</th>
      <th>four_is_bad</th>
      <th>five_is_bad</th>
      <th>one</th>
      <th>two</th>
      <th>three</th>
      <th>five</th>
      <th>four_impact_code</th>
      <th>four_prevalence_code</th>
      <th>four_lev_foo</th>
      <th>four_lev__NA_</th>
      <th>four_lev_blog</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>-1.286413</td>
      <td>0.436261</td>
      <td>-0.123267</td>
      <td>0.0</td>
      <td>5.134640e-01</td>
      <td>0.500</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>-0.254442</td>
      <td>-0.045588</td>
      <td>0.162844</td>
      <td>0.4</td>
      <td>-4.440892e-16</td>
      <td>0.125</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>-1.304813</td>
      <td>0.406402</td>
      <td>0.441027</td>
      <td>0.0</td>
      <td>5.134640e-01</td>
      <td>0.500</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>-0.254442</td>
      <td>-0.045588</td>
      <td>0.162844</td>
      <td>0.4</td>
      <td>0.000000e+00</td>
      <td>0.000</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.785464</td>
      <td>0.904092</td>
      <td>0.920157</td>
      <td>1.0</td>
      <td>-7.037040e-02</td>
      <td>0.500</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>5</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.794746</td>
      <td>-0.866337</td>
      <td>0.223271</td>
      <td>1.0</td>
      <td>-1.173304e-03</td>
      <td>0.500</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>6</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>-0.254442</td>
      <td>-0.045588</td>
      <td>0.162844</td>
      <td>0.4</td>
      <td>0.000000e+00</td>
      <td>0.125</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>7</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>-0.261195</td>
      <td>-1.108357</td>
      <td>-0.646967</td>
      <td>0.0</td>
      <td>-7.037040e-02</td>
      <td>0.500</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
plan.transform(df2)
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
      <th>one_is_bad</th>
      <th>two_is_bad</th>
      <th>three_is_bad</th>
      <th>four_is_bad</th>
      <th>five_is_bad</th>
      <th>one</th>
      <th>two</th>
      <th>three</th>
      <th>five</th>
      <th>four_impact_code</th>
      <th>four_prevalence_code</th>
      <th>four_lev_foo</th>
      <th>four_lev__NA_</th>
      <th>four_lev_blog</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>-1.286413</td>
      <td>0.436261</td>
      <td>-0.123267</td>
      <td>0.0</td>
      <td>0.007412</td>
      <td>0.500</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>-0.254442</td>
      <td>-0.045588</td>
      <td>0.162844</td>
      <td>0.4</td>
      <td>0.000000</td>
      <td>0.125</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>-1.304813</td>
      <td>0.406402</td>
      <td>0.441027</td>
      <td>0.0</td>
      <td>0.007412</td>
      <td>0.500</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>-0.254442</td>
      <td>-0.045588</td>
      <td>0.162844</td>
      <td>0.4</td>
      <td>0.000000</td>
      <td>0.000</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.785464</td>
      <td>0.904092</td>
      <td>0.920157</td>
      <td>1.0</td>
      <td>0.007412</td>
      <td>0.500</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>5</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.794746</td>
      <td>-0.866337</td>
      <td>0.223271</td>
      <td>1.0</td>
      <td>0.007412</td>
      <td>0.500</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>6</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>-0.254442</td>
      <td>-0.045588</td>
      <td>0.162844</td>
      <td>0.4</td>
      <td>0.000000</td>
      <td>0.125</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>7</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>-0.261195</td>
      <td>-1.108357</td>
      <td>-0.646967</td>
      <td>0.0</td>
      <td>0.007412</td>
      <td>0.500</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
plan.score_frame_
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
      <th>PearsonR</th>
      <th>significance</th>
      <th>vcount</th>
      <th>recommended</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>one_is_bad</td>
      <td>missing_indicator</td>
      <td>False</td>
      <td>-0.056344</td>
      <td>0.894579</td>
      <td>5.0</td>
      <td>False</td>
    </tr>
    <tr>
      <th>1</th>
      <td>two_is_bad</td>
      <td>missing_indicator</td>
      <td>False</td>
      <td>-0.056344</td>
      <td>0.894579</td>
      <td>5.0</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2</th>
      <td>three_is_bad</td>
      <td>missing_indicator</td>
      <td>False</td>
      <td>-0.056344</td>
      <td>0.894579</td>
      <td>5.0</td>
      <td>False</td>
    </tr>
    <tr>
      <th>3</th>
      <td>four_is_bad</td>
      <td>missing_indicator</td>
      <td>False</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>5.0</td>
      <td>False</td>
    </tr>
    <tr>
      <th>4</th>
      <td>five_is_bad</td>
      <td>missing_indicator</td>
      <td>False</td>
      <td>-0.056344</td>
      <td>0.894579</td>
      <td>5.0</td>
      <td>False</td>
    </tr>
    <tr>
      <th>5</th>
      <td>one</td>
      <td>clean_copy</td>
      <td>False</td>
      <td>0.536904</td>
      <td>0.170038</td>
      <td>4.0</td>
      <td>False</td>
    </tr>
    <tr>
      <th>6</th>
      <td>two</td>
      <td>clean_copy</td>
      <td>False</td>
      <td>-0.595514</td>
      <td>0.119314</td>
      <td>4.0</td>
      <td>False</td>
    </tr>
    <tr>
      <th>7</th>
      <td>three</td>
      <td>clean_copy</td>
      <td>False</td>
      <td>-0.232899</td>
      <td>0.578848</td>
      <td>4.0</td>
      <td>False</td>
    </tr>
    <tr>
      <th>8</th>
      <td>five</td>
      <td>clean_copy</td>
      <td>False</td>
      <td>0.253546</td>
      <td>0.544582</td>
      <td>4.0</td>
      <td>False</td>
    </tr>
    <tr>
      <th>9</th>
      <td>four_impact_code</td>
      <td>impact_code</td>
      <td>True</td>
      <td>-0.663636</td>
      <td>0.072754</td>
      <td>1.0</td>
      <td>False</td>
    </tr>
    <tr>
      <th>10</th>
      <td>four_prevalence_code</td>
      <td>prevalance</td>
      <td>False</td>
      <td>0.066556</td>
      <td>0.875576</td>
      <td>1.0</td>
      <td>False</td>
    </tr>
    <tr>
      <th>11</th>
      <td>four_lev_foo</td>
      <td>indicator</td>
      <td>False</td>
      <td>0.056344</td>
      <td>0.894579</td>
      <td>3.0</td>
      <td>False</td>
    </tr>
    <tr>
      <th>12</th>
      <td>four_lev__NA_</td>
      <td>indicator</td>
      <td>False</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>3.0</td>
      <td>False</td>
    </tr>
    <tr>
      <th>13</th>
      <td>four_lev_blog</td>
      <td>indicator</td>
      <td>False</td>
      <td>-0.082479</td>
      <td>0.846053</td>
      <td>3.0</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
</div>




```python

```