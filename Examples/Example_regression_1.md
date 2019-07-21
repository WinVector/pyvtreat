

```python
!pip install https://github.com/WinVector/pyvtreat/raw/master/dist/vtreat-0.1.tar.gz
```

    Collecting https://github.com/WinVector/pyvtreat/raw/master/dist/vtreat-0.1.tar.gz
      Using cached https://github.com/WinVector/pyvtreat/raw/master/dist/vtreat-0.1.tar.gz
    Requirement already satisfied (use --upgrade to upgrade): vtreat==0.1 from https://github.com/WinVector/pyvtreat/raw/master/dist/vtreat-0.1.tar.gz in /Users/johnmount/anaconda3/envs/aiAcademy/lib/python3.7/site-packages
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
    [?25h  Stored in directory: /Users/johnmount/Library/Caches/pip/wheels/8f/d2/65/9ba24b7ca3ce85588a0aa24e1695428cafb12cbec807d43f65
    Successfully built vtreat



```python
import vtreat.util
```


```python
vtreat.util.k_way_cross_plan(10,4)
```




    [{'train': [0, 3, 4, 5, 6, 7, 8], 'app': [1, 2, 9]},
     {'train': [1, 2, 4, 5, 7, 8, 9], 'app': [0, 3, 6]},
     {'train': [0, 1, 2, 3, 5, 6, 7, 9], 'app': [4, 8]},
     {'train': [0, 1, 2, 3, 4, 6, 8, 9], 'app': [5, 7]}]




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
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>a</th>
      <td>1.892427</td>
      <td>-0.740102</td>
      <td>0.634821</td>
      <td>foo</td>
      <td>True</td>
    </tr>
    <tr>
      <th>b</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>c</th>
      <td>-0.713772</td>
      <td>1.282062</td>
      <td>0.411180</td>
      <td>foo</td>
      <td>False</td>
    </tr>
    <tr>
      <th>d</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>e</th>
      <td>-0.479925</td>
      <td>-1.075762</td>
      <td>0.323755</td>
      <td>foo</td>
      <td>False</td>
    </tr>
    <tr>
      <th>f</th>
      <td>1.012460</td>
      <td>-1.565270</td>
      <td>-0.165303</td>
      <td>foo</td>
      <td>True</td>
    </tr>
    <tr>
      <th>g</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>h</th>
      <td>-0.414984</td>
      <td>-0.532008</td>
      <td>-1.931477</td>
      <td>foo</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
</div>




```python
df2.reset_index(inplace=True, drop=True)
df2["y"] = range(df2.shape[0])
df2.loc[3, "four"] = "blog"
```


```python
#df2 = pandas.concat([df2, df2, df2, df2, df2], axis=0)
#df2.reset_index(inplace=True, drop=True)
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
      <td>1.892427</td>
      <td>-0.740102</td>
      <td>0.634821</td>
      <td>foo</td>
      <td>True</td>
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
      <td>-0.713772</td>
      <td>1.282062</td>
      <td>0.411180</td>
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
      <td>-0.479925</td>
      <td>-1.075762</td>
      <td>0.323755</td>
      <td>foo</td>
      <td>False</td>
      <td>4</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1.012460</td>
      <td>-1.565270</td>
      <td>-0.165303</td>
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
      <td>-0.414984</td>
      <td>-0.532008</td>
      <td>-1.931477</td>
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




    <vtreat.numeric_outcome_treatment at 0x1a1be4df28>




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
      <th>four_foo</th>
      <th>four__NA_</th>
      <th>four_blog</th>
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
      <td>1.892427</td>
      <td>-0.740102</td>
      <td>0.634821</td>
      <td>1.0</td>
      <td>0.009018</td>
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
      <td>0.259241</td>
      <td>-0.526216</td>
      <td>-0.145405</td>
      <td>0.4</td>
      <td>0.000000</td>
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
      <td>-0.713772</td>
      <td>1.282062</td>
      <td>0.411180</td>
      <td>0.0</td>
      <td>0.009018</td>
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
      <td>0.259241</td>
      <td>-0.526216</td>
      <td>-0.145405</td>
      <td>0.4</td>
      <td>-0.009719</td>
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
      <td>-0.479925</td>
      <td>-1.075762</td>
      <td>0.323755</td>
      <td>0.0</td>
      <td>0.009018</td>
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
      <td>1.012460</td>
      <td>-1.565270</td>
      <td>-0.165303</td>
      <td>1.0</td>
      <td>0.009018</td>
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
      <td>0.259241</td>
      <td>-0.526216</td>
      <td>-0.145405</td>
      <td>0.4</td>
      <td>0.000000</td>
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
      <td>-0.414984</td>
      <td>-0.532008</td>
      <td>-1.931477</td>
      <td>0.0</td>
      <td>0.009018</td>
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
      <th>four_foo</th>
      <th>four__NA_</th>
      <th>four_blog</th>
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
      <td>1.892427</td>
      <td>-0.740102</td>
      <td>0.634821</td>
      <td>1.0</td>
      <td>0.195219</td>
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
      <td>0.259241</td>
      <td>-0.526216</td>
      <td>-0.145405</td>
      <td>0.4</td>
      <td>0.000000</td>
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
      <td>-0.713772</td>
      <td>1.282062</td>
      <td>0.411180</td>
      <td>0.0</td>
      <td>0.043956</td>
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
      <td>0.259241</td>
      <td>-0.526216</td>
      <td>-0.145405</td>
      <td>0.4</td>
      <td>0.000000</td>
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
      <td>-0.479925</td>
      <td>-1.075762</td>
      <td>0.323755</td>
      <td>0.0</td>
      <td>0.295775</td>
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
      <td>1.012460</td>
      <td>-1.565270</td>
      <td>-0.165303</td>
      <td>1.0</td>
      <td>-0.001562</td>
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
      <td>0.259241</td>
      <td>-0.526216</td>
      <td>-0.145405</td>
      <td>0.4</td>
      <td>0.000000</td>
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
      <td>-0.414984</td>
      <td>-0.532008</td>
      <td>-1.931477</td>
      <td>0.0</td>
      <td>-0.491063</td>
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
      <th>four_foo</th>
      <th>four__NA_</th>
      <th>four_blog</th>
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
      <td>1.892427</td>
      <td>-0.740102</td>
      <td>0.634821</td>
      <td>1.0</td>
      <td>0.009018</td>
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
      <td>0.259241</td>
      <td>-0.526216</td>
      <td>-0.145405</td>
      <td>0.4</td>
      <td>0.000000</td>
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
      <td>-0.713772</td>
      <td>1.282062</td>
      <td>0.411180</td>
      <td>0.0</td>
      <td>0.009018</td>
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
      <td>0.259241</td>
      <td>-0.526216</td>
      <td>-0.145405</td>
      <td>0.4</td>
      <td>-0.009719</td>
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
      <td>-0.479925</td>
      <td>-1.075762</td>
      <td>0.323755</td>
      <td>0.0</td>
      <td>0.009018</td>
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
      <td>1.012460</td>
      <td>-1.565270</td>
      <td>-0.165303</td>
      <td>1.0</td>
      <td>0.009018</td>
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
      <td>0.259241</td>
      <td>-0.526216</td>
      <td>-0.145405</td>
      <td>0.4</td>
      <td>0.000000</td>
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
      <td>-0.414984</td>
      <td>-0.532008</td>
      <td>-1.931477</td>
      <td>0.0</td>
      <td>0.009018</td>
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
      <th>PearsonR</th>
      <th>significance</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>one_is_bad</td>
      <td>-0.056344</td>
      <td>0.894579</td>
    </tr>
    <tr>
      <th>1</th>
      <td>two_is_bad</td>
      <td>-0.056344</td>
      <td>0.894579</td>
    </tr>
    <tr>
      <th>2</th>
      <td>three_is_bad</td>
      <td>-0.056344</td>
      <td>0.894579</td>
    </tr>
    <tr>
      <th>3</th>
      <td>four_is_bad</td>
      <td>0.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>five_is_bad</td>
      <td>-0.056344</td>
      <td>0.894579</td>
    </tr>
    <tr>
      <th>5</th>
      <td>one</td>
      <td>-0.396942</td>
      <td>0.330218</td>
    </tr>
    <tr>
      <th>6</th>
      <td>two</td>
      <td>-0.271784</td>
      <td>0.514943</td>
    </tr>
    <tr>
      <th>7</th>
      <td>three</td>
      <td>-0.712851</td>
      <td>0.047176</td>
    </tr>
    <tr>
      <th>8</th>
      <td>five</td>
      <td>-0.309890</td>
      <td>0.455084</td>
    </tr>
    <tr>
      <th>9</th>
      <td>four_impact_code</td>
      <td>-0.590365</td>
      <td>0.123373</td>
    </tr>
    <tr>
      <th>10</th>
      <td>four_foo</td>
      <td>0.056344</td>
      <td>0.894579</td>
    </tr>
    <tr>
      <th>11</th>
      <td>four__NA_</td>
      <td>0.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>12</th>
      <td>four_blog</td>
      <td>-0.082479</td>
      <td>0.846053</td>
    </tr>
  </tbody>
</table>
</div>




```python

```


```python

```
