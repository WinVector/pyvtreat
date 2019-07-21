

```python
import vtreat.util
```


```python
vtreat.util.k_way_cross_plan(10,4)
```




    [{'train': [0, 1, 3, 4, 6, 7, 8], 'test': [2, 5, 9]},
     {'train': [0, 2, 4, 5, 7, 8, 9], 'test': [1, 3, 6]},
     {'train': [0, 1, 2, 3, 5, 6, 7, 9], 'test': [4, 8]},
     {'train': [1, 2, 3, 4, 5, 6, 8, 9], 'test': [0, 7]}]




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
      <td>-0.685734</td>
      <td>-0.187367</td>
      <td>2.483612</td>
      <td>foo</td>
      <td>False</td>
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
      <td>-0.331823</td>
      <td>1.273405</td>
      <td>0.207794</td>
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
      <td>-1.763357</td>
      <td>-0.108552</td>
      <td>-0.525345</td>
      <td>foo</td>
      <td>False</td>
    </tr>
    <tr>
      <th>f</th>
      <td>-0.212728</td>
      <td>-0.488208</td>
      <td>-1.692696</td>
      <td>foo</td>
      <td>False</td>
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
      <td>0.868518</td>
      <td>0.717318</td>
      <td>-0.582037</td>
      <td>foo</td>
      <td>True</td>
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
      <td>-0.685734</td>
      <td>-0.187367</td>
      <td>2.483612</td>
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
      <td>-0.331823</td>
      <td>1.273405</td>
      <td>0.207794</td>
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
      <td>-1.763357</td>
      <td>-0.108552</td>
      <td>-0.525345</td>
      <td>foo</td>
      <td>False</td>
      <td>4</td>
    </tr>
    <tr>
      <th>5</th>
      <td>-0.212728</td>
      <td>-0.488208</td>
      <td>-1.692696</td>
      <td>foo</td>
      <td>False</td>
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
      <td>0.868518</td>
      <td>0.717318</td>
      <td>-0.582037</td>
      <td>foo</td>
      <td>True</td>
      <td>7</td>
    </tr>
  </tbody>
</table>
</div>




```python
df2.loc[3, "four"]
```




    'blog'




```python
plan.fit(df2, df2["y"])
```




    <vtreat.numeric_outcome_treatment at 0x11a582978>




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
      <td>-0.685734</td>
      <td>-0.187367</td>
      <td>2.483612</td>
      <td>0.0</td>
      <td>0.009018</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>-0.425025</td>
      <td>0.241319</td>
      <td>-0.021734</td>
      <td>0.2</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>-0.331823</td>
      <td>1.273405</td>
      <td>0.207794</td>
      <td>0.0</td>
      <td>0.009018</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>-0.425025</td>
      <td>0.241319</td>
      <td>-0.021734</td>
      <td>0.2</td>
      <td>-0.009719</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>-1.763357</td>
      <td>-0.108552</td>
      <td>-0.525345</td>
      <td>0.0</td>
      <td>0.009018</td>
    </tr>
    <tr>
      <th>5</th>
      <td>5</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>-0.212728</td>
      <td>-0.488208</td>
      <td>-1.692696</td>
      <td>0.0</td>
      <td>0.009018</td>
    </tr>
    <tr>
      <th>6</th>
      <td>6</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>-0.425025</td>
      <td>0.241319</td>
      <td>-0.021734</td>
      <td>0.2</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>7</th>
      <td>7</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.868518</td>
      <td>0.717318</td>
      <td>-0.582037</td>
      <td>1.0</td>
      <td>0.009018</td>
    </tr>
  </tbody>
</table>
</div>




```python

```
