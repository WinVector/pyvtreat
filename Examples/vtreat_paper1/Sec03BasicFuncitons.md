
Translating the R sequences from https://arxiv.org/abs/1611.09477 into Python vtreat https://github.com/WinVector/pyvtreat


```python

```

Note: for these small examples it is not-determined if the impact/logit codes show up, as they can often have an unlucky cross-validation split.


```python

```

R original
## ----VOpsSimpleDataFrameD------------------------------------------------
d <- data.frame(
   x=c('a', 'a', 'b', 'b', NA), 
   z=c(0, 1, 2, NA, 4), 
   y=c(TRUE, TRUE, FALSE, TRUE, TRUE), 
   stringsAsFactors = FALSE)
d$yN <- as.numeric(d$y)
print(d)


## ----VTypesN1, results='hide'--------------------------------------------
library("vtreat")
treatments <- designTreatmentsN(d, c('x', 'z'), 'yN')

## ----VTypesN1s-----------------------------------------------------------
scols <- c('varName', 'sig', 'extraModelDegrees', 'origName', 'code')
print(treatments$scoreFrame[, scols])


## ----VTypesN1p-----------------------------------------------------------
dTreated <- prepare(treatments, d, pruneSig=NULL)
print(dTreated)
Python translation


```python
## ----VOpsSimpleDataFrameD------------------------------------------------
```


```python
import pandas
import numpy
import vtreat # https://github.com/WinVector/pyvtreat

d = pandas.DataFrame({
   'x':['a', 'a', 'b', 'b', numpy.NaN], 
   'z':[0, 1, 2, numpy.NaN, 4], 
   'y':[True, True, False, True, True]
    })
d['yN'] = numpy.asarray(d["y"], dtype=float)
d
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
      <th>x</th>
      <th>z</th>
      <th>y</th>
      <th>yN</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>a</td>
      <td>0.0</td>
      <td>True</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>a</td>
      <td>1.0</td>
      <td>True</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>b</td>
      <td>2.0</td>
      <td>False</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>b</td>
      <td>NaN</td>
      <td>True</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>NaN</td>
      <td>4.0</td>
      <td>True</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
d.dtypes
```




    x      object
    z     float64
    y        bool
    yN    float64
    dtype: object




```python
## ----VTypesN1, results='hide'--------------------------------------------
```


```python
treatments = vtreat.NumericOutcomeTreatment(outcome_name='yN',
                                            params = vtreat.vtreat_parameters({
                                               'filter_to_recommended':False
                                            }))
treatments.fit(d[['x', 'z']], d['yN'])
```




    <vtreat.NumericOutcomeTreatment at 0x1a19f37748>




```python
## ----VTypesN1s-----------------------------------------------------------
```


```python
treatments.score_frame_
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
      <th>0</th>
      <td>x_is_bad</td>
      <td>x</td>
      <td>missing_indicator</td>
      <td>False</td>
      <td>True</td>
      <td>0.250000</td>
      <td>0.685038</td>
      <td>2.0</td>
      <td>False</td>
    </tr>
    <tr>
      <th>1</th>
      <td>z_is_bad</td>
      <td>z</td>
      <td>missing_indicator</td>
      <td>False</td>
      <td>True</td>
      <td>0.250000</td>
      <td>0.685038</td>
      <td>2.0</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2</th>
      <td>z</td>
      <td>z</td>
      <td>clean_copy</td>
      <td>False</td>
      <td>True</td>
      <td>-0.094491</td>
      <td>0.879869</td>
      <td>1.0</td>
      <td>False</td>
    </tr>
    <tr>
      <th>3</th>
      <td>x_impact_code</td>
      <td>x</td>
      <td>impact_code</td>
      <td>True</td>
      <td>True</td>
      <td>0.468462</td>
      <td>0.426133</td>
      <td>1.0</td>
      <td>False</td>
    </tr>
    <tr>
      <th>4</th>
      <td>x_deviance_code</td>
      <td>x</td>
      <td>deviance_code</td>
      <td>True</td>
      <td>True</td>
      <td>-0.508002</td>
      <td>0.382203</td>
      <td>1.0</td>
      <td>False</td>
    </tr>
    <tr>
      <th>5</th>
      <td>x_prevalence_code</td>
      <td>x</td>
      <td>prevalence_code</td>
      <td>False</td>
      <td>True</td>
      <td>-0.250000</td>
      <td>0.685038</td>
      <td>1.0</td>
      <td>False</td>
    </tr>
    <tr>
      <th>6</th>
      <td>x_lev_a</td>
      <td>x</td>
      <td>indicator_code</td>
      <td>False</td>
      <td>True</td>
      <td>0.408248</td>
      <td>0.495025</td>
      <td>3.0</td>
      <td>False</td>
    </tr>
    <tr>
      <th>7</th>
      <td>x_lev_b</td>
      <td>x</td>
      <td>indicator_code</td>
      <td>False</td>
      <td>True</td>
      <td>-0.612372</td>
      <td>0.272228</td>
      <td>3.0</td>
      <td>False</td>
    </tr>
    <tr>
      <th>8</th>
      <td>x_lev__NA_</td>
      <td>x</td>
      <td>indicator_code</td>
      <td>False</td>
      <td>True</td>
      <td>0.250000</td>
      <td>0.685038</td>
      <td>3.0</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
</div>




```python
## ----VTypesN1p-----------------------------------------------------------
```


```python
dTreated = treatments.transform(d)
dTreated
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
      <th>yN</th>
      <th>x_is_bad</th>
      <th>z_is_bad</th>
      <th>z</th>
      <th>x_impact_code</th>
      <th>x_deviance_code</th>
      <th>x_prevalence_code</th>
      <th>x_lev_a</th>
      <th>x_lev_b</th>
      <th>x_lev__NA_</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>1.976562e-01</td>
      <td>0.031623</td>
      <td>0.2</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.00</td>
      <td>1.976562e-01</td>
      <td>0.031623</td>
      <td>0.2</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2.00</td>
      <td>-4.322323e-02</td>
      <td>0.707814</td>
      <td>0.2</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.75</td>
      <td>-4.322323e-02</td>
      <td>0.707814</td>
      <td>0.2</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>4.00</td>
      <td>-1.110223e-16</td>
      <td>0.500999</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python

```

R original
## ----VTypesC1, results='hide'--------------------------------------------
treatments <- designTreatmentsC(d, c('x', 'z'), 'y', TRUE)

## ----VTypesC1s-----------------------------------------------------------
print(treatments$scoreFrame[, scols])


## ----VTypesC1p-----------------------------------------------------------
dTreated <- prepare(treatments, d, pruneSig=NULL)
print(dTreated)
Python translation


```python
## ----VTypesC1, results='hide'--------------------------------------------
```


```python
treatments = vtreat.BinomialOutcomeTreatment(outcome_name='y',
                                             outcome_target=True,
                                             params = vtreat.vtreat_parameters({
                                               'filter_to_recommended':False
                                             }))
treatments.fit(d[['x', 'z']], d['y'])
```




    <vtreat.BinomialOutcomeTreatment at 0x1a19f8cef0>




```python
## ----VTypesC1s-----------------------------------------------------------
```


```python
treatments.score_frame_
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
      <th>0</th>
      <td>x_is_bad</td>
      <td>x</td>
      <td>missing_indicator</td>
      <td>False</td>
      <td>True</td>
      <td>0.250000</td>
      <td>0.685038</td>
      <td>2.0</td>
      <td>False</td>
    </tr>
    <tr>
      <th>1</th>
      <td>z_is_bad</td>
      <td>z</td>
      <td>missing_indicator</td>
      <td>False</td>
      <td>True</td>
      <td>0.250000</td>
      <td>0.685038</td>
      <td>2.0</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2</th>
      <td>z</td>
      <td>z</td>
      <td>clean_copy</td>
      <td>False</td>
      <td>True</td>
      <td>-0.094491</td>
      <td>0.879869</td>
      <td>1.0</td>
      <td>False</td>
    </tr>
    <tr>
      <th>3</th>
      <td>x_logit_code</td>
      <td>x</td>
      <td>logit_code</td>
      <td>True</td>
      <td>True</td>
      <td>0.475449</td>
      <td>0.418291</td>
      <td>1.0</td>
      <td>False</td>
    </tr>
    <tr>
      <th>4</th>
      <td>x_prevalence_code</td>
      <td>x</td>
      <td>prevalence_code</td>
      <td>False</td>
      <td>True</td>
      <td>-0.250000</td>
      <td>0.685038</td>
      <td>1.0</td>
      <td>False</td>
    </tr>
    <tr>
      <th>5</th>
      <td>x_lev_a</td>
      <td>x</td>
      <td>indicator_code</td>
      <td>False</td>
      <td>True</td>
      <td>0.408248</td>
      <td>0.495025</td>
      <td>3.0</td>
      <td>False</td>
    </tr>
    <tr>
      <th>6</th>
      <td>x_lev_b</td>
      <td>x</td>
      <td>indicator_code</td>
      <td>False</td>
      <td>True</td>
      <td>-0.612372</td>
      <td>0.272228</td>
      <td>3.0</td>
      <td>False</td>
    </tr>
    <tr>
      <th>7</th>
      <td>x_lev__NA_</td>
      <td>x</td>
      <td>indicator_code</td>
      <td>False</td>
      <td>True</td>
      <td>0.250000</td>
      <td>0.685038</td>
      <td>3.0</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
</div>




```python
## ----VTypesC1p-----------------------------------------------------------
```


```python
dTreated = treatments.transform(d)
dTreated
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
      <th>x_is_bad</th>
      <th>z_is_bad</th>
      <th>z</th>
      <th>x_logit_code</th>
      <th>x_prevalence_code</th>
      <th>x_lev_a</th>
      <th>x_lev_b</th>
      <th>x_lev__NA_</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>True</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>2.207971e-01</td>
      <td>0.2</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>True</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.00</td>
      <td>2.207971e-01</td>
      <td>0.2</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>False</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2.00</td>
      <td>-5.554341e-02</td>
      <td>0.2</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>True</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.75</td>
      <td>-5.554341e-02</td>
      <td>0.2</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>True</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>4.00</td>
      <td>-1.387779e-16</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python

```

R original
## ----VTypesZ1, results='hide'--------------------------------------------
treatments <- designTreatmentsZ(d, c('x', 'z'))

## ----VTypesZ1s-----------------------------------------------------------
print(treatments$scoreFrame[, scols])


## ----VTypesZ1p-----------------------------------------------------------
dTreated <- prepare(treatments, d, pruneSig=NULL)
print(dTreated)
Python translation


```python
## ----VTypesZ1, results='hide'--------------------------------------------
```


```python
treatments = vtreat.UnsupervisedTreatment()
treatments.fit(d[['x', 'z']])
```




    <vtreat.UnsupervisedTreatment at 0x1a19fd71d0>




```python
## ----VTypesZ1p-----------------------------------------------------------
```


```python
dTreated = treatments.transform(d)
dTreated
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
      <th>x_is_bad</th>
      <th>z_is_bad</th>
      <th>z</th>
      <th>x_prevalence_code</th>
      <th>x_lev_a</th>
      <th>x_lev_b</th>
      <th>x_lev__NA_</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.2</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.00</td>
      <td>0.2</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>2.00</td>
      <td>0.2</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.75</td>
      <td>0.2</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1.0</td>
      <td>0.0</td>
      <td>4.00</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python

```

R original
## ----VTypesCFN1, results='hide'------------------------------------------
cfe <- mkCrossFrameNExperiment(d, c('x', 'z'), 'yN')
treatments <- cfe$treatments
dTreated <- cfe$crossFrame
Python translation


```python
treatments = vtreat.NumericOutcomeTreatment(outcome_name='yN',
                                            params = vtreat.vtreat_parameters({
                                               'filter_to_recommended':False
                                            }))
treatments.fit_transform(d[['x', 'z']], d['yN'])
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
      <th>x_is_bad</th>
      <th>z_is_bad</th>
      <th>z</th>
      <th>x_impact_code</th>
      <th>x_deviance_code</th>
      <th>x_prevalence_code</th>
      <th>x_lev_a</th>
      <th>x_lev_b</th>
      <th>x_lev__NA_</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>1.976562e-01</td>
      <td>0.031623</td>
      <td>0.2</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.00</td>
      <td>1.976562e-01</td>
      <td>0.031623</td>
      <td>0.2</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>2.00</td>
      <td>-4.322323e-02</td>
      <td>0.707814</td>
      <td>0.2</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.75</td>
      <td>-4.322323e-02</td>
      <td>0.707814</td>
      <td>0.2</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1.0</td>
      <td>0.0</td>
      <td>4.00</td>
      <td>-1.110223e-16</td>
      <td>0.500999</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
treatments.score_frame_
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
      <th>0</th>
      <td>x_is_bad</td>
      <td>x</td>
      <td>missing_indicator</td>
      <td>False</td>
      <td>True</td>
      <td>0.250000</td>
      <td>0.685038</td>
      <td>2.0</td>
      <td>False</td>
    </tr>
    <tr>
      <th>1</th>
      <td>z_is_bad</td>
      <td>z</td>
      <td>missing_indicator</td>
      <td>False</td>
      <td>True</td>
      <td>0.250000</td>
      <td>0.685038</td>
      <td>2.0</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2</th>
      <td>z</td>
      <td>z</td>
      <td>clean_copy</td>
      <td>False</td>
      <td>True</td>
      <td>-0.094491</td>
      <td>0.879869</td>
      <td>1.0</td>
      <td>False</td>
    </tr>
    <tr>
      <th>3</th>
      <td>x_impact_code</td>
      <td>x</td>
      <td>impact_code</td>
      <td>True</td>
      <td>True</td>
      <td>0.468462</td>
      <td>0.426133</td>
      <td>1.0</td>
      <td>False</td>
    </tr>
    <tr>
      <th>4</th>
      <td>x_deviance_code</td>
      <td>x</td>
      <td>deviance_code</td>
      <td>True</td>
      <td>True</td>
      <td>-0.508002</td>
      <td>0.382203</td>
      <td>1.0</td>
      <td>False</td>
    </tr>
    <tr>
      <th>5</th>
      <td>x_prevalence_code</td>
      <td>x</td>
      <td>prevalence_code</td>
      <td>False</td>
      <td>True</td>
      <td>-0.250000</td>
      <td>0.685038</td>
      <td>1.0</td>
      <td>False</td>
    </tr>
    <tr>
      <th>6</th>
      <td>x_lev_a</td>
      <td>x</td>
      <td>indicator_code</td>
      <td>False</td>
      <td>True</td>
      <td>0.408248</td>
      <td>0.495025</td>
      <td>3.0</td>
      <td>False</td>
    </tr>
    <tr>
      <th>7</th>
      <td>x_lev_b</td>
      <td>x</td>
      <td>indicator_code</td>
      <td>False</td>
      <td>True</td>
      <td>-0.612372</td>
      <td>0.272228</td>
      <td>3.0</td>
      <td>False</td>
    </tr>
    <tr>
      <th>8</th>
      <td>x_lev__NA_</td>
      <td>x</td>
      <td>indicator_code</td>
      <td>False</td>
      <td>True</td>
      <td>0.250000</td>
      <td>0.685038</td>
      <td>3.0</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
</div>




```python

```

R original
## ----VTypesCFN2, results='hide'------------------------------------------
cfe <- mkCrossFrameCExperiment(d, c('x', 'z'), 'y', TRUE)
treatments <- cfe$treatments
dTreated <- cfe$crossFrame
Python translation


```python
## ----VTypesCFN2, results='hide'------------------------------------------
```


```python
treatments = vtreat.BinomialOutcomeTreatment(outcome_name='y',
                                             outcome_target=True,
                                             params = vtreat.vtreat_parameters({
                                               'filter_to_recommended':False
                                             }))
treatments.fit_transform(d[['x', 'z']], d['y'])
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
      <th>x_is_bad</th>
      <th>z_is_bad</th>
      <th>z</th>
      <th>x_logit_code</th>
      <th>x_prevalence_code</th>
      <th>x_lev_a</th>
      <th>x_lev_b</th>
      <th>x_lev__NA_</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>2.207971e-01</td>
      <td>0.2</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.00</td>
      <td>2.207971e-01</td>
      <td>0.2</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>2.00</td>
      <td>-5.554341e-02</td>
      <td>0.2</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.75</td>
      <td>-5.554341e-02</td>
      <td>0.2</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1.0</td>
      <td>0.0</td>
      <td>4.00</td>
      <td>-1.387779e-16</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
treatments.score_frame_
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
      <th>0</th>
      <td>x_is_bad</td>
      <td>x</td>
      <td>missing_indicator</td>
      <td>False</td>
      <td>True</td>
      <td>0.250000</td>
      <td>0.685038</td>
      <td>2.0</td>
      <td>False</td>
    </tr>
    <tr>
      <th>1</th>
      <td>z_is_bad</td>
      <td>z</td>
      <td>missing_indicator</td>
      <td>False</td>
      <td>True</td>
      <td>0.250000</td>
      <td>0.685038</td>
      <td>2.0</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2</th>
      <td>z</td>
      <td>z</td>
      <td>clean_copy</td>
      <td>False</td>
      <td>True</td>
      <td>-0.094491</td>
      <td>0.879869</td>
      <td>1.0</td>
      <td>False</td>
    </tr>
    <tr>
      <th>3</th>
      <td>x_logit_code</td>
      <td>x</td>
      <td>logit_code</td>
      <td>True</td>
      <td>True</td>
      <td>0.475449</td>
      <td>0.418291</td>
      <td>1.0</td>
      <td>False</td>
    </tr>
    <tr>
      <th>4</th>
      <td>x_prevalence_code</td>
      <td>x</td>
      <td>prevalence_code</td>
      <td>False</td>
      <td>True</td>
      <td>-0.250000</td>
      <td>0.685038</td>
      <td>1.0</td>
      <td>False</td>
    </tr>
    <tr>
      <th>5</th>
      <td>x_lev_a</td>
      <td>x</td>
      <td>indicator_code</td>
      <td>False</td>
      <td>True</td>
      <td>0.408248</td>
      <td>0.495025</td>
      <td>3.0</td>
      <td>False</td>
    </tr>
    <tr>
      <th>6</th>
      <td>x_lev_b</td>
      <td>x</td>
      <td>indicator_code</td>
      <td>False</td>
      <td>True</td>
      <td>-0.612372</td>
      <td>0.272228</td>
      <td>3.0</td>
      <td>False</td>
    </tr>
    <tr>
      <th>7</th>
      <td>x_lev__NA_</td>
      <td>x</td>
      <td>indicator_code</td>
      <td>False</td>
      <td>True</td>
      <td>0.250000</td>
      <td>0.685038</td>
      <td>3.0</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
</div>




```python

```

R original
## ----VTypesfsplitexample-------------------------------------------------
str(vtreat::oneWayHoldout(3, NULL, NULL, NULL))
Python translation


```python
import vtreat.util

vtreat.util.k_way_cross_plan(10, 3)
```




    [{'train': [0, 2, 4, 5, 6, 7], 'app': [1, 3, 8, 9]},
     {'train': [0, 1, 3, 5, 6, 8, 9], 'app': [2, 4, 7]},
     {'train': [1, 2, 3, 4, 7, 8, 9], 'app': [0, 5, 6]}]




```python
vtreat.util.k_way_cross_plan(2, 1)
```




    [{'train': [0, 1], 'app': [0, 1]}]




```python
vtreat.util.k_way_cross_plan(1, 0)
```




    [{'train': [0], 'app': [0]}]




```python
vtreat.util.k_way_cross_plan(0, 0)
```




    [{'train': [], 'app': []}]




```python

```


```python

```

R original
## ----VTypesParellel, results='hide'--------------------------------------
ncore <- 2
parallelCluster <- parallel::makeCluster(ncore)
cfe <- mkCrossFrameNExperiment(d, c('x', 'z'), 'yN', 
   parallelCluster=parallelCluster)
parallel::stopCluster(parallelCluster)
Python translation


```python
# We currently do not have a parallel option for the Python version of vtreat.
```
