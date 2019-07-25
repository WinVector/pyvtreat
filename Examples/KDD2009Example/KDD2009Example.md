
This is an supervised classification example taken from the KDD 2009 cup.  A copy of the data and details can be found here: [https://github.com/WinVector/PDSwR2/tree/master/KDD2009](https://github.com/WinVector/PDSwR2/tree/master/KDD2009).  The problem was to predict account cancellation ("churn") from very messy data (column names not given, numeric and categorical variables, many missing values, some categorical variables with a large number of possible levels).  In this example we show how to quickly use `vtreat` to prepare the data for modeling.  `vtreat` takes in `Pandas` `DataFrame`s and returns both a treatment plan and a clean `Pandas` `DataFrame` ready for modeling.
# To install:
!pip install https://github.com/WinVector/pyvtreat/raw/master/pkg/dist/vtreat-0.1.tar.gz
!pip install https://github.com/WinVector/wvpy/raw/master/pkg/dist/wvpy-0.1.tar.gz
Load our packages/modules.


```python
import pandas
import xgboost
import vtreat
import numpy
import numpy.random
import wvpy.util
```

Read in explanitory variables.


```python


# data from https://github.com/WinVector/PDSwR2/tree/master/KDD2009
dir = "../../../PracticalDataScienceWithR2nd/PDSwR2/KDD2009/"
d = pandas.read_csv(dir + 'orange_small_train.data.gz', sep='\t', header=0)
vars = [c for c in d.columns]
d.shape
```




    (50000, 230)



Read in dependent variable we are trying to predict.


```python
churn = pandas.read_csv(dir + 'orange_small_train_churn.labels.txt', header=None)
churn.columns = ["churn"]
churn.shape
```




    (50000, 1)




```python
churn["churn"].value_counts()
```




    -1    46328
     1     3672
    Name: churn, dtype: int64



Arrange test/train split.


```python


n = d.shape[0]
is_train = numpy.random.uniform(size=n)<=0.9
is_test = numpy.logical_not(is_train)
```


```python
d_train = d.loc[is_train, :].copy()
churn_train = numpy.asarray(churn.loc[is_train, :]["churn"]==1)
d_test = d.loc[is_test, :].copy()
churn_test = numpy.asarray(churn.loc[is_test, :]["churn"]==1)
```

Take a look at the dependent variables.  They are a mess, many missing values.  Categorical variables that can not be directly used without some re-encoding.


```python
d_train.head()
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
      <th>Var1</th>
      <th>Var2</th>
      <th>Var3</th>
      <th>Var4</th>
      <th>Var5</th>
      <th>Var6</th>
      <th>Var7</th>
      <th>Var8</th>
      <th>Var9</th>
      <th>Var10</th>
      <th>...</th>
      <th>Var221</th>
      <th>Var222</th>
      <th>Var223</th>
      <th>Var224</th>
      <th>Var225</th>
      <th>Var226</th>
      <th>Var227</th>
      <th>Var228</th>
      <th>Var229</th>
      <th>Var230</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1526.0</td>
      <td>7.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>oslk</td>
      <td>fXVEsaq</td>
      <td>jySVZNlOJy</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>xb3V</td>
      <td>RAYp</td>
      <td>F2FyR07IdsN7I</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>5236.0</td>
      <td>7.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>Al6ZaUT</td>
      <td>NKv4yOc</td>
      <td>jySVZNlOJy</td>
      <td>NaN</td>
      <td>kG3k</td>
      <td>Qu4f</td>
      <td>02N6s8f</td>
      <td>ib5G6X1eUxUn6</td>
      <td>am7c</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>oslk</td>
      <td>CE7uk3u</td>
      <td>LM8l689qOp</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>FSa2</td>
      <td>RAYp</td>
      <td>F2FyR07IdsN7I</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1029.0</td>
      <td>7.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>oslk</td>
      <td>1J2cvxe</td>
      <td>LM8l689qOp</td>
      <td>NaN</td>
      <td>kG3k</td>
      <td>FSa2</td>
      <td>RAYp</td>
      <td>F2FyR07IdsN7I</td>
      <td>mj86</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>5</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>658.0</td>
      <td>7.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>zCkv</td>
      <td>QqVuch3</td>
      <td>LM8l689qOp</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Qcbd</td>
      <td>02N6s8f</td>
      <td>Zy3gnGM</td>
      <td>am7c</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 230 columns</p>
</div>




```python
d_train.shape
```




    (44965, 230)



Try building a model directly off this data (this will fail).


```python
fitter = xgboost.XGBClassifier(n_estimators=10, max_depth=3, objective='binary:logistic')
try:
    fitter.fit(d_train, churn_train)
except Exception as ex:
    print(ex)
```

    DataFrame.dtypes for data must be int, float or bool.
                    Did not expect the data types in fields Var191, Var192, Var193, Var194, Var195, Var196, Var197, Var198, Var199, Var200, Var201, Var202, Var203, Var204, Var205, Var206, Var207, Var208, Var210, Var211, Var212, Var213, Var214, Var215, Var216, Var217, Var218, Var219, Var220, Var221, Var222, Var223, Var224, Var225, Var226, Var227, Var228, Var229


Let's quickly prepare a data frame with none of these issues.

Build our treatment plan, this has the `sklearn.pipeline.Pipeline` interfaces.


```python
plan = vtreat.BinomialOutcomeTreatment(outcome_target=True)
```

Use `.fit_transform()` to get a special copy of the treated training data that has cross-validated mitigations againsst nested model bias. We call this a "cross frame." `.fit_transform()` is deliberately a different `DataFrame` than what would be returned by `.fit().transform()` (the `.fit().transform()` would damage the modeling effort due nested model bias, the `.fit_transform()` "cross frame" uses cross-validation techniques similar to "stacking" to mitigate these issues).


```python
cross_frame = plan.fit_transform(d_train, churn_train)
```

Take a look at the new data.  This frame is guaranteed to be all numeric with no missing values.


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
      <th>Var1_is_bad</th>
      <th>Var2_is_bad</th>
      <th>Var3_is_bad</th>
      <th>Var4_is_bad</th>
      <th>Var5_is_bad</th>
      <th>Var6_is_bad</th>
      <th>Var7_is_bad</th>
      <th>Var9_is_bad</th>
      <th>Var10_is_bad</th>
      <th>Var11_is_bad</th>
      <th>...</th>
      <th>Var228_logit_code</th>
      <th>Var228_prevalence_code</th>
      <th>Var228_lev_F2FyR07IdsN7I</th>
      <th>Var228_lev_55YFVY9</th>
      <th>Var228_lev_ib5G6X1eUxUn6</th>
      <th>Var229_logit_code</th>
      <th>Var229_prevalence_code</th>
      <th>Var229_lev__NA_</th>
      <th>Var229_lev_am7c</th>
      <th>Var229_lev_mj86</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>...</td>
      <td>0.006403</td>
      <td>0.653820</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>-0.017896</td>
      <td>0.569198</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>...</td>
      <td>0.041186</td>
      <td>0.053664</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0.010103</td>
      <td>0.233359</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>...</td>
      <td>-0.079843</td>
      <td>0.653820</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0.008275</td>
      <td>0.569198</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>...</td>
      <td>0.022627</td>
      <td>0.653820</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>-0.005632</td>
      <td>0.195797</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>...</td>
      <td>0.015539</td>
      <td>0.018414</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>-0.019172</td>
      <td>0.233359</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 518 columns</p>
</div>




```python
cross_frame.shape
```




    (44965, 518)



Pick a recommended subset of the new derived variables.


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
      <td>Var1_is_bad</td>
      <td>missing_indicator</td>
      <td>False</td>
      <td>0.003385</td>
      <td>4.729142e-01</td>
      <td>193.0</td>
      <td>False</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Var2_is_bad</td>
      <td>missing_indicator</td>
      <td>False</td>
      <td>0.019442</td>
      <td>3.740540e-05</td>
      <td>193.0</td>
      <td>True</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Var3_is_bad</td>
      <td>missing_indicator</td>
      <td>False</td>
      <td>0.019410</td>
      <td>3.850970e-05</td>
      <td>193.0</td>
      <td>True</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Var4_is_bad</td>
      <td>missing_indicator</td>
      <td>False</td>
      <td>0.021402</td>
      <td>5.661738e-06</td>
      <td>193.0</td>
      <td>True</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Var5_is_bad</td>
      <td>missing_indicator</td>
      <td>False</td>
      <td>0.015501</td>
      <td>1.012393e-03</td>
      <td>193.0</td>
      <td>True</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Var6_is_bad</td>
      <td>missing_indicator</td>
      <td>False</td>
      <td>-0.032383</td>
      <td>6.493446e-12</td>
      <td>193.0</td>
      <td>True</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Var7_is_bad</td>
      <td>missing_indicator</td>
      <td>False</td>
      <td>-0.027788</td>
      <td>3.781064e-09</td>
      <td>193.0</td>
      <td>True</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Var9_is_bad</td>
      <td>missing_indicator</td>
      <td>False</td>
      <td>0.003385</td>
      <td>4.729142e-01</td>
      <td>193.0</td>
      <td>False</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Var10_is_bad</td>
      <td>missing_indicator</td>
      <td>False</td>
      <td>0.015501</td>
      <td>1.012393e-03</td>
      <td>193.0</td>
      <td>True</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Var11_is_bad</td>
      <td>missing_indicator</td>
      <td>False</td>
      <td>0.019410</td>
      <td>3.850970e-05</td>
      <td>193.0</td>
      <td>True</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Var12_is_bad</td>
      <td>missing_indicator</td>
      <td>False</td>
      <td>0.001826</td>
      <td>6.985905e-01</td>
      <td>193.0</td>
      <td>False</td>
    </tr>
    <tr>
      <th>11</th>
      <td>Var13_is_bad</td>
      <td>missing_indicator</td>
      <td>False</td>
      <td>-0.027788</td>
      <td>3.781064e-09</td>
      <td>193.0</td>
      <td>True</td>
    </tr>
    <tr>
      <th>12</th>
      <td>Var14_is_bad</td>
      <td>missing_indicator</td>
      <td>False</td>
      <td>0.019410</td>
      <td>3.850970e-05</td>
      <td>193.0</td>
      <td>True</td>
    </tr>
    <tr>
      <th>13</th>
      <td>Var16_is_bad</td>
      <td>missing_indicator</td>
      <td>False</td>
      <td>0.015501</td>
      <td>1.012393e-03</td>
      <td>193.0</td>
      <td>True</td>
    </tr>
    <tr>
      <th>14</th>
      <td>Var17_is_bad</td>
      <td>missing_indicator</td>
      <td>False</td>
      <td>0.021402</td>
      <td>5.661738e-06</td>
      <td>193.0</td>
      <td>True</td>
    </tr>
    <tr>
      <th>15</th>
      <td>Var18_is_bad</td>
      <td>missing_indicator</td>
      <td>False</td>
      <td>0.021402</td>
      <td>5.661738e-06</td>
      <td>193.0</td>
      <td>True</td>
    </tr>
    <tr>
      <th>16</th>
      <td>Var19_is_bad</td>
      <td>missing_indicator</td>
      <td>False</td>
      <td>0.021402</td>
      <td>5.661738e-06</td>
      <td>193.0</td>
      <td>True</td>
    </tr>
    <tr>
      <th>17</th>
      <td>Var21_is_bad</td>
      <td>missing_indicator</td>
      <td>False</td>
      <td>-0.032383</td>
      <td>6.493446e-12</td>
      <td>193.0</td>
      <td>True</td>
    </tr>
    <tr>
      <th>18</th>
      <td>Var22_is_bad</td>
      <td>missing_indicator</td>
      <td>False</td>
      <td>-0.032652</td>
      <td>4.343949e-12</td>
      <td>193.0</td>
      <td>True</td>
    </tr>
    <tr>
      <th>19</th>
      <td>Var23_is_bad</td>
      <td>missing_indicator</td>
      <td>False</td>
      <td>0.015501</td>
      <td>1.012393e-03</td>
      <td>193.0</td>
      <td>True</td>
    </tr>
    <tr>
      <th>20</th>
      <td>Var24_is_bad</td>
      <td>missing_indicator</td>
      <td>False</td>
      <td>-0.008777</td>
      <td>6.271690e-02</td>
      <td>193.0</td>
      <td>False</td>
    </tr>
    <tr>
      <th>21</th>
      <td>Var25_is_bad</td>
      <td>missing_indicator</td>
      <td>False</td>
      <td>-0.032652</td>
      <td>4.343949e-12</td>
      <td>193.0</td>
      <td>True</td>
    </tr>
    <tr>
      <th>22</th>
      <td>Var26_is_bad</td>
      <td>missing_indicator</td>
      <td>False</td>
      <td>0.015501</td>
      <td>1.012393e-03</td>
      <td>193.0</td>
      <td>True</td>
    </tr>
    <tr>
      <th>23</th>
      <td>Var27_is_bad</td>
      <td>missing_indicator</td>
      <td>False</td>
      <td>0.015501</td>
      <td>1.012393e-03</td>
      <td>193.0</td>
      <td>True</td>
    </tr>
    <tr>
      <th>24</th>
      <td>Var28_is_bad</td>
      <td>missing_indicator</td>
      <td>False</td>
      <td>-0.032687</td>
      <td>4.121249e-12</td>
      <td>193.0</td>
      <td>True</td>
    </tr>
    <tr>
      <th>25</th>
      <td>Var29_is_bad</td>
      <td>missing_indicator</td>
      <td>False</td>
      <td>0.003385</td>
      <td>4.729142e-01</td>
      <td>193.0</td>
      <td>False</td>
    </tr>
    <tr>
      <th>26</th>
      <td>Var30_is_bad</td>
      <td>missing_indicator</td>
      <td>False</td>
      <td>0.003385</td>
      <td>4.729142e-01</td>
      <td>193.0</td>
      <td>False</td>
    </tr>
    <tr>
      <th>27</th>
      <td>Var33_is_bad</td>
      <td>missing_indicator</td>
      <td>False</td>
      <td>0.011216</td>
      <td>1.739026e-02</td>
      <td>193.0</td>
      <td>False</td>
    </tr>
    <tr>
      <th>28</th>
      <td>Var34_is_bad</td>
      <td>missing_indicator</td>
      <td>False</td>
      <td>0.019442</td>
      <td>3.740540e-05</td>
      <td>193.0</td>
      <td>True</td>
    </tr>
    <tr>
      <th>29</th>
      <td>Var35_is_bad</td>
      <td>missing_indicator</td>
      <td>False</td>
      <td>-0.032652</td>
      <td>4.343949e-12</td>
      <td>193.0</td>
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
      <td>...</td>
    </tr>
    <tr>
      <th>488</th>
      <td>Var224_lev__NA_</td>
      <td>indicator</td>
      <td>False</td>
      <td>0.009378</td>
      <td>4.675376e-02</td>
      <td>77.0</td>
      <td>False</td>
    </tr>
    <tr>
      <th>489</th>
      <td>Var225_logit_code</td>
      <td>logit_code</td>
      <td>True</td>
      <td>-0.003979</td>
      <td>3.987913e-01</td>
      <td>38.0</td>
      <td>False</td>
    </tr>
    <tr>
      <th>490</th>
      <td>Var225_prevalence_code</td>
      <td>prevalance</td>
      <td>False</td>
      <td>0.050386</td>
      <td>1.123177e-26</td>
      <td>38.0</td>
      <td>True</td>
    </tr>
    <tr>
      <th>491</th>
      <td>Var225_lev__NA_</td>
      <td>indicator</td>
      <td>False</td>
      <td>0.051948</td>
      <td>2.971946e-28</td>
      <td>77.0</td>
      <td>True</td>
    </tr>
    <tr>
      <th>492</th>
      <td>Var225_lev_ELof</td>
      <td>indicator</td>
      <td>False</td>
      <td>-0.043013</td>
      <td>7.186227e-20</td>
      <td>77.0</td>
      <td>True</td>
    </tr>
    <tr>
      <th>493</th>
      <td>Var225_lev_kG3k</td>
      <td>indicator</td>
      <td>False</td>
      <td>-0.013334</td>
      <td>4.689932e-03</td>
      <td>77.0</td>
      <td>True</td>
    </tr>
    <tr>
      <th>494</th>
      <td>Var226_logit_code</td>
      <td>logit_code</td>
      <td>True</td>
      <td>0.007817</td>
      <td>9.739843e-02</td>
      <td>38.0</td>
      <td>False</td>
    </tr>
    <tr>
      <th>495</th>
      <td>Var226_prevalence_code</td>
      <td>prevalance</td>
      <td>False</td>
      <td>0.022791</td>
      <td>1.342776e-06</td>
      <td>38.0</td>
      <td>True</td>
    </tr>
    <tr>
      <th>496</th>
      <td>Var226_lev_FSa2</td>
      <td>indicator</td>
      <td>False</td>
      <td>0.033970</td>
      <td>5.795839e-13</td>
      <td>77.0</td>
      <td>True</td>
    </tr>
    <tr>
      <th>497</th>
      <td>Var226_lev_Qu4f</td>
      <td>indicator</td>
      <td>False</td>
      <td>-0.010371</td>
      <td>2.787348e-02</td>
      <td>77.0</td>
      <td>False</td>
    </tr>
    <tr>
      <th>498</th>
      <td>Var226_lev_WqMG</td>
      <td>indicator</td>
      <td>False</td>
      <td>-0.003786</td>
      <td>4.220729e-01</td>
      <td>77.0</td>
      <td>False</td>
    </tr>
    <tr>
      <th>499</th>
      <td>Var226_lev_szEZ</td>
      <td>indicator</td>
      <td>False</td>
      <td>-0.018318</td>
      <td>1.025274e-04</td>
      <td>77.0</td>
      <td>True</td>
    </tr>
    <tr>
      <th>500</th>
      <td>Var226_lev_7P5s</td>
      <td>indicator</td>
      <td>False</td>
      <td>-0.021580</td>
      <td>4.731554e-06</td>
      <td>77.0</td>
      <td>True</td>
    </tr>
    <tr>
      <th>501</th>
      <td>Var226_lev_fKCe</td>
      <td>indicator</td>
      <td>False</td>
      <td>0.000981</td>
      <td>8.351445e-01</td>
      <td>77.0</td>
      <td>False</td>
    </tr>
    <tr>
      <th>502</th>
      <td>Var226_lev_Aoh3</td>
      <td>indicator</td>
      <td>False</td>
      <td>-0.006156</td>
      <td>1.918027e-01</td>
      <td>77.0</td>
      <td>False</td>
    </tr>
    <tr>
      <th>503</th>
      <td>Var227_logit_code</td>
      <td>logit_code</td>
      <td>True</td>
      <td>0.009193</td>
      <td>5.125406e-02</td>
      <td>38.0</td>
      <td>False</td>
    </tr>
    <tr>
      <th>504</th>
      <td>Var227_prevalence_code</td>
      <td>prevalance</td>
      <td>False</td>
      <td>0.048016</td>
      <td>2.259444e-24</td>
      <td>38.0</td>
      <td>True</td>
    </tr>
    <tr>
      <th>505</th>
      <td>Var227_lev_RAYp</td>
      <td>indicator</td>
      <td>False</td>
      <td>0.049265</td>
      <td>1.424677e-25</td>
      <td>77.0</td>
      <td>True</td>
    </tr>
    <tr>
      <th>506</th>
      <td>Var227_lev_ZI9m</td>
      <td>indicator</td>
      <td>False</td>
      <td>-0.044310</td>
      <td>5.438132e-21</td>
      <td>77.0</td>
      <td>True</td>
    </tr>
    <tr>
      <th>507</th>
      <td>Var227_lev_6fzt</td>
      <td>indicator</td>
      <td>False</td>
      <td>-0.004029</td>
      <td>3.929762e-01</td>
      <td>77.0</td>
      <td>False</td>
    </tr>
    <tr>
      <th>508</th>
      <td>Var228_logit_code</td>
      <td>logit_code</td>
      <td>True</td>
      <td>0.003193</td>
      <td>4.984331e-01</td>
      <td>38.0</td>
      <td>False</td>
    </tr>
    <tr>
      <th>509</th>
      <td>Var228_prevalence_code</td>
      <td>prevalance</td>
      <td>False</td>
      <td>0.059180</td>
      <td>3.512280e-36</td>
      <td>38.0</td>
      <td>True</td>
    </tr>
    <tr>
      <th>510</th>
      <td>Var228_lev_F2FyR07IdsN7I</td>
      <td>indicator</td>
      <td>False</td>
      <td>0.059825</td>
      <td>6.153401e-37</td>
      <td>77.0</td>
      <td>True</td>
    </tr>
    <tr>
      <th>511</th>
      <td>Var228_lev_55YFVY9</td>
      <td>indicator</td>
      <td>False</td>
      <td>-0.027128</td>
      <td>8.746792e-09</td>
      <td>77.0</td>
      <td>True</td>
    </tr>
    <tr>
      <th>512</th>
      <td>Var228_lev_ib5G6X1eUxUn6</td>
      <td>indicator</td>
      <td>False</td>
      <td>-0.033013</td>
      <td>2.522182e-12</td>
      <td>77.0</td>
      <td>True</td>
    </tr>
    <tr>
      <th>513</th>
      <td>Var229_logit_code</td>
      <td>logit_code</td>
      <td>True</td>
      <td>-0.013827</td>
      <td>3.366715e-03</td>
      <td>38.0</td>
      <td>False</td>
    </tr>
    <tr>
      <th>514</th>
      <td>Var229_prevalence_code</td>
      <td>prevalance</td>
      <td>False</td>
      <td>0.061386</td>
      <td>8.388562e-39</td>
      <td>38.0</td>
      <td>True</td>
    </tr>
    <tr>
      <th>515</th>
      <td>Var229_lev__NA_</td>
      <td>indicator</td>
      <td>False</td>
      <td>0.061517</td>
      <td>5.826863e-39</td>
      <td>77.0</td>
      <td>True</td>
    </tr>
    <tr>
      <th>516</th>
      <td>Var229_lev_am7c</td>
      <td>indicator</td>
      <td>False</td>
      <td>-0.036813</td>
      <td>5.781241e-15</td>
      <td>77.0</td>
      <td>True</td>
    </tr>
    <tr>
      <th>517</th>
      <td>Var229_lev_mj86</td>
      <td>indicator</td>
      <td>False</td>
      <td>-0.037487</td>
      <td>1.841089e-15</td>
      <td>77.0</td>
      <td>True</td>
    </tr>
  </tbody>
</table>
<p>518 rows × 7 columns</p>
</div>




```python
model_vars = numpy.asarray(plan.score_frame_["variable"][plan.score_frame_["recommended"]])
len(model_vars)
```




    229



Fit the model


```python
fd = xgboost.DMatrix(data=cross_frame.loc[:, model_vars], label=churn_train)
x_parameters = {"max_depth":3, "objective":'binary:logistic'}
cv = xgboost.cv(x_parameters, fd, num_boost_round=100, verbose_eval=False)
```


```python
cv.head()
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
      <th>train-error-mean</th>
      <th>train-error-std</th>
      <th>test-error-mean</th>
      <th>test-error-std</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.072723</td>
      <td>0.000813</td>
      <td>0.073457</td>
      <td>0.001824</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.072712</td>
      <td>0.000871</td>
      <td>0.073012</td>
      <td>0.001685</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.072868</td>
      <td>0.000888</td>
      <td>0.072901</td>
      <td>0.001698</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.072901</td>
      <td>0.000849</td>
      <td>0.072901</td>
      <td>0.001698</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.072912</td>
      <td>0.000848</td>
      <td>0.072901</td>
      <td>0.001698</td>
    </tr>
  </tbody>
</table>
</div>




```python
best = cv.loc[cv["test-error-mean"]<= min(cv["test-error-mean"] + 1.0e-9), :]
best


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
      <th>train-error-mean</th>
      <th>train-error-std</th>
      <th>test-error-mean</th>
      <th>test-error-std</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>25</th>
      <td>0.071856</td>
      <td>0.000892</td>
      <td>0.072434</td>
      <td>0.001722</td>
    </tr>
    <tr>
      <th>77</th>
      <td>0.070143</td>
      <td>0.000772</td>
      <td>0.072434</td>
      <td>0.002102</td>
    </tr>
  </tbody>
</table>
</div>




```python
ntree = best.index.values[0]
ntree
```




    25




```python
fitter = xgboost.XGBClassifier(n_estimators=ntree, max_depth=3, objective='binary:logistic')
fitter
```




    XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
                  colsample_bytree=1, gamma=0, learning_rate=0.1, max_delta_step=0,
                  max_depth=3, min_child_weight=1, missing=None, n_estimators=25,
                  n_jobs=1, nthread=None, objective='binary:logistic',
                  random_state=0, reg_alpha=0, reg_lambda=1, scale_pos_weight=1,
                  seed=None, silent=True, subsample=1)




```python
model = fitter.fit(cross_frame.loc[:, model_vars], churn_train)



```

Apply the data transform to our held-out data.


```python
test_processed = plan.transform(d_test)
```

Plot the quality of the model score on the held-out data.  This AUC is not great, but in the ballpark of the original contest winners.


```python

pf = pandas.DataFrame({"churn":churn_test})
preds = model.predict_proba(test_processed.loc[:, model_vars])


```


```python
pf["pred"] = preds[:, 1]
```


```python
wvpy.util.plot_roc(pf["pred"], pf["churn"])
```


![png](output_40_0.png)





    0.7267540362494079



Notice we dealt with many problem columns at once, and in a statistically sound manner. More on the `vtreat` package for Python can be found here: [https://github.com/WinVector/pyvtreat](https://github.com/WinVector/pyvtreat).  Details on the `R` version can be found here: [https://github.com/WinVector/vtreat](https://github.com/WinVector/vtreat).

Compare to [R solution](https://github.com/WinVector/PDSwR2/blob/master/KDD2009/KDD2009vtreat.md).
