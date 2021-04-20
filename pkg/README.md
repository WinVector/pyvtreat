

[This](https://github.com/WinVector/pyvtreat) is the Python version of the `vtreat` data preparation system
(also available as an [`R` package](http://winvector.github.io/vtreat/)).

`vtreat` is a `DataFrame` processor/conditioner that prepares
real-world data for supervised machine learning or predictive modeling
in a statistically sound manner.

# Installing

Install `vtreat` with either:

  * `pip install vtreat`

# Video Introduction


[Our PyData LA 2019 talk](https://youtu.be/qMCQFjEV90k) on `vtreat` is a good video introduction
to what problems `vtreat` can be used to solve.  The slides can be found [here](https://github.com/WinVector/Examples/blob/master/PyDataLA2019/vtreat_pydata2019.pdf).

# Details

`vtreat` takes an input `DataFrame`
that has a specified column called "the outcome variable" (or "y")
that is the quantity to be predicted (and must not have missing
values).  Other input columns are possible explanatory variables
(typically numeric or categorical/string-valued, these columns may
have missing values) that the user later wants to use to predict "y".
In practice such an input `DataFrame` may not be immediately suitable
for machine learning procedures that often expect only numeric
explanatory variables, and may not tolerate missing values.

To solve this, `vtreat` builds a transformed `DataFrame` where all
explanatory variable columns have been transformed into a number of
numeric explanatory variable columns, without missing values.  The
`vtreat` implementation produces derived numeric columns that capture
most of the information relating the explanatory columns to the
specified "y" or dependent/outcome column through a number of numeric
transforms (indicator variables, impact codes, prevalence codes, and
more).  This transformed `DataFrame` is suitable for a wide range of
supervised learning methods from linear regression, through gradient
boosted machines.

The idea is: you can take a `DataFrame` of messy real world data and
easily, faithfully, reliably, and repeatably prepare it for machine
learning using documented methods using `vtreat`.  Incorporating
`vtreat` into your machine learning workflow lets you quickly work
with very diverse structured data.

To get started with `vtreat` please check out our documentation:

  * [Getting started using `vtreat` for classification](https://github.com/WinVector/pyvtreat/blob/master/Examples/Classification/Classification.md).
  * [Getting started using `vtreat` for regression](https://github.com/WinVector/pyvtreat/blob/master/Examples/Regression/Regression.md).
  * [Getting started using `vtreat` for multi-category classification](https://github.com/WinVector/pyvtreat/blob/master/Examples/Multinomial/MultinomialExample.md).
  * [Getting started using `vtreat` for unsupervised tasks](https://github.com/WinVector/pyvtreat/blob/master/Examples/Unsupervised/Unsupervised.md).
  * [The `vtreat` Score Frame](https://github.com/WinVector/pyvtreat/blob/master/Examples/ScoreFrame/ScoreFrame.md) (a table mapping new derived variables to original columns).
  * [The original `vtreat` paper](https://arxiv.org/abs/1611.09477) this note describes the methodology and theory. (The article describes the `R` version, however all of the examples can be found worked in `Python` [here](https://github.com/WinVector/pyvtreat/tree/master/Examples/vtreat_paper1)).

Some `vtreat` common capabilities are documented here:

  * **Score Frame** [score_frame_](https://github.com/WinVector/pyvtreat/blob/master/Examples/ScoreFrame/ScoreFrame.md), using the `score_frame_` information.
  * **Cross Validation** [Customized Cross Plans](https://github.com/WinVector/pyvtreat/blob/master/Examples/CustomizedCrossPlan/CustomizedCrossPlan.md), controlling the cross validation plan.

`vtreat` is available as a [`Python`/`Pandas` package](https://github.com/WinVector/vtreat), and also as an [`R` package](https://github.com/WinVector/vtreat).


![](https://github.com/WinVector/vtreat/raw/master/tools/vtreat.png)

(logo: Julie Mount, source: “The Harvest” by Boris Kustodiev 1914)

`vtreat` is used by instantiating one of the classes
`vtreat.NumericOutcomeTreatment`, `vtreat.BinomialOutcomeTreatment`, `vtreat.MultinomialOutcomeTreatment`, or `vtreat.UnsupervisedTreatment`.
Each of these implements the [`sklearn.pipeline.Pipeline`](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html) interfaces
expecting a [Pandas DataFrame](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.html) as input. The `vtreat` steps are intended to
be a "one step fix" that works well with [`sklearn.preprocessing`](https://scikit-learn.org/stable/modules/preprocessing.html) stages.

The `vtreat` `Pipeline.fit_transform()`
method implements the powerful [cross-frame](https://cran.r-project.org/web/packages/vtreat/vignettes/vtreatCrossFrames.html) ideas (allowing the same data to be used for `vtreat` fitting and for later model construction, while
mitigating nested model bias issues).

## Background

Even with modern machine learning techniques (random forests, support
vector machines, neural nets, gradient boosted trees, and so on) or
standard statistical methods (regression, generalized regression,
generalized additive models) there are *common* data issues that can
cause modeling to fail. vtreat deals with a number of these in a
principled and automated fashion.

In particular `vtreat` emphasizes a concept called “y-aware
pre-processing” and implements:

  - Treatment of missing values through safe replacement plus an indicator
    column (a simple but very powerful method when combined with
    downstream machine learning algorithms).
  - Treatment of novel levels (new values of categorical variable seen
    during test or application, but not seen during training) through
    sub-models (or impact/effects coding of pooled rare events).
  - Explicit coding of categorical variable levels as new indicator
    variables (with optional suppression of non-significant indicators).
  - Treatment of categorical variables with very large numbers of levels
    through sub-models (again [impact/effects
    coding](http://www.win-vector.com/blog/2012/07/modeling-trick-impact-coding-of-categorical-variables-with-many-levels/)).
  - Correct treatment of nested models or sub-models through data split / cross-frame methods
    (please see
    [here](https://winvector.github.io/vtreat/articles/vtreatOverfit.html))
    or through the generation of “cross validated” data frames (see
    [here](https://winvector.github.io/vtreat/articles/vtreatCrossFrames.html));
    these are issues similar to what is required to build statistically
    efficient stacked models or super-learners).

The idea is: even with a sophisticated machine learning algorithm there
are *many* ways messy real world data can defeat the modeling process,
and vtreat helps with at least ten of them. We emphasize: these problems
are already in your data, you simply build better and more reliable
models if you attempt to mitigate them. Automated processing is no
substitute for actually looking at the data, but vtreat supplies
efficient, reliable, documented, and tested implementations of many of
the commonly needed transforms.

To help explain the methods we have prepared some documentation:

  - The [vtreat package
    overall](https://winvector.github.io/vtreat/index.html).
  - [Preparing data for analysis using R
    white-paper](http://winvector.github.io/DataPrep/EN-CNTNT-Whitepaper-Data-Prep-Using-R.pdf)
  - The [types of new
    variables](https://winvector.github.io/vtreat/articles/vtreatVariableTypes.html)
    introduced by vtreat processing (including how to limit down to
    domain appropriate variable types).
  - Statistically sound treatment of the nested modeling issue
    introduced by any sort of pre-processing (such as vtreat itself):
    [nested over-fit
    issues](https://winvector.github.io/vtreat/articles/vtreatOverfit.html)
    and a general [cross-frame
    solution](https://winvector.github.io/vtreat/articles/vtreatCrossFrames.html).
  - [Principled ways to pick significance based pruning
    levels](https://winvector.github.io/vtreat/articles/vtreatSignificance.html).

## Example


This is an supervised classification example taken from the KDD 2009 cup.  A copy of the data and details can be found here: [https://github.com/WinVector/PDSwR2/tree/master/KDD2009](https://github.com/WinVector/PDSwR2/tree/master/KDD2009).  The problem was to predict account cancellation ("churn") from very messy data (column names not given, numeric and categorical variables, many missing values, some categorical variables with a large number of possible levels).  In this example we show how to quickly use `vtreat` to prepare the data for modeling.  `vtreat` takes in `Pandas` `DataFrame`s and returns both a treatment plan and a clean `Pandas` `DataFrame` ready for modeling.
# to install
!pip install vtreat
!pip install wvpy
Load our packages/modules.


```python
import pandas
import xgboost
import vtreat
import vtreat.cross_plan
import numpy.random
import wvpy.util
import scipy.sparse
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
numpy.random.seed(855885)
n = d.shape[0]
# https://github.com/WinVector/pyvtreat/blob/master/Examples/CustomizedCrossPlan/CustomizedCrossPlan.md
split1 = vtreat.cross_plan.KWayCrossPlanYStratified().split_plan(n_rows=n, k_folds=10, y=churn.iloc[:, 0])
train_idx = set(split1[0]['train'])
is_train = [i in train_idx for i in range(n)]
is_test = numpy.logical_not(is_train)
```

(The reported performance runs of this example were sensitive to the prevalance of the churn variable in the test set, we are cutting down on this source of evaluation variarance by using the stratified split.)


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
      <th>1</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>525.0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>oslk</td>
      <td>2Kb5FSF</td>
      <td>LM8l689qOp</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>fKCe</td>
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
  </tbody>
</table>
<p>5 rows × 230 columns</p>
</div>




```python
d_train.shape
```




    (45000, 230)



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

We start by building our treatment plan, this has the `sklearn.pipeline.Pipeline` interfaces.


```python
plan = vtreat.BinomialOutcomeTreatment(outcome_target=True)
```

Use `.fit_transform()` to get a special copy of the treated training data that has cross-validated mitigations againsst nested model bias. We call this a "cross frame." `.fit_transform()` is deliberately a different `DataFrame` than what would be returned by `.fit().transform()` (the `.fit().transform()` would damage the modeling effort due nested model bias, the `.fit_transform()` "cross frame" uses cross-validation techniques similar to "stacking" to mitigate these issues).


```python
cross_frame = plan.fit_transform(d_train, churn_train)
```

Take a look at the new data.  This frame is guaranteed to be all numeric with no missing values, with the rows in the same order as the training data.


```python
cross_frame.head()
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Var2_is_bad</th>
      <th>Var3_is_bad</th>
      <th>Var4_is_bad</th>
      <th>Var5_is_bad</th>
      <th>Var6_is_bad</th>
      <th>Var7_is_bad</th>
      <th>Var10_is_bad</th>
      <th>Var11_is_bad</th>
      <th>Var13_is_bad</th>
      <th>Var14_is_bad</th>
      <th>...</th>
      <th>Var227_lev_RAYp</th>
      <th>Var227_lev_ZI9m</th>
      <th>Var228_logit_code</th>
      <th>Var228_prevalence_code</th>
      <th>Var228_lev_F2FyR07IdsN7I</th>
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
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>...</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.151682</td>
      <td>0.653733</td>
      <td>1.0</td>
      <td>0.172744</td>
      <td>0.567422</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>...</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.146119</td>
      <td>0.653733</td>
      <td>1.0</td>
      <td>0.175707</td>
      <td>0.567422</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>-0.629820</td>
      <td>0.053956</td>
      <td>0.0</td>
      <td>-0.263504</td>
      <td>0.234400</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>...</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.145871</td>
      <td>0.653733</td>
      <td>1.0</td>
      <td>0.159486</td>
      <td>0.567422</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>...</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.147432</td>
      <td>0.653733</td>
      <td>1.0</td>
      <td>-0.286852</td>
      <td>0.196600</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 216 columns</p>
</div>




```python
cross_frame.shape
```




    (45000, 216)



Pick a recommended subset of the new derived variables.


```python
plan.score_frame_.head()
```




<div>

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
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Var1_is_bad</td>
      <td>Var1</td>
      <td>missing_indicator</td>
      <td>False</td>
      <td>True</td>
      <td>0.003283</td>
      <td>0.486212</td>
      <td>193.0</td>
      <td>0.001036</td>
      <td>False</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Var2_is_bad</td>
      <td>Var2</td>
      <td>missing_indicator</td>
      <td>False</td>
      <td>True</td>
      <td>0.019270</td>
      <td>0.000044</td>
      <td>193.0</td>
      <td>0.001036</td>
      <td>True</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Var3_is_bad</td>
      <td>Var3</td>
      <td>missing_indicator</td>
      <td>False</td>
      <td>True</td>
      <td>0.019238</td>
      <td>0.000045</td>
      <td>193.0</td>
      <td>0.001036</td>
      <td>True</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Var4_is_bad</td>
      <td>Var4</td>
      <td>missing_indicator</td>
      <td>False</td>
      <td>True</td>
      <td>0.018744</td>
      <td>0.000070</td>
      <td>193.0</td>
      <td>0.001036</td>
      <td>True</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Var5_is_bad</td>
      <td>Var5</td>
      <td>missing_indicator</td>
      <td>False</td>
      <td>True</td>
      <td>0.017575</td>
      <td>0.000193</td>
      <td>193.0</td>
      <td>0.001036</td>
      <td>True</td>
    </tr>
  </tbody>
</table>
</div>




```python
model_vars = numpy.asarray(plan.score_frame_["variable"][plan.score_frame_["recommended"]])
len(model_vars)
```




    216



Fit the model


```python
cross_frame.dtypes
```




    Var2_is_bad                            float64
    Var3_is_bad                            float64
    Var4_is_bad                            float64
    Var5_is_bad                            float64
    Var6_is_bad                            float64
                                      ...         
    Var229_logit_code                      float64
    Var229_prevalence_code                 float64
    Var229_lev__NA_           Sparse[float64, 0.0]
    Var229_lev_am7c           Sparse[float64, 0.0]
    Var229_lev_mj86           Sparse[float64, 0.0]
    Length: 216, dtype: object




```python
# fails due to sparse columns
# can also work around this by setting the vtreat parameter 'sparse_indicators' to False
try:
    cross_sparse = xgboost.DMatrix(data=cross_frame.loc[:, model_vars], label=churn_train)
except Exception as ex:
    print(ex)
```

    DataFrame.dtypes for data must be int, float or bool.
                    Did not expect the data types in fields Var193_lev_RO12, Var193_lev_2Knk1KF, Var194_lev__NA_, Var194_lev_SEuy, Var195_lev_taul, Var200_lev__NA_, Var201_lev__NA_, Var201_lev_smXZ, Var205_lev_VpdQ, Var206_lev_IYzP, Var206_lev_zm5i, Var206_lev__NA_, Var207_lev_me75fM6ugJ, Var207_lev_7M47J5GA0pTYIFxg5uy, Var210_lev_uKAI, Var211_lev_L84s, Var211_lev_Mtgm, Var212_lev_NhsEn4L, Var212_lev_XfqtO3UdzaXh_, Var213_lev__NA_, Var214_lev__NA_, Var218_lev_cJvF, Var218_lev_UYBR, Var221_lev_oslk, Var221_lev_zCkv, Var225_lev__NA_, Var225_lev_ELof, Var225_lev_kG3k, Var226_lev_FSa2, Var227_lev_RAYp, Var227_lev_ZI9m, Var228_lev_F2FyR07IdsN7I, Var229_lev__NA_, Var229_lev_am7c, Var229_lev_mj86



```python
# also fails
try:
    cross_sparse = scipy.sparse.csc_matrix(cross_frame[model_vars])
except Exception as ex:
    print(ex)
```

    no supported conversion for types: (dtype('O'),)



```python
# works
cross_sparse = scipy.sparse.hstack([scipy.sparse.csc_matrix(cross_frame[[vi]]) for vi in model_vars])
```


```python
# https://xgboost.readthedocs.io/en/latest/python/python_intro.html
fd = xgboost.DMatrix(
    data=cross_sparse, 
    label=churn_train)
```


```python
x_parameters = {"max_depth":3, "objective":'binary:logistic'}
cv = xgboost.cv(x_parameters, fd, num_boost_round=100, verbose_eval=False)
```


```python
cv.head()
```




<div>

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
      <td>0.073378</td>
      <td>0.000322</td>
      <td>0.073733</td>
      <td>0.000669</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.073411</td>
      <td>0.000257</td>
      <td>0.073511</td>
      <td>0.000529</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.073433</td>
      <td>0.000268</td>
      <td>0.073578</td>
      <td>0.000514</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.073444</td>
      <td>0.000283</td>
      <td>0.073533</td>
      <td>0.000525</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.073444</td>
      <td>0.000283</td>
      <td>0.073533</td>
      <td>0.000525</td>
    </tr>
  </tbody>
</table>
</div>




```python
best = cv.loc[cv["test-error-mean"]<= min(cv["test-error-mean"] + 1.0e-9), :]
best


```




<div>

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
      <th>21</th>
      <td>0.072756</td>
      <td>0.000177</td>
      <td>0.073267</td>
      <td>0.000327</td>
    </tr>
  </tbody>
</table>
</div>




```python
ntree = best.index.values[0]
ntree
```




    21




```python
fitter = xgboost.XGBClassifier(n_estimators=ntree, max_depth=3, objective='binary:logistic')
fitter
```




    XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
                  colsample_bynode=1, colsample_bytree=1, gamma=0,
                  learning_rate=0.1, max_delta_step=0, max_depth=3,
                  min_child_weight=1, missing=None, n_estimators=21, n_jobs=1,
                  nthread=None, objective='binary:logistic', random_state=0,
                  reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,
                  silent=None, subsample=1, verbosity=1)




```python
model = fitter.fit(cross_sparse, churn_train)
```

Apply the data transform to our held-out data.


```python
test_processed = plan.transform(d_test)
```

Plot the quality of the model on training data (a biased measure of performance).


```python
pf_train = pandas.DataFrame({"churn":churn_train})
pf_train["pred"] = model.predict_proba(cross_sparse)[:, 1]
wvpy.util.plot_roc(pf_train["pred"], pf_train["churn"], title="Model on Train")
```


![png](https://github.com/WinVector/pyvtreat/raw/master/Examples/KDD2009Example/output_44_0.png)





    0.7424056263753072



Plot the quality of the model score on the held-out data.  This AUC is not great, but in the ballpark of the original contest winners.


```python
test_sparse = scipy.sparse.hstack([scipy.sparse.csc_matrix(test_processed[[vi]]) for vi in model_vars])
pf = pandas.DataFrame({"churn":churn_test})
pf["pred"] = model.predict_proba(test_sparse)[:, 1]
wvpy.util.plot_roc(pf["pred"], pf["churn"], title="Model on Test")
```


![png](https://github.com/WinVector/pyvtreat/raw/master/Examples/KDD2009Example/output_46_0.png)





    0.7328696191869485



Notice we dealt with many problem columns at once, and in a statistically sound manner. More on the `vtreat` package for Python can be found here: [https://github.com/WinVector/pyvtreat](https://github.com/WinVector/pyvtreat).  Details on the `R` version can be found here: [https://github.com/WinVector/vtreat](https://github.com/WinVector/vtreat).

We can compare this to the [R solution (link)](https://github.com/WinVector/PDSwR2/blob/master/KDD2009/KDD2009vtreat.md).

We can compare the above cross-frame solution to a naive "design transform and model on the same data set" solution as we show below.  Note we turn off `filter_to_recommended` as this is computed using cross-frame techniques (and hence is a non-naive estimate).


```python
plan_naive = vtreat.BinomialOutcomeTreatment(
    outcome_target=True,              
    params=vtreat.vtreat_parameters({'filter_to_recommended':False}))
plan_naive.fit(d_train, churn_train)
naive_frame = plan_naive.transform(d_train)
```


```python
naive_sparse = scipy.sparse.hstack([scipy.sparse.csc_matrix(naive_frame[[vi]]) for vi in model_vars])
```


```python
fd_naive = xgboost.DMatrix(data=naive_sparse, label=churn_train)
x_parameters = {"max_depth":3, "objective":'binary:logistic'}
cvn = xgboost.cv(x_parameters, fd_naive, num_boost_round=100, verbose_eval=False)
```


```python
bestn = cvn.loc[cvn["test-error-mean"]<= min(cvn["test-error-mean"] + 1.0e-9), :]
bestn
```




<div>

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
      <th>94</th>
      <td>0.0485</td>
      <td>0.000438</td>
      <td>0.058622</td>
      <td>0.000545</td>
    </tr>
  </tbody>
</table>
</div>




```python
ntreen = bestn.index.values[0]
ntreen
```




    94




```python
fittern = xgboost.XGBClassifier(n_estimators=ntreen, max_depth=3, objective='binary:logistic')
fittern
```




    XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
                  colsample_bynode=1, colsample_bytree=1, gamma=0,
                  learning_rate=0.1, max_delta_step=0, max_depth=3,
                  min_child_weight=1, missing=None, n_estimators=94, n_jobs=1,
                  nthread=None, objective='binary:logistic', random_state=0,
                  reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,
                  silent=None, subsample=1, verbosity=1)




```python
modeln = fittern.fit(naive_sparse, churn_train)
```


```python
test_processedn = plan_naive.transform(d_test)
test_processedn = scipy.sparse.hstack([scipy.sparse.csc_matrix(test_processedn[[vi]]) for vi in model_vars])
```


```python
pfn_train = pandas.DataFrame({"churn":churn_train})
pfn_train["pred_naive"] = modeln.predict_proba(naive_sparse)[:, 1]
wvpy.util.plot_roc(pfn_train["pred_naive"], pfn_train["churn"], title="Overfit Model on Train")
```


![png](https://github.com/WinVector/pyvtreat/raw/master/Examples/KDD2009Example/output_58_0.png)





    0.9492686875296688




```python
pfn = pandas.DataFrame({"churn":churn_test})
pfn["pred_naive"] = modeln.predict_proba(test_processedn)[:, 1]
wvpy.util.plot_roc(pfn["pred_naive"], pfn["churn"], title="Overfit Model on Test")
```


![png](https://github.com/WinVector/pyvtreat/raw/master/Examples/KDD2009Example/output_59_0.png)





    0.5960012412998182



Note the naive test performance is worse, despite its far better training performance.  This is over-fit due to the nested model bias of using the same data to build the treatment plan and model without any cross-frame mitigations.




## Solution Details

Some `vreat` data treatments are “y-aware” (use distribution relations between
independent variables and the dependent variable).

The purpose of `vtreat` library is to reliably prepare data for
supervised machine learning. We try to leave as much as possible to the
machine learning algorithms themselves, but cover most of the truly
necessary typically ignored precautions. The library is designed to
produce a `DataFrame` that is entirely numeric and takes common
precautions to guard against the following real world data issues:

  - Categorical variables with very many levels.
    
    We re-encode such variables as a family of indicator or dummy
    variables for common levels plus an additional [impact
    code](http://www.win-vector.com/blog/2012/07/modeling-trick-impact-coding-of-categorical-variables-with-many-levels/)
    (also called “effects coded”). This allows principled use (including
    smoothing) of huge categorical variables (like zip-codes) when
    building models. This is critical for some libraries (such as
    `randomForest`, which has hard limits on the number of allowed
    levels).

  - Rare categorical levels.
    
    Levels that do not occur often during training tend not to have
    reliable effect estimates and contribute to over-fit.

  - Novel categorical levels.
    
    A common problem in deploying a classifier to production is: new
    levels (levels not seen during training) encountered during model
    application. We deal with this by encoding categorical variables in
    a possibly redundant manner: reserving a dummy variable for all
    levels (not the more common all but a reference level scheme). This
    is in fact the correct representation for regularized modeling
    techniques and lets us code novel levels as all dummies
    simultaneously zero (which is a reasonable thing to try). This
    encoding while limited is cheaper than the fully Bayesian solution
    of computing a weighted sum over previously seen levels during model
    application.

  - Missing/invalid values NA, NaN, +-Inf.
    
    Variables with these issues are re-coded as two columns. The first
    column is clean copy of the variable (with missing/invalid values
    replaced with either zero or the grand mean, depending on the user
    chose of the `scale` parameter). The second column is a dummy or
    indicator that marks if the replacement has been performed. This is
    simpler than imputation of missing values, and allows the downstream
    model to attempt to use missingness as a useful signal (which it
    often is in industrial data).

The above are all awful things that often lurk in real world data.
Automating mitigation steps ensures they are easy enough that you actually
perform them and leaves the analyst time to look for additional data
issues. For example this allowed us to essentially automate a number of
the steps taught in chapters 4 and 6 of [*Practical Data Science with R*
(Zumel, Mount; Manning 2014)](http://practicaldatascience.com/) into a
[very short
worksheet](https://github.com/WinVector/pyvtreat/blob/master/Examples/KDD2009Example/KDD2009Example.md) (though we
think for understanding it is *essential* to work all the steps by hand
as we did in the book).  The 2nd edition of *Practical Data Science with R* covers
using `vtreat` in `R` in chapter 8 "Advanced Data Preparation."

The idea is: `DataFrame`s prepared with the
`vtreat` library are somewhat safe to train on as some precaution has
been taken against all of the above issues. Also of interest are the
`vtreat` variable significances (help in initial variable pruning, a
necessity when there are a large number of columns) and
`vtreat::prepare(scale=TRUE)` which re-encodes all variables into
effect units making them suitable for y-aware dimension reduction
(variable clustering, or principal component analysis) and for geometry
sensitive machine learning techniques (k-means, knn, linear SVM, and
more). You may want to do more than the `vtreat` library does (such as
Bayesian imputation, variable clustering, and more) but you certainly do
not want to do less.

## References

Some of our related articles (which should make clear some of our
motivations, and design decisions):

  - [The `vtreat` technical paper](https://arxiv.org/abs/1611.09477).
  - [Modeling trick: impact coding of categorical variables with many
    levels](http://www.win-vector.com/blog/2012/07/modeling-trick-impact-coding-of-categorical-variables-with-many-levels/)
  - [A bit more on impact
    coding](http://www.win-vector.com/blog/2012/08/a-bit-more-on-impact-coding/)
  - [vtreat: designing a package for variable
    treatment](http://www.win-vector.com/blog/2014/08/vtreat-designing-a-package-for-variable-treatment/)
  - [A comment on preparing data for
    classifiers](http://www.win-vector.com/blog/2014/12/a-comment-on-preparing-data-for-classifiers/)
  - [Nina Zumel presenting on
    vtreat](http://www.slideshare.net/ChesterChen/vtreat)


A directory of worked examples can be found [here](https://github.com/WinVector/pyvtreat/tree/master/Examples).

We intend to add better Python documentation and a certification suite going forward.

## Installation

To install, please run:

```python
# To install:
pip install vtreat
```

Some notes on controlling `vtreat` cross-validation can be found [here](https://github.com/WinVector/pyvtreat/blob/master/Examples/CustomizedCrossPlan/CustomizedCrossPlan.md).

## Note on data types.

`.fit_transform()` expects the first argument to be a `pandas.DataFrame` with trivial row-indexing and scalar column names, (i.e. `.reset_index(inplace=True, drop=True)`) and the second to be a vector-like object with a `len()` equal to the number of rows of the first argument. We are working on supporting column types other than string and numeric at this time.

