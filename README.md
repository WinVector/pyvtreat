

[This](https://github.com/WinVector/pyvtreat) is the Python version of the 
[`R` `vtreat`](http://winvector.github.io/vtreat/) package.

In each case: `vtreat` is an data.frame processor/conditioner that
prepares real-world data for predictive modeling in a statistically
sound manner.

For more detail please see here: [arXiv:1611.09477
stat.AP](https://arxiv.org/abs/1611.09477).

‘vtreat’ is supplied by [Win-Vector LLC](http://www.win-vector.com)
under a [BSD 3-clause license](LICENSE), without warranty. We are also developing
a [Python version of ‘vtreat’]().

![](https://github.com/WinVector/vtreat/raw/master/tools/vtreat.png)

(logo: Julie Mount, source: “The Harvest” by Boris Kustodiev 1914)

Some operational examples can be found [here](https://github.com/WinVector/pyvtreat/tree/master/Examples).

We are working on new documentation. But for now understand `vtreat` is used by instantiating one of the classes
`vtreat.numeric_outcome_treatment()`, `vtreat.binomial_outcome_treatment()`, or `vtreat.multinomial_outcome_treatment()`.
Each of these implements the [`sklearn.pipeline.Pipeline`](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html) interfaces
expecting a [Pandas Data.Frame](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.html) as input.  The `Pipeline.fit_transform()`
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

## Solution

Some `vreat` data treatments are “y-aware” (use distribution relations between
independent variables and the dependent variable).

The purpose of ‘vtreat’ library is to reliably prepare data for
supervised machine learning. We try to leave as much as possible to the
machine learning algorithms themselves, but cover most of the truly
necessary typically ignored precautions. The library is designed to
produce a ‘data.frame’ that is entirely numeric and takes common
precautions to guard against the following real world data issues:

  - Categorical variables with very many levels.
    
    We re-encode such variables as a family of indicator or dummy
    variables for common levels plus an additional [impact
    code](http://www.win-vector.com/blog/2012/07/modeling-trick-impact-coding-of-categorical-variables-with-many-levels/)
    (also called “effects coded”). This allows principled use (including
    smoothing) of huge categorical variables (like zip-codes) when
    building models. This is critical for some libraries (such as
    ‘randomForest’, which has hard limits on the number of allowed
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
    chose of the ‘scale’ parameter). The second column is a dummy or
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
worksheet](http://winvector.github.io/KDD2009/KDD2009RF.html) (though we
think for understanding it is *essential* to work all the steps by hand
as we did in the book). The idea is: ‘data.frame’s prepared with the
’vtreat’ library are somewhat safe to train on as some precaution has
been taken against all of the above issues. Also of interest are the
‘vtreat’ variable significances (help in initial variable pruning, a
necessity when there are a large number of columns) and
‘vtreat::prepare(scale=TRUE)’ which re-encodes all variables into
effect units making them suitable for y-aware dimension reduction
(variable clustering, or principal component analysis) and for geometry
sensitive machine learning techniques (k-means, knn, linear SVM, and
more). You may want to do more than the ‘vtreat’ library does (such as
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
  - [What is new in the vtreat
    library?](http://www.win-vector.com/blog/2015/05/what-is-new-in-the-vtreat-library/)
  - [How do you know if your data has
    signal?](http://www.win-vector.com/blog/2015/08/how-do-you-know-if-your-data-has-signal/)

Examples of current best practice using ‘vtreat’ (variable coding,
train, test split) can be found
[here](https://winvector.github.io/vtreat/articles/vtreatOverfit.html)
and [here](http://winvector.github.io/KDD2009/KDD2009RF.html).

We intend to add better Python documentation and a certification suite going forward.

## Installation

To install, from inside `R` please run:

```
!pip install https://github.com/WinVector/pyvtreat/raw/master/pkg/dist/vtreat-0.1.tar.gz
```
