"""
vtreat data preparation package: https://github.com/WinVector/pyvtreat
"""

# noinspection PyUnresolvedReferences
import pandas

# noinspection PyUnresolvedReferences
import numpy

from vtreat.vtreat_api import *

__docformat__ = "restructuredtext"
__version__ = "1.2.3"

__doc__ = """
This<https://github.com/WinVector/pyvtreat> is the Python version of the vtreat data preparation system
(also available as an R package<https://winvector.github.io/vtreat/>.

vtreat is a DataFrame processor/conditioner that prepares
real-world data for supervised machine learning or predictive modeling
in a statistically sound manner.

vtreat takes an input DataFrame
that has a specified column called "the outcome variable" (or "y")
that is the quantity to be predicted (and must not have missing
values).  Other input columns are possible explanatory variables
(typically numeric or categorical/string-valued, these columns may
have missing values) that the user later wants to use to predict "y".
In practice such an input DataFrame may not be immediately suitable
for machine learning procedures that often expect only numeric
explanatory variables, and may not tolerate missing values.

To solve this, vtreat builds a transformed DataFrame where all
explanatory variable columns have been transformed into a number of
numeric explanatory variable columns, without missing values.  The
vtreat implementation produces derived numeric columns that capture
most of the information relating the explanatory columns to the
specified "y" or dependent/outcome column through a number of numeric
transforms (indicator variables, impact codes, prevalence codes, and
more).  This transformed DataFrame is suitable for a wide range of
supervised learning methods from linear regression, through gradient
boosted machines.

The idea is: you can take a DataFrame of messy real world data and
easily, faithfully, reliably, and repeatably prepare it for machine
learning using documented methods using vtreat.  Incorporating
vtreat into your machine learning workflow lets you quickly work
with very diverse structured data.

Worked examples can be found `here`<https://github.com/WinVector/pyvtreat/tree/master/Examples>.

For more detail please see here: `arXiv:1611.09477
stat.AP`<https://arxiv.org/abs/1611.09477> (the documentation describes the R version,
however all of the examples can be found worked in Python 
`here`<https://github.com/WinVector/pyvtreat/tree/master/Examples/vtreat_paper1>).

vtreat is available
as a `Python/Pandas package`<https://github.com/WinVector/vtreat>,
and also as an `R package`<https://github.com/WinVector/vtreat>.
"""
