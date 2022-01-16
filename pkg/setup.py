"""
vtreat is a pandas.DataFrame processor/conditioner that prepares real-world data for predictive modeling in a
statistically sound manner.

https://github.com/WinVector/pyvtreat
"""


# setup.py
import setuptools

DESCRIPTION = """vtreat is a pandas.DataFrame processor/conditioner that prepares real-world data for predictive modeling in a statistically sound manner. """

LONG_DESCRIPTION = """
[This](https://github.com/WinVector/pyvtreat) is the Python version of the `vtreat` data preparation system
(also available as an [`R` package](https://winvector.github.io/vtreat/)).

`vtreat` is a `DataFrame` processor/conditioner that prepares
real-world data for supervised machine learning or predictive modeling
in a statistically sound manner.

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

Worked examples can be found [here](https://github.com/WinVector/pyvtreat/tree/master/Examples).

For more detail please see here: [arXiv:1611.09477
stat.AP](https://arxiv.org/abs/1611.09477) (the documentation describes the `R` version,
however all of the examples can be found worked in `Python` [here](https://github.com/WinVector/pyvtreat/tree/master/Examples/vtreat_paper1)).

`vtreat` is available
as a [`Python`/`Pandas` package](https://github.com/WinVector/vtreat),
and also as an [`R` package](https://github.com/WinVector/vtreat).
"""


setuptools.setup(
    name='vtreat',
    version='1.2.3',
    author='John Mount, Nina Zumel',
    author_email='jmount@win-vector.com',
    url='https://github.com/WinVector/pyvtreat',
    packages=setuptools.find_packages(where='.', exclude=['tests', 'Examples']),
    install_requires=[
        'numpy',
        'pandas',
        'scipy',
        'sklearn',
        'data_algebra>=1.3.0',
    ],
    platforms=['any'],
    license='License :: OSI Approved :: BSD 3-clause License',
    python_requires=">=3.5.3",
    long_description_content_type='text/markdown',
    description=DESCRIPTION,
    classifiers=[
          'Intended Audience :: Science/Research',
          'Programming Language :: Python',
          'Programming Language :: Python :: 3',
          'Programming Language :: Python :: 3.5',
          'Programming Language :: Python :: 3.6',
          'Programming Language :: Python :: 3.7',
          'Programming Language :: Python :: 3.8',
          'Programming Language :: Python :: 3.9',
          'License :: OSI Approved :: BSD License',
    ],
    long_description=LONG_DESCRIPTION
)
