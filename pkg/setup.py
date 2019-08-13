# setup.py
import setuptools

DESCRIPTION = """
vtreat is a pandas.DataFrame processor/conditioner that prepares real-world data for predictive modeling in a statistically sound manner. 
"""
LONG_DESCRIPTION = """
vtreat prepares variables so that data has fewer exceptional cases, making it easier to safely use models in production. 
Common problems vtreat defends against: Inf, NA, too many categorical levels, rare categorical levels, and new categorical levels (levels seen during application, but not during training). 
Reference: "vtreat: a data.frame Processor for Predictive Modeling", Zumel, Mount, 2016, <doi:10.5281/zenodo.1173313>.
"""

setuptools.setup(
    name='vtreat',
    version='0.2.3',
    author='John Mount',
    author_email='jmount@win-vector.com',
    url='https://github.com/WinVector/pyvtreat',
    packages=setuptools.find_packages(),
    install_requires=[
        'numpy',
        'pandas',
        'statistics',
        'scipy'
    ],
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    classifiers=[
          'Intended Audience :: Science/Research',
          'Programming Language :: Python',
          'Programming Language :: Python :: 3',
          'Programming Language :: Python :: 3.5',
          'Programming Language :: Python :: 3.6',
          'Programming Language :: Python :: 3.7',
          'License :: OSI Approved :: BSD License',
    ],
)
