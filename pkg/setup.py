# setup.py
import setuptools

DESCRIPTION = """
vtreat is a pandas.DataFrame processor/conditioner that prepares real-world data for predictive modeling in a statistically sound manner. 
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
        'scipy',
    ],
    platforms=['any'],
    license='License :: OSI Approved :: BSD 3-clause License',
    description=DESCRIPTION,
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
