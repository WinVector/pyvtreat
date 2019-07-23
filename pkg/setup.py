# setup.py
import setuptools

setuptools.setup(
    name='vtreat',
    version='0.1',
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
    classifiers=[
        'License :: OSI Approved :: BSD-3-Clause'
    ]
)
