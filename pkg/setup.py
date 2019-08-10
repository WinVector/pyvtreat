# setup.py
import setuptools

setuptools.setup(
    name='vtreat',
    version='0.2.2',
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
          'Intended Audience :: Science/Research',
          'Programming Language :: Python',
          'Programming Language :: Python :: 3',
          'Programming Language :: Python :: 3.5',
          'Programming Language :: Python :: 3.6',
          'Programming Language :: Python :: 3.7',
          'License :: OSI Approved :: BSD License',
    ],
)
