
pushd pkg
rm -rf dist build vtreat.egg-info vtreat/__pycache__ tests/__pycache__
pip uninstall -y vtreat
python3 setup.py sdist bdist_wheel
pip install dist/vtreat-*.tar.gz
popd
pytest
pytest --cov pkg/vtreat pkg/tests > coverage.txt
# pytest --cov-report term-missing --cov pkg/vtreat pkg/tests > coverage.txt
cat coverage.txt
twine check pkg/dist/*

