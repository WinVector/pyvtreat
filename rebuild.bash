
pushd pkg
rm -rf dist build vtreat.egg-info vtreat/__pycache__ tests/__pycache__
pip uninstall -y vtreat
python3 setup.py sdist bdist_wheel
# pip install dist/vtreat-*.tar.gz
popd
pip install --no-deps -e "$(pwd)/pkg"  # sym link to source files
conda list --export > vtreat_dev_env_package_list.txt
pdoc -o docs pkg/vtreat
pytest --cov pkg/vtreat pkg/tests > coverage.txt
# pytest --cov-report term-missing --cov pkg/vtreat pkg/tests > coverage.txt
cat coverage.txt
twine check pkg/dist/*

