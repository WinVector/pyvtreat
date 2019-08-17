
pushd pkg
rm -rf dist build vtreat.egg-info vtreat/__pycache__
pip uninstall -y vtreat
python3 setup.py sdist bdist_wheel
pip install dist/vtreat-*.tar.gz
popd

