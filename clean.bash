
pushd pkg
rm -rf dist build vtreat.egg-info vtreat/__pycache__ tests/__pycache__ docs
popd
pip uninstall -y vtreat


