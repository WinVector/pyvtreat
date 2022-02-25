

conda remove --name vtreat_dev_env --all --yes
conda env create -f vtreat_dev_env.yaml
conda activate vtreat_dev_env
pip install --no-deps -e "$(pwd)/pkg"  # sym link to source files


