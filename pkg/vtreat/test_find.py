import os
import numpy as np
import numpy.random
import pandas as pd

from data_algebra.data_ops import *
import data_algebra.SQLite
import data_algebra.test_util
import vtreat
from vtreat.vtreat_db_adapter import as_data_algebra_pipeline

dir_path = '/Users/johnmount/Documents/work/pyvtreat/pkg/tests'

data_all = pd.read_csv(os.path.join(dir_path, "diabetes_head.csv"))
n = data_all.shape[0]
data_all["orig_index"] = range(n)
d_train = data_all.loc[range(n - 5), :].reset_index(inplace=False, drop=True)
d_app = data_all.loc[range(n - 5, n)].reset_index(inplace=False, drop=True)

# %%

outcome_name = "readmitted"
cols_to_copy = ["orig_index", "encounter_id", "patient_nbr"] + [outcome_name]
vars = ["time_in_hospital", "weight"]
columns = vars + cols_to_copy

# d_train.loc[:, columns]

# %%

treatment = vtreat.BinomialOutcomeTreatment(
    cols_to_copy=cols_to_copy,
    outcome_name=outcome_name,
    outcome_target=True,
    params=vtreat.vtreat_parameters(
        {"sparse_indicators": False, "filter_to_recommended": False, }
    ),
)
d_train_treated = treatment.fit_transform(d_train.loc[:, columns])

d_app_treated = treatment.transform(d_app.loc[:, columns])

# d_app_treated

# %%

transform_as_data = treatment.description_matrix()

# transform_as_data

# %%

ops = as_data_algebra_pipeline(
    source=descr(d_app=d_app.loc[:, columns]),
    vtreat_descr=transform_as_data,
    treatment_table_name="transform_as_data",
    use_cdata_merges=True,
    row_keys=['orig_index'],
)

# print(ops)

# %%

transformed = ops.eval(
    {"d_app": d_app.loc[:, columns], "transform_as_data": transform_as_data}
)