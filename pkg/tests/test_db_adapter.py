

import os
import pandas as pd

from data_algebra.data_ops import *
import data_algebra.SQLite
import data_algebra.test_util
import vtreat
from vtreat.vtreat_db_adapter import as_data_algebra_pipeline


def test_db_adapter_1():
    # Example from:
    # https://github.com/WinVector/pyvtreat/blob/main/Examples/Database/vtreat_db_adapter.ipynb
    # Data from:
    # https://archive.ics.uci.edu/ml/datasets/Diabetes+130-US+hospitals+for+years+1999-2008

    # data_all = pd.read_csv("diabetes_head.csv")
    dir_path = os.path.dirname(os.path.realpath(__file__))
    data_all = pd.read_csv(os.path.join(dir_path, "diabetes_head.csv"))
    n = data_all.shape[0]
    data_all['orig_index'] = range(n)
    d_train = data_all.loc[range(n-5), :].reset_index(inplace=False, drop=True)
    d_app = data_all.loc[range(n-5, n)].reset_index(inplace=False, drop=True)

    #%%

    outcome_name = "readmitted"
    cols_to_copy = ["orig_index", "encounter_id", "patient_nbr"] + [outcome_name]
    vars = ['time_in_hospital', 'weight']
    columns = vars + cols_to_copy

    # d_train.loc[:, columns]


    #%%

    treatment = vtreat.BinomialOutcomeTreatment(
        cols_to_copy=cols_to_copy,
        outcome_name=outcome_name,
        outcome_target=True,
        params=vtreat.vtreat_parameters(
            {"sparse_indicators": False, "filter_to_recommended": False,}
        ),
    )
    d_train_treated = treatment.fit_transform(d_train.loc[:, columns])

    d_app_treated = treatment.transform(d_app.loc[:, columns])

    # d_app_treated

    #%%

    transform_as_data = treatment.description_matrix()

    # transform_as_data

    #%%

    ops = as_data_algebra_pipeline(
        source=descr(d_app=d_app.loc[:, columns]),
        vtreat_descr=transform_as_data,
        treatment_table_name='transform_as_data',
    )

    # print(ops)

    #%%

    transformed = ops.eval({
        'd_app': d_app.loc[:, columns],
        'transform_as_data': transform_as_data})

    # transformed

    #%%


    assert data_algebra.test_util.equivalent_frames(transformed, d_app_treated)

    #%%

    db_handle = data_algebra.SQLite.example_handle()

    sql = db_handle.to_sql(ops)
    assert isinstance(sql, str)
    # print(sql)

    #%%

    db_handle.insert_table(d_app.loc[:, columns], table_name='d_app')
    db_handle.insert_table(transform_as_data, table_name='transform_as_data')

    db_handle.execute('CREATE TABLE res AS ' + sql)

    res_db = db_handle.read_query('SELECT * FROM res ORDER BY orig_index LIMIT 10')

    # res_db

    #%%

    assert data_algebra.test_util.equivalent_frames(res_db, d_app_treated)

    #%%

    db_handle.close()

