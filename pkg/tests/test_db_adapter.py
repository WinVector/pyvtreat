import os
import numpy as np
import numpy.random
import pandas as pd

from data_algebra.data_ops import *
import data_algebra.SQLite
import data_algebra.test_util
import vtreat
from vtreat.vtreat_db_adapter import as_data_algebra_pipeline


def test_db_adapter_1_cdata():
    # Example from:
    # https://github.com/WinVector/pyvtreat/blob/main/Examples/Database/vtreat_db_adapter.ipynb
    # Data from:
    # https://archive.ics.uci.edu/ml/datasets/Diabetes+130-US+hospitals+for+years+1999-2008

    # data_all = pd.read_csv("diabetes_head.csv")
    dir_path = os.path.dirname(os.path.realpath(__file__))
    data_all = pd.read_csv(os.path.join(dir_path, "diabetes_head.csv"))
    n = data_all.shape[0]
    data_all["orig_index"] = range(n)
    d_train = data_all.loc[range(n - 5), :].reset_index(inplace=False, drop=True)
    d_app = data_all.loc[range(n - 5, n)].reset_index(inplace=False, drop=True)

    outcome_name = "readmitted"
    cols_to_copy = ["orig_index", "encounter_id", "patient_nbr"] + [outcome_name]
    vars = ["time_in_hospital", "weight"]
    columns = vars + cols_to_copy

    # d_train.loc[:, columns]

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

    transform_as_data = treatment.description_matrix()

    # transform_as_data

    ops = as_data_algebra_pipeline(
        source=descr(d_app=d_app.loc[:, columns]),
        vtreat_descr=transform_as_data,
        treatment_table_name="transform_as_data",
        row_keys=['orig_index'],
    )

    # print(ops)

    transformed = ops.eval(
        {"d_app": d_app.loc[:, columns], "transform_as_data": transform_as_data}
    )

    # transformed

    assert data_algebra.test_util.equivalent_frames(transformed, d_app_treated)

    db_handle = data_algebra.SQLite.example_handle()

    sql = db_handle.to_sql(ops)
    assert isinstance(sql, str)
    # print(sql)

    db_handle.insert_table(d_app.loc[:, columns], table_name="d_app")
    db_handle.insert_table(transform_as_data, table_name="transform_as_data")

    db_handle.execute("CREATE TABLE res AS " + sql)

    res_db = db_handle.read_query("SELECT * FROM res ORDER BY orig_index LIMIT 10")

    # res_db

    assert data_algebra.test_util.equivalent_frames(res_db, d_app_treated)

    db_handle.close()


def test_db_adapter_general():

    # set up example data
    def mk_data(
        n_rows: int = 100,
        *,
        outcome_name: str = "y",
        n_cat_vars: int = 5,
        n_num_vars: int = 5,
        add_unknowns: bool = False,
    ):
        step = 1 / np.sqrt(n_cat_vars + n_num_vars)
        cols = dict()
        y = np.random.normal(size=n_rows)
        for i in range(n_cat_vars):
            vname = f"vc_{i}"
            levels = ["a", "b", "c", "none"]
            if add_unknowns:
                levels = levels + ["d"]
            level_values = {v: step * np.random.normal(size=1)[0] for v in levels}
            v = np.random.choice(levels, replace=True, size=n_rows)
            y = y + np.array([level_values[vi] for vi in v])
            v = np.array([vi if vi != "none" else None for vi in v])
            cols[vname] = v
        for i in range(n_num_vars):
            vname = f"vn_{i}"
            v = np.random.normal(size=n_rows)
            y = y + step * v
            v[np.random.uniform(size=n_rows) < 0.24] = None
            cols[vname] = v

        vars = list(cols.keys())
        vars.sort()
        cols[outcome_name] = y
        d = pd.DataFrame(cols)
        d["orig_index"] = range(d.shape[0])
        return d, outcome_name, vars

    d, outcome_name, vars = mk_data(100)
    d_app, _, _ = mk_data(50, add_unknowns=True)
    cols_to_copy = [outcome_name, "orig_index"]
    columns = vars + cols_to_copy

    # get reference result
    treatment = vtreat.NumericOutcomeTreatment(
        cols_to_copy=cols_to_copy,
        outcome_name=outcome_name,
        params=vtreat.vtreat_parameters(
            {"sparse_indicators": False, "filter_to_recommended": False,}
        ),
    )
    d_train_treated = treatment.fit_transform(d)
    assert isinstance(d_train_treated, pd.DataFrame)
    d_app_treated = treatment.transform(d_app)

    # test ops path
    transform_as_data = treatment.description_matrix()
    ops = as_data_algebra_pipeline(
        source=descr(d_app=d),
        vtreat_descr=transform_as_data,
        treatment_table_name="transform_as_data",
        row_keys=["orig_index"],
    )
    ops_source = str(ops)
    assert isinstance(ops_source, str)
    d_app_res = ops.eval({"d_app": d_app, "transform_as_data": transform_as_data})
    assert data_algebra.test_util.equivalent_frames(d_app_treated, d_app_res)

    # test ops db path
    source_descr = TableDescription(table_name="d_app", column_names=columns,)
    db_handle = data_algebra.SQLite.example_handle()
    db_handle.insert_table(d_app.loc[:, columns], table_name="d_app")
    db_handle.insert_table(transform_as_data, table_name="transform_as_data")
    db_handle.execute("CREATE TABLE res AS " + db_handle.to_sql(ops))
    res_db = db_handle.read_query("SELECT * FROM res ORDER BY orig_index")
    assert data_algebra.test_util.equivalent_frames(res_db, d_app_treated)
    db_handle.close()


def test_db_adapter_monster():
    outcome_name = "y"
    row_id_name = 'row_id'
    n_vars = 5

    def mk_data(n_rows: int = 100):
        step = 1 / np.sqrt(n_vars)
        cols = dict()
        y = np.random.normal(size=n_rows)
        for i in range(n_vars):
            vname = f"v_{i}"
            v = np.random.choice(["a", "b"], replace=True, size=n_rows)
            y = y + np.where(v == "a", step, -step)
            cols[vname] = v
        vars = list(cols.keys())
        vars.sort()
        cols[outcome_name] = y
        cols[row_id_name] = range(n_rows)
        d = pd.DataFrame(cols)
        return d, vars

    d, vars = mk_data(100)
    d_app, _ = mk_data(10)
    cols_to_copy = [outcome_name, row_id_name]
    columns = vars + cols_to_copy

    treatment = vtreat.NumericOutcomeTreatment(
        cols_to_copy=cols_to_copy,
        outcome_name=outcome_name,
        params=vtreat.vtreat_parameters(
            {"sparse_indicators": False, "filter_to_recommended": False,}
        ),
    )
    d_train_treated = treatment.fit_transform(d)
    assert isinstance(d_train_treated, pd.DataFrame)
    d_app_treated = treatment.transform(d_app)

    transform_as_data = treatment.description_matrix()
    # transform_as_data.to_csv('example_transform.csv', index=False)

    ops = as_data_algebra_pipeline(
        source=descr(d_app=d),
        vtreat_descr=transform_as_data,
        treatment_table_name="transform_as_data",
        row_keys=[row_id_name],
    )

    ops_source = str(ops)
    assert isinstance(ops_source, str)

    d_app_res = ops.eval({"d_app": d_app, "transform_as_data": transform_as_data})
    assert data_algebra.test_util.equivalent_frames(d_app_treated, d_app_res)
    assert numpy.all([c in d_app_res.columns for c in cols_to_copy])
