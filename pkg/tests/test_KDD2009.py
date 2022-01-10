

import os
import pandas
import xgboost
import vtreat
import vtreat.cross_plan
import numpy.random
import wvpy.util
import scipy.sparse
from data_algebra.data_ops import descr, TableDescription
import vtreat.vtreat_db_adapter
import data_algebra.BigQuery


# From:
# https://github.com/WinVector/pyvtreat/blob/main/Examples/KDD2009Example/KDD2009Example.ipynb
def test_KDD2009_vtreat_1():
    data_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'KDD2009')
    # data from https://github.com/WinVector/PDSwR2/tree/master/KDD2009
    d = pandas.read_csv(
        os.path.join(data_dir, 'orange_small_train.data.gz'),
        sep='\t',
        header=0)
    orig_vars = list(d.columns)
    # Read in dependent variable we are trying to predict.
    churn = pandas.read_csv(
        os.path.join(data_dir, 'orange_small_train_churn.labels.txt'),
        header=None)
    churn.columns = ["churn"]
    churn['churn'] = churn['churn'] == 1  # replace with True / False
    # Arrange test/train split.
    numpy.random.seed(2020)
    n = d.shape[0]
    # https://github.com/WinVector/pyvtreat/blob/master/Examples/CustomizedCrossPlan/CustomizedCrossPlan.md
    split1 = vtreat.cross_plan.KWayCrossPlanYStratified().split_plan(n_rows=n, k_folds=10, y=churn.iloc[:, 0])
    train_idx = set(split1[0]['train'])
    is_train = [i in train_idx for i in range(n)]
    is_test = numpy.logical_not(is_train)
    d['orig_index'] = range(d.shape[0])
    d_train = d.loc[is_train, :].reset_index(drop=True, inplace=False)
    churn_train = numpy.asarray(churn.loc[is_train, :]["churn"])
    d_test = d.loc[is_test, :].reset_index(drop=True, inplace=False)
    churn_test = numpy.asarray(churn.loc[is_test, :]["churn"])
    # build treatment plan
    plan = vtreat.BinomialOutcomeTreatment(
        outcome_target=True,
        outcome_name='churn',
        cols_to_copy=['orig_index'],
        params=vtreat.vtreat_parameters({
            'filter_to_recommended': True,
            'sparse_indicators': True,
        }))
    cross_frame = plan.fit_transform(d_train, churn_train)
    test_processed = plan.transform(d_test)
    # test transform conversion
    transform_as_data = plan.description_matrix()
    incoming_vars = list(set(transform_as_data['orig_var']))
    ops = vtreat.vtreat_db_adapter.as_data_algebra_pipeline(
        source=TableDescription(
            table_name='d_test',
            column_names=incoming_vars + ['orig_index']),
        vtreat_descr=transform_as_data,
        treatment_table_name='transform_as_data',
        row_keys=['orig_index'],
    )
    test_by_pipeline = ops.eval({
        'd_test': d_test.loc[:, incoming_vars + ['orig_index']],
        'transform_as_data': transform_as_data})
    assert test_by_pipeline.shape[0] == test_processed.shape[0]
    assert test_by_pipeline.shape[1] >= test_processed.shape[1]
    assert not numpy.any(numpy.isnan(test_by_pipeline))
    test_pipeline_cols = set(test_by_pipeline.columns)
    assert numpy.all([c in test_pipeline_cols for c in test_processed.columns])
    test_cols_sorted = list(test_processed.columns)
    test_cols_sorted.sort()
    assert numpy.max(numpy.max(numpy.abs(test_processed[test_cols_sorted] - test_by_pipeline[test_cols_sorted]))) < 1e-5
    # data algebra pipeline in database
    sql = data_algebra.BigQuery.BigQueryModel().to_sql(ops)
    test_on_BigQuery = False
    if test_on_BigQuery:
        db_handle = data_algebra.BigQuery.example_handle()
        db_handle.drop_table('d_test_processed')
        db_handle.insert_table(d_test.loc[:, incoming_vars + ['orig_index']], table_name='d_test', allow_overwrite=True)
        _ = db_handle.insert_table(transform_as_data, table_name='transform_as_data', allow_overwrite=True)
        db_handle.execute(
            f"CREATE TABLE {db_handle.db_model.table_prefix}.d_test_processed AS {db_handle.to_sql(ops)}")
        db_res = db_handle.read_query(
            f"SELECT * FROM {db_handle.db_model.table_prefix}.d_test_processed ORDER BY orig_index")
        assert db_res.shape[0] == test_processed.shape[0]
        assert numpy.max(numpy.max(numpy.abs(test_processed[test_cols_sorted] - db_res[test_cols_sorted]))) < 1e-5
        db_handle.drop_table('d_test')
        db_handle.drop_table('transform_as_data')
        db_handle.drop_table('d_test_processed')
        db_handle.close()
