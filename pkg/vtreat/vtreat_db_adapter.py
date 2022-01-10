"""
Convert the description of a vtreat variable treatment into a data algebra pipeline.
"""

from typing import Dict, Iterable, List, Optional, Tuple
import numpy
import pandas

from vtreat.vtreat_impl import bad_sentinel, replace_bad_with_sentinel


from data_algebra.data_ops import data, descr, describe_table, TableDescription, ViewRepresentation
from data_algebra.solutions import def_multi_column_map


def _check_treatment_table(vtreat_descr: pandas.DataFrame):
    """
    Assert if expected invariants don't hold for vtreat_descr.

    :param vtreat_descr: .description_matrix() description of a transform to check.
    :return: no return, assert on failure
    """

    # belt and suspenders replace missing with sentinel
    vtreat_descr = vtreat_descr.copy()
    vtreat_descr["value"] = replace_bad_with_sentinel(vtreat_descr["value"])
    # check our expected invariants
    assert isinstance(vtreat_descr, pandas.DataFrame)
    # numeric is a function of original variable only
    check_fn_relnn = (
        data(vtreat_descr=vtreat_descr)
            .project({}, group_by=["orig_var", "orig_was_numeric"])
            .extend({"one": 1})
            .project({"count": "one.sum()"}, group_by=["orig_var"])
    ).ex()
    assert numpy.all(check_fn_relnn["count"] == 1)
    # variable consumed is function of variable produced and treatment only
    check_fn_reln2 = (
        data(vtreat_descr=vtreat_descr)
            .project({}, group_by=["treatment", "orig_var", "variable"])
            .extend({"one": 1})
            .project({"count": "one.sum()"}, group_by=["treatment", "variable"])
    ).ex()
    assert numpy.all(check_fn_reln2["count"] == 1)
    # clean copies don't change variable names
    cn_rows = vtreat_descr.loc[
        vtreat_descr["treatment_class"] == "CleanNumericTransform", :
    ].reset_index(inplace=False, drop=True)
    assert numpy.all(cn_rows["variable"] == cn_rows["orig_var"])
    # operations other than clean copy produce new variable names
    ot_rows = vtreat_descr.loc[
        vtreat_descr["treatment_class"] != "CleanNumericTransform", :
    ].reset_index(inplace=False, drop=True)
    assert len(set(ot_rows["variable"]).intersection(vtreat_descr["orig_var"])) == 0
    # clean copy and re-mapping take disjoint inputs (one alters input as a prep-step, so they would interfere)
    mp_rows = (
        data(vtreat_descr=vtreat_descr)
        .select_rows("treatment_class == 'MappedCodeTransform'")
        .project({}, group_by=["orig_var", "variable"])
        .order_rows(["orig_var", "variable"])
    ).ex()
    assert len(set(mp_rows["orig_var"]).intersection(cn_rows["orig_var"])) == 0


def as_data_algebra_pipeline(
    *,
    source: TableDescription,
    vtreat_descr: pandas.DataFrame,
    treatment_table_name: str,
    row_keys: Iterable[str],
) -> ViewRepresentation:
    """
    Convert the description of a vtreat transform (gotten via .description_matrix())
    into a data algebra pipeline.
    See: https://github.com/WinVector/data_algebra and https://github.com/WinVector/pyvtreat .
    Missing and nan are treated as synonyms for '_NA_'.
    Assembling the entire pipeline can be expensive. If one is willing to instantiate tables
    it can be better to sequence operations instead of composing them.
    Another way to use this methodology would be to port this code as a stored procedure
    in a target database of choice, meaning only the vtreat_descr table would be needed on such systems.

    :param source: input data.
    :param vtreat_descr: .description_matrix() description of transform.
                         Expected invariant: CleanNumericTransform doesn't change variable names,
                         all other operations produce new names.
    :param treatment_table_name: name to use for the vtreat_descr table.
    :param row_keys: list of columns uniquely keying rows
    :return: data algebra pipeline implementing specified vtreat treatment
    """

    assert isinstance(source, TableDescription)
    assert isinstance(vtreat_descr, pandas.DataFrame)
    assert isinstance(treatment_table_name, str)
    assert row_keys is not None
    assert not isinstance(row_keys, str)
    row_keys = list(row_keys)
    assert len(row_keys) > 0
    assert numpy.all([isinstance(v, str) for v in row_keys])

    _check_treatment_table(vtreat_descr)
    # belt and suspenders replace missing with sentinel
    vtreat_descr = vtreat_descr.copy()
    vtreat_descr["value"] = replace_bad_with_sentinel(vtreat_descr["value"])
    # start building up operator pipeline
    ops = source
    step_1_ops = dict()
    # add in is_bad indicators
    im_rows = vtreat_descr.loc[
        vtreat_descr["treatment_class"] == "IndicateMissingTransform", :
    ].reset_index(inplace=False, drop=True)
    for i in range(im_rows.shape[0]):
        if im_rows['orig_was_numeric'][i]:
            step_1_ops[
                im_rows["variable"][i]
            ] = f"{im_rows['orig_var'][i]}.is_bad().if_else(1.0, 0.0)"
        else:
            step_1_ops[
                im_rows["variable"][i]
            ] = f"({im_rows['orig_var'][i]}.coalesce('{bad_sentinel}') == '{bad_sentinel}').if_else(1.0, 0.0)"
    # add in general value indicators or dummies, all indicators are non-numeric (string)
    ic_rows = vtreat_descr.loc[
        vtreat_descr["treatment_class"] == "IndicatorCodeTransform", :
    ].reset_index(inplace=False, drop=True)
    for i in range(ic_rows.shape[0]):
        ov = ic_rows["orig_var"].values[i]
        vi = ic_rows["value"].values[i]
        step_1_ops[
            ic_rows["variable"][i]
        ] = f"({ov}.coalesce('{bad_sentinel}') == '{vi}').if_else(1.0, 0.0)"
    if len(step_1_ops) > 0:
        ops = ops.extend(step_1_ops)
    # mapped columns
    mapping_table = (
        describe_table(vtreat_descr, table_name=treatment_table_name)
            .select_rows('treatment_class == "MappedCodeTransform"')
            .select_columns(['orig_var', 'value', 'replacement', 'treatment']))
    mapping_rows = mapping_table.transform(vtreat_descr)
    if mapping_rows.shape[0] > 0:
        groups = list(set(mapping_rows['treatment']))
        mapping_rows = mapping_rows.groupby('treatment')
        for group_name in groups:
            mg = mapping_rows.get_group(group_name)
            if mg.shape[0] > 0:
                cols_to_map = list(set(mg['orig_var']))
                cols_to_map_back = [f'{c}_{group_name}' for c in cols_to_map]
                ops_g = def_multi_column_map(
                    source.extend({v: f"{v}.coalesce('{bad_sentinel}')" for v in cols_to_map}),
                    mapping_table=mapping_table.select_rows(f'treatment == "{group_name}"'),
                    row_keys=row_keys,
                    cols_to_map=cols_to_map,
                    cols_to_map_back=cols_to_map_back,
                    coalesce_value=0.0,
                    col_name_key='orig_var',
                    col_value_key='value',
                    mapped_value_key='replacement',
                )
                ops = ops.natural_join(
                    b=ops_g,
                    by=row_keys,
                    jointype='left',
                )
    # add in any clean numeric copies, inputs are numeric- so disjoint of categorical processing
    cn_rows = vtreat_descr.loc[
        vtreat_descr["treatment_class"] == "CleanNumericTransform", :
    ].reset_index(inplace=False, drop=True)
    if cn_rows.shape[0] > 0:
        step_3_exprs = dict()
        for i in range(cn_rows.shape[0]):
            step_3_exprs[
                cn_rows["variable"][i]
            ] = f"{cn_rows['orig_var'][i]}.coalesce({cn_rows['replacement'][i]})"
        ops = ops.extend(step_3_exprs)
    # remove any input variables that are not the same name as variables we produced
    # this prevents non-numerics from leaking forward
    to_del = list(set(vtreat_descr["orig_var"]) - set(vtreat_descr["variable"]))
    if len(to_del) > 0:
        to_del.sort()
        ops = ops.drop_columns(to_del)
    return ops
