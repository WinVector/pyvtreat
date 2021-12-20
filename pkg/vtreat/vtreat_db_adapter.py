"""
Convert the description of a vtreat variable treatment into a data algebra pipeline.
"""

from typing import Dict, List, Tuple
import numpy
import pandas

from vtreat.vtreat_impl import bad_sentinel, replace_bad_with_sentinel


have_data_algebra = False
try:
    from data_algebra.data_ops import *

    have_data_algebra = True
except FileNotFoundError:
    pass


def check_treatment_table(vtreat_descr: pandas.DataFrame):
    """
    Assert if expected invariants don't hold for vtreat_descr.

    :param vtreat_descr: .description_matrix() description of a transform to check.
    :return: no return, assert on failure
    """

    global have_data_algebra
    assert have_data_algebra
    # belt and suspenders replace missing with sentinel
    vtreat_descr = vtreat_descr.copy()
    vtreat_descr["value"] = replace_bad_with_sentinel(vtreat_descr["value"])
    # check our expected invariants
    assert isinstance(vtreat_descr, pandas.DataFrame)
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


def _build_data_pipelines_stages(
    *,
    source: ViewRepresentation,
    vtreat_descr: pandas.DataFrame,
    treatment_table_name: str,
    stage_3_name: str,
) -> Tuple[ViewRepresentation, List[str], List[Dict], ViewRepresentation]:
    """
    Convert the description of a vtreat transform (gotten via .description_matrix())
    into data algebra pipeline components.
    See: https://github.com/WinVector/data_algebra and https://github.com/WinVector/pyvtreat .
    Missing and nan are treated as synonyms for '_NA_'.
    Another way to use this methodology would be to port this code as a stored procedure
    in a target database of choice, meaning only the vtreat_descr table would be needed on such systems.

    :param source: input data.
    :param vtreat_descr: .description_matrix() description of transform.
                         Expected invariant: CleanNumericTransform doesn't change variable names,
                         all other operations produce new names.
    :param treatment_table_name: name to use for the vtreat_descr table.
    :param stage_3_name: name for stage 3 operators
    :return: phase1 pipeline,  map result names, map stages, phase3 pipeline
    """

    global have_data_algebra
    assert have_data_algebra
    assert isinstance(source, ViewRepresentation)
    assert isinstance(vtreat_descr, pandas.DataFrame)
    assert isinstance(treatment_table_name, str)
    assert isinstance(stage_3_name, str)
    check_treatment_table(vtreat_descr)
    # belt and suspenders replace missing with sentinel
    vtreat_descr = vtreat_descr.copy()
    vtreat_descr["value"] = replace_bad_with_sentinel(vtreat_descr["value"])
    # start building up operator pipeline
    stage_1_ops = source
    step_1_ops = dict()
    # add in is_bad indicators
    im_rows = vtreat_descr.loc[
        vtreat_descr["treatment_class"] == "IndicateMissingTransform", :
    ].reset_index(inplace=False, drop=True)
    for i in range(im_rows.shape[0]):
        step_1_ops[
            im_rows["variable"][i]
        ] = f"{im_rows['orig_var'][i]}.is_bad().if_else(1.0, 0.0)"
    # add in general value indicators or dummies
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
        stage_1_ops = stage_1_ops.extend(step_1_ops)
    # add in any value mapped columns (these should all be string valued)
    mp_rows = (
        data(vtreat_descr=vtreat_descr)
        .select_rows("treatment_class == 'MappedCodeTransform'")
        .project({}, group_by=["orig_var", "variable"])
        .order_rows(["orig_var", "variable"])
    ).ex()
    stage_3_cols = list(stage_1_ops.column_names)
    stage_3_ops = TableDescription(table_name=stage_3_name, column_names=stage_3_cols)
    map_vars = []
    mapping_steps = []
    if mp_rows.shape[0] > 0:
        # prepare incoming variables to use sentinel for missing, this is after other steps using these values
        mapping_inputs = list(set([v for v in mp_rows["orig_var"].values]))
        mapping_inputs.sort()
        mapping_outputs = list(set([v for v in mp_rows["variable"].values]))
        mapping_outputs.sort()
        stage_3_cols = stage_3_cols + mapping_outputs
        stage_3_ops = TableDescription(
            table_name=stage_3_name, column_names=stage_3_cols
        )
        stage_1_ops = stage_1_ops.extend(
            {v: f"{v}.coalesce('{bad_sentinel}')" for v in mapping_inputs}
        )
        # do the re-mapping joins, these don't depend on each other
        jt = describe_table(vtreat_descr, table_name=treatment_table_name)
        for i in range(mp_rows.shape[0]):
            # print(f'map({i}/{mp_rows.shape[0]}) {datetime.datetime.now()}')
            ov = mp_rows["orig_var"].values[i]
            vi = mp_rows["variable"].values[i]
            match_q = f"(treatment_class == 'MappedCodeTransform') & (orig_var == '{ov}') & (variable == '{vi}')"
            bi = (
                jt.select_rows(match_q)
                .extend({ov: "value", vi: "replacement"})
                .select_columns([ov, vi])
            )
            mi_table = vtreat_descr.loc[
                (vtreat_descr["treatment_class"] == "MappedCodeTransform")
                & (vtreat_descr["orig_var"] == ov)
                & (vtreat_descr["variable"] == vi),
                :,
            ].reset_index(inplace=False, drop=True)
            mi = {
                k: v
                for k, v in zip(
                    mi_table["value"].values, mi_table["replacement"].values
                )
            }
            map_vars.append(vi)
            mapping_steps.append({"bi": bi, "ov": ov, "vi": vi, "mi": mi})
        # handle any novel values
        stage_3_ops = stage_3_ops.extend(
            {v: f"{v}.coalesce(0.0)" for v in mapping_outputs}
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
        stage_3_ops = stage_3_ops.extend(step_3_exprs)
    # remove any input variables that are not the same name as variables we produced
    # this prevents non-numerics from leaking forward
    to_del = list(set(vtreat_descr["orig_var"]) - set(vtreat_descr["variable"]))
    if len(to_del) > 0:
        to_del.sort()
        stage_3_ops = stage_3_ops.drop_columns(to_del)
    return stage_1_ops, map_vars, mapping_steps, stage_3_ops


def as_data_algebra_pipeline(
    *,
    source: ViewRepresentation,
    vtreat_descr: pandas.DataFrame,
    treatment_table_name: str,
    use_case_merges: bool = False,
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
    :param use_case_merges: if True use CASE WHEN statements instead of JOINs to merge values.
    :return: data algebra pipeline implementing specified vtreat treatment
    """

    global have_data_algebra
    assert have_data_algebra
    assert isinstance(source, ViewRepresentation)
    assert isinstance(vtreat_descr, pandas.DataFrame)
    assert isinstance(treatment_table_name, str)
    ops, map_vars, mapping_steps, stage_3_ops = _build_data_pipelines_stages(
        source=source,
        vtreat_descr=vtreat_descr,
        treatment_table_name=treatment_table_name,
        stage_3_name="vtreat_temp_stage_3",
    )
    if use_case_merges:
        merge_statements = {
            map_step["vi"]: f'{map_step["ov"]}.mapv({map_step["mi"].__repr__()}, 0.0)'
            for map_step in mapping_steps
        }
        ops = ops.extend(merge_statements)
    else:
        for map_step in mapping_steps:
            ops = ops.natural_join(
                b=map_step["bi"], by=[map_step["ov"]], jointype="left",
            )
    composed_ops = ops >> stage_3_ops
    return composed_ops


def as_sql_update_sequence(
    *,
    db_model,
    source: TableDescription,
    vtreat_descr: pandas.DataFrame,
    treatment_table_name: str,
    stage_3_name: str,
    result_name: str,
) -> List[str]:
    """
    Convert the description of a vtreat transform (gotten via .description_matrix())
    into a SQL update sequence.
    See: https://github.com/WinVector/data_algebra and https://github.com/WinVector/pyvtreat .
    Missing and nan are treated as synonyms for '_NA_'.
    Assembling the entire pipeline can be expensive. If one is willing to instantiate tables
    it can be better to sequence operations instead of composing them.
    Another way to use this methodology would be to port this code as a stored procedure
    in a target database of choice, meaning only the vtreat_descr table would be needed on such systems.

    :param db_model: data algebra database model or handle for SQL translation
    :param source: input data.
    :param vtreat_descr: .description_matrix() description of transform.
                         Expected invariant: CleanNumericTransform doesn't change variable names,
                         all other operations produce new names.
    :param treatment_table_name: name to use for the vtreat_descr table.
    :param stage_3_name: name for one of the temp tables.
    :param result_name: name for result table.
    :return: list of SQL statements
    """

    # translate the transform
    ops, map_vars, mapping_steps, stage_3_ops = _build_data_pipelines_stages(
        source=source,
        vtreat_descr=vtreat_descr,
        treatment_table_name=treatment_table_name,
        stage_3_name=stage_3_name,
    )
    # give variables pre-update values
    ops = ops.extend({v: "0.0" for v in map_vars})

    def update_code(i):
        """
        Build one update statement.

        :param i:
        :return:
        """
        step_i = mapping_steps[i]
        ov = step_i["ov"]
        vi = step_i["vi"]
        update_stmt = f"""
    WITH tmp_update AS (
      SELECT
        value AS {db_model.quote_identifier(ov)},
        replacement AS {db_model.quote_identifier(vi)}
      FROM
        {db_model.quote_identifier(treatment_table_name)}
      WHERE
        (treatment_class = {db_model.quote_string('MappedCodeTransform')})
        AND (orig_var = {db_model.quote_string(ov)})
        AND (variable == {db_model.quote_string(vi)})
    )
    UPDATE
      {db_model.quote_identifier(stage_3_name)}
    SET {db_model.quote_identifier(vi)} = tmp_update.{db_model.quote_identifier(vi)}
    FROM
      tmp_update
    WHERE
       {db_model.quote_identifier(stage_3_name)}.{db_model.quote_identifier(ov)} = tmp_update.{db_model.quote_identifier(ov)}
    """
        return update_stmt

    sql_sequence = (
        [f"DROP TABLE IF EXISTS {db_model.quote_identifier(stage_3_name)}"]
        + [f"DROP TABLE IF EXISTS {db_model.quote_identifier(result_name)}"]
        + [
            f"CREATE TABLE {db_model.quote_identifier(stage_3_name)} AS \n"
            + db_model.to_sql(ops)
        ]
        + [update_code(i) for i in range(len(mapping_steps))]
        + [
            f"CREATE TABLE {db_model.quote_identifier(result_name)} AS \n"
            + db_model.to_sql(stage_3_ops)
        ]
        + [f"DROP TABLE {db_model.quote_identifier(stage_3_name)}"]
    )
    return sql_sequence
