
"""
Convert the description of a vtreat variable treatment into a data algebra pipeline.
"""

import numpy
import pandas

from vtreat.vtreat_impl import bad_sentinel, replace_bad_with_sentinel
from data_algebra.data_ops import *


def as_data_algebra_pipeline(
        *,
        source: ViewRepresentation,
        vtreat_descr: pandas.DataFrame,
        treatment_table_name: str) -> ViewRepresentation:
    """
    Convert the description of a vtreat transform (gotten via .description_matrix())
    into a data algebra pipeline.
    See: https://github.com/WinVector/data_algebra and https://github.com/WinVector/pyvtreat .
    Missing and nan are treated as synonums for '_NA_'.
    Another way to use this methodology would be to port this code as a stored procedure
    in a target database of choice, meaning only the vtreat_descr table would be needed on such systems.

    :param source: input data.
    :param vtreat_descr: .description_matrix() description of transform.
                         Expected invariant: CleanNumericTransform doesn't change variable names,
                         all other operations produce new names.
    :param treatment_table_name: name to use for the vtreat_descr table.
    :return: data algebra pipeline implementing specified vtreat treatment
    """

    # belt and suspenders replace missing with sentinel
    vtreat_descr = vtreat_descr.copy()
    vtreat_descr['value'] = replace_bad_with_sentinel(vtreat_descr['value'])
    # check our expected invariants
    assert isinstance(source, ViewRepresentation)
    assert isinstance(vtreat_descr, pandas.DataFrame)
    assert isinstance(treatment_table_name, str)
    # variable produced is function of orig_var and treatment only
    check_fn_reln = (
        data(vtreat_descr=vtreat_descr)
            .project({}, group_by=['treatment', 'orig_var', 'variable'])
            .extend({'one': 1})
            .project({'count': 'one.sum()'}, group_by=['treatment', 'orig_var'])
    ).ex()
    assert numpy.all(check_fn_reln['count'] == 1)
    # variable consumed is function of orig_var and treatment only
    check_fn_reln2 = (
        data(vtreat_descr=vtreat_descr)
            .project({}, group_by=['treatment', 'orig_var', 'variable'])
            .extend({'one': 1})
            .project({'count': 'one.sum()'}, group_by=['treatment', 'variable'])
    ).ex()
    assert numpy.all(check_fn_reln2['count'] == 1)
    # clean copies don't change variable names
    cn_rows = vtreat_descr.loc[vtreat_descr['treatment_class'] == 'CleanNumericTransform', :].reset_index(
        inplace=False, drop=True)
    assert numpy.all(cn_rows['variable'] == cn_rows['orig_var'])
    # operations other than clean copy produce new variable names
    ot_rows = vtreat_descr.loc[vtreat_descr['treatment_class'] != 'CleanNumericTransform', :].reset_index(
        inplace=False, drop=True)
    assert len(set(ot_rows['variable']).intersection(vtreat_descr['orig_var'])) == 0
    # clean copy and re-mapping take disjoint inputs (one alters input as a prep-step, so they would interfere)
    mp_rows = (
        data(vtreat_descr=vtreat_descr)
            .select_rows("treatment_class == 'MappedCodeTransform'")
            .project({}, group_by=['orig_var', 'variable'])
            .order_rows(['orig_var', 'variable'])
        ).ex()
    assert len(set(mp_rows['orig_var']).intersection(cn_rows['orig_var'])) == 0
    # start building up operator pipeline
    ops = source
    step_1_ops = dict()
    # add in is_bad indicators
    im_rows = vtreat_descr.loc[vtreat_descr['treatment_class'] == 'IndicateMissingTransform', :].reset_index(
        inplace=False, drop=True)
    for i in range(im_rows.shape[0]):
        step_1_ops[im_rows['variable'][i]] = f"{im_rows['orig_var'][i]}.is_bad().if_else(1.0, 0.0)"
    # add in general value indicators or dummies
    ic_rows = vtreat_descr.loc[vtreat_descr['treatment_class'] == 'IndicatorCodeTransform', :].reset_index(
        inplace=False, drop=True)
    for i in range(ic_rows.shape[0]):
        ov = ic_rows['orig_var'].values[i]
        vi = ic_rows['value'].values[i]
        step_1_ops[ic_rows['variable'][i]] = f"({ov}.coalesce('{bad_sentinel}') == '{vi}').if_else(1.0, 0.0)"
    if len(step_1_ops) > 0:
        ops = ops.extend(step_1_ops)
    # add in any value mapped columns (these should all be string valued)
    if mp_rows.shape[0] > 0:
        # prepare incoming variables to use sentinel for missing, this is after other steps using these values
        mapping_inputs = list(set([v for v in mp_rows['orig_var'].values]))
        mapping_inputs.sort()
        mapping_outputs = list(set([v for v in mp_rows['variable'].values]))
        mapping_outputs.sort()
        ops = ops.extend({v : f"{v}.coalesce('{bad_sentinel}')" for v in mapping_inputs})
        # do the re-mapping joins, these don't depend on each other
        jt = describe_table(vtreat_descr, table_name=treatment_table_name)
        for i in range(mp_rows.shape[0]):
            ov = mp_rows['orig_var'].values[i]
            vi = mp_rows['variable'].values[i]
            match_q = f"(treatment_class == 'MappedCodeTransform') & (orig_var == '{ov}') & (variable == '{vi}')"
            ops = (
                ops
                    .natural_join(
                        b=(
                            jt
                                .select_rows(match_q)
                                .extend({
                                    ov: 'value',
                                    vi: 'replacement',
                                    })
                                .select_columns([ov, vi])
                            ),
                        by=[ov],
                        jointype='left',
                        )
            )
        # handle any novel values
        ops = ops.extend({v: f'{v}.coalesce(0.0)' for v in mapping_outputs})
    # add in any clean numeric copies, inputs are numeric- so disjoint of categorical processing
    if cn_rows.shape[0] > 0:
        step_3_ops = dict()
        for i in range(cn_rows.shape[0]):
            step_3_ops[cn_rows['variable'][i]] = f"{cn_rows['orig_var'][i]}.coalesce({cn_rows['replacement'][i]})"
        ops = ops.extend(step_3_ops)
    # remove any input variables that are not the same name as variables we produced
    # this prevents non-numerics from leaking forward
    to_del = list(set(vtreat_descr['orig_var']) - set(vtreat_descr['variable']))
    if len(to_del) > 0:
        to_del.sort()
        ops = ops.drop_columns(to_del)
    return ops
