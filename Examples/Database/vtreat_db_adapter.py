
"""
Convert the description of a vtreat variable treatment into a data algebra pipeline.
"""


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

    :param source: input data.
    :param vtreat_descr: .description_matrix() description of transform.
    :param treatment_table_name: name to use for the vtreat_descr table.
    :return: data algebra pipeline implementing specified vtreat treatment
    """

    vtreat_descr = vtreat_descr.copy()
    vtreat_descr['value'] = replace_bad_with_sentinel(vtreat_descr['value'])
    assert isinstance(source, ViewRepresentation)
    assert isinstance(vtreat_descr, pandas.DataFrame)
    ops = source
    step_1_ops = dict()
    # add in is_bad indicators
    im_rows = vtreat_descr.loc[vtreat_descr['treatment_class'] == 'IndicateMissingTransform', :].reset_index(
        inplace=False, drop=True)
    for i in range(im_rows.shape[0]):
        step_1_ops[im_rows['variable'][i]] =\
            f"{im_rows['orig_var'][i]}.is_bad().if_else(1.0, 0.0)"
    # add in value indicators or dummies
    ic_rows = vtreat_descr.loc[vtreat_descr['treatment_class'] == 'IndicatorCodeTransform', :].reset_index(
        inplace=False, drop=True)
    for i in range(ic_rows.shape[0]):
        ov = ic_rows['orig_var'].values[i]
        vi = ic_rows['value'].values[i]
        step_1_ops[ic_rows['variable'][i]] =\
            f"({ov}.coalesce('{bad_sentinel}') == '{vi}').if_else(1.0, 0.0)"
    if len(step_1_ops) > 0:
        ops = ops.extend(step_1_ops)
    # add in any value mapped columns (these should all be string valued)
    mp_rows = (
        data(vtreat_descr=vtreat_descr)
            .select_rows("treatment_class == 'MappedCodeTransform'")
            .project({}, group_by=['orig_var', 'variable'])
        ).ex()
    to_del = set()
    if mp_rows.shape[0] > 0:
        jt = describe_table(vtreat_descr, table_name=treatment_table_name)
        join_key_name = 'vtreat_join_key'
        for i in range(mp_rows.shape[0]):
            ov = mp_rows['orig_var'].values[i]
            vi = mp_rows['variable'].values[i]
            to_del.add(ov)
            ops = (
                ops
                    .extend({join_key_name: f"{ov}.coalesce('{bad_sentinel}')"})
                    .natural_join(
                        b=(
                            jt
                                .select_rows(f"(treatment_class == 'MappedCodeTransform') & (orig_var == '{ov}') & (variable == '{vi}')")
                                .extend({
                                    join_key_name: 'value',
                                    vi: 'replacement',
                                    })
                                .select_columns([join_key_name, vi])
                            ),
                        by=[join_key_name],
                        jointype='left',
                        )
                    .drop_columns([join_key_name])
            )
    # add in any clean numeric copies
    cn_rows = vtreat_descr.loc[vtreat_descr['treatment_class'] == 'CleanNumericTransform', :].reset_index(
        inplace=False, drop=True)
    if cn_rows.shape[0] > 0:
        step_3_ops = dict()
        for i in range(cn_rows.shape[0]):
            step_3_ops[cn_rows['variable'][i]] =\
                f"{cn_rows['orig_var'][i]}.coalesce({cn_rows['replacement'][i]})"
        ops = ops.extend(step_3_ops)
    # remove any re-mapped variables, as they are not numeric
    if len(to_del) > 0:
        to_del = list(to_del)
        to_del.sort()
        ops = ops.drop_columns(to_del)
    return ops
