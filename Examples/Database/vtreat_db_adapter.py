
import numpy
import pandas

from vtreat.vtreat_impl import bad_sentinel, replace_bad_with_sentinel
from data_algebra.data_ops import *



def as_data_algebra_pipeline(
        source: ViewRepresentation,
        vtreat_descr: pandas.DataFrame) -> ViewRepresentation:
    """
    Convert the description of a vtreat transform (gotten via .description_matrix())
    into a data algebra pipeline.
    See: https://github.com/WinVector/data_algebra and https://github.com/WinVector/pyvtreat .
    Missing and nan are treated as synonums for '_NA_'.

    :param source: input data.
    :param vtreat_descr: .description_matrix() description of transform.
    :return: data algebra pipeline implementing specified vtreat treatment
    """

    vtreat_descr = vtreat_descr.copy()
    vtreat_descr['value'] = replace_bad_with_sentinel(vtreat_descr['value'])
    assert isinstance(source, ViewRepresentation)
    assert isinstance(vtreat_descr, pandas.DataFrame)
    step_1_ops = dict()
    im_rows = vtreat_descr.loc[vtreat_descr['treatment_class'] == 'IndicateMissingTransform', :].reset_index(
        inplace=False, drop=True)
    ic_rows = vtreat_descr.loc[vtreat_descr['treatment_class'] == 'IndicatorCodeTransform', :].reset_index(
        inplace=False, drop=True)
    cn_rows = vtreat_descr.loc[vtreat_descr['treatment_class'] == 'CleanNumericTransform', :].reset_index(
        inplace=False, drop=True)
    mp_rows = vtreat_descr.loc[vtreat_descr['treatment_class'] == 'MappedCodeTransform', :].reset_index(
        inplace=False, drop=True)
    for i in range(im_rows.shape[0]):
        step_1_ops[im_rows['variable'][i]] =\
            f"{im_rows['orig_var'][i]}.is_bad().if_else(1.0, 0.0)"
    for i in range(ic_rows.shape[0]):
        step_1_ops[ic_rows['variable'][i]] =\
            f"{ic_rows['orig_var'][i]}.is_bad('{bad_sentinel}', {ic_rows['orig_var'][i]}) == {ic_rows['value'][i]}"
    ops = source.extend(step_1_ops)
    jt = descr(code_table=vtreat_descr)
    for i in range(mp_rows.shape[0]):
        ops = (
            ops
                .extend({'join_key': f"{mp_rows['orig_var'][i]}.is_bad('{bad_sentinel}', {mp_rows['orig_var'][i]})"})
                .natural_join(
                    b=(
                        jt
                            .select_rows(f"(treatment_class == 'MappedCodeTransform') & (orig_var == {mp_rows['orig_var'][i]})")
                            .extend({
                                'join_key': mp_rows['orig_var'][i],
                                mp_rows['orig_var'][i]: 'replacement',
                                })
                            .select_columns(['join_key', mp_rows['orig_var'][i]])
                        ),
                    by=['join_key'],
                    jointype='left',
                    )
                .drop_columns(['join_key'])
        )
    step_3_ops = dict()
    for i in range(cn_rows.shape[0]):
        step_3_ops[cn_rows['variable'][i]] =\
            f"{cn_rows['orig_var'][i]}.is_bad().if_else({cn_rows['replacement'][i]}, {cn_rows['orig_var'][i]})"
