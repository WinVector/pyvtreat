"""
Utils that help with testing. This module is allowed to import many other modules.
"""

import numpy as np
import pandas as pd




def equivalent_frames(
    a: pd.DataFrame,
    b: pd.DataFrame,
    *,
    float_tol: float = 1e-8,
    check_column_order: bool = False,
    cols_case_sensitive: bool = False,
    check_row_order: bool = False,
) -> bool:
    """return False if the frames are equivalent (up to column re-ordering and possible row-reordering).
    Ignores indexing. None and nan are considered equivalent in numeric contexts."""
    assert isinstance(a, pd.DataFrame)
    assert isinstance(b, pd.DataFrame)
    a = a.reset_index(drop=True, inplace=False)
    b = b.reset_index(drop=True, inplace=False)
    if a.shape != b.shape:
        return False
    if a.shape[1] < 1:
        return True
    if a.equals(b):
        return True
    if not cols_case_sensitive:
        a.columns = [c.lower() for c in a.columns]
        b.columns = [c.lower() for c in b.columns]
    a_columns = [c for c in a.columns]
    b_columns = [c for c in b.columns]
    if set(a_columns) != set(b_columns):
        return False
    if check_column_order:
        if not np.all([a.columns[i] == b.columns[i] for i in range(a.shape[0])]):
            return False
    else:
        # re-order b into a's column order
        b = b[a_columns]
        b = b.reset_index(drop=True, inplace=False)
    if a.shape[0] < 1:
        return True
    if not check_row_order:
        a = a.sort_values(by=a_columns, ignore_index=True)
        a = a.reset_index(drop=True, inplace=False)
        b = b.sort_values(by=a_columns, ignore_index=True)
        b = b.reset_index(drop=True, inplace=False)
    for c in a_columns:
        ca = a[c]
        cb = b[c]
        if (ca is None) != (cb is None):
            return False
        if ca is not None:
            if len(ca) != len(cb):
                return False
            ca_null = ca.isnull()
            cb_null = cb.isnull()
            if (ca_null is None) != (cb_null is None):
                return False
            if not np.all(ca_null == cb_null):
                return False
            if not np.all(ca_null):
                ca = ca[ca_null == False]
                cb = cb[cb_null == False]
                ca_can_be_numeric = False
                ca_n = np.asarray([0.0])  # just a typing hint
                # noinspection PyBroadException
                try:
                    ca_n = np.asarray(ca, dtype=float)
                    ca_can_be_numeric = True
                except Exception:
                    pass
                cb_can_be_numeric = False
                cb_n = np.asarray([0.0])  # just a typing hint
                # noinspection PyBroadException
                try:
                    cb_n = np.asarray(cb, dtype=float)
                    cb_can_be_numeric = True
                except Exception:
                    pass
                if ca_can_be_numeric != cb_can_be_numeric:
                    return False
                if ca_can_be_numeric and cb_can_be_numeric:
                    if len(ca_n) != len(cb_n):
                        return False
                    ca_inf = np.isinf(ca_n)
                    cb_inf = np.isinf(cb_n)
                    if np.any(ca_inf != cb_inf):
                        return False
                    if np.any(ca_inf):
                        if np.any(
                            np.sign(ca_n[ca_inf]) != np.sign(cb_n[cb_inf])
                        ):
                            return False
                    if np.any(np.logical_not(ca_inf)):
                        ca_f = ca_n[np.logical_not(ca_inf)]
                        cb_f = cb_n[np.logical_not(cb_inf)]
                        dif = np.abs(ca_f - cb_f) / np.maximum(
                            np.maximum(np.abs(ca_f), np.abs(cb_f)), 1.0
                        )
                        if np.max(dif) > float_tol:
                            return False
                else:
                    if not np.all(ca == cb):
                        return False
    return True
