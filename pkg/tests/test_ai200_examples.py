import os

import pandas

import vtreat


def test_homes_example():
    dir_path = os.path.dirname(os.path.realpath(__file__))
    d = pandas.read_pickle(os.path.join(dir_path, 'homes_76.pkl'))
    assert d.shape[0] == 38
    assert d.shape[1] == 8

    # from AI200: day_01/02_Regression/Part2_LRPractice/LRExample.ipynb
    # documentation: https://github.com/WinVector/pyvtreat/blob/main/Examples/Unsupervised/Unsupervised.md
    treatment = vtreat.UnsupervisedTreatment(
        cols_to_copy=['Price'],
        params=vtreat.unsupervised_parameters({
            'sparse_indicators': False,
            'coders': {'clean_copy',
                       'indicator_code',
                       'missing_indicator'}
        })
    )
    df = treatment.fit_transform(d)

    assert df.shape[0] == 38
    expect_cols = [
        'Price', 'Size', 'Bath', 'Bed', 'Year', 'Garage', 'Lot_lev_4',
        'Lot_lev_5', 'Lot_lev_3', 'Lot_lev_1', 'Lot_lev_2', 'Lot_lev_11',
        'Elem_lev_edge', 'Elem_lev_edison', 'Elem_lev_parker',
        'Elem_lev_harris', 'Elem_lev_adams', 'Elem_lev_crest'
        ]
    assert set(df.columns) == set(expect_cols)


