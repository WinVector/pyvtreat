import os

import pandas

import vtreat


def test_homes_example():
    dir_path = os.path.dirname(os.path.realpath(__file__))
    d = pandas.read_csv(os.path.join(dir_path, "homes_76.csv"))
    d['Lot'] = d['Lot'].astype(str)
    assert d.shape[0] == 38
    assert d.shape[1] == 8

    # from AI200: day_01/02_Regression/Part2_LRPractice/LRExample.ipynb
    # documentation: https://github.com/WinVector/pyvtreat/blob/main/Examples/Unsupervised/Unsupervised.md
    treatment = vtreat.UnsupervisedTreatment(
        cols_to_copy=["Price"],
        params=vtreat.unsupervised_parameters(
            {
                "sparse_indicators": False,
                "coders": {"clean_copy", "indicator_code", "missing_indicator"},
            }
        ),
    )
    df = treatment.fit_transform(d)

    assert df.shape[0] == d.shape[0]
    expect_cols = [
        "Price",
        "Size",
        "Bath",
        "Bed",
        "Year",
        "Garage",
        "Lot_lev_4",
        "Lot_lev_5",
        "Lot_lev_3",
        "Lot_lev_1",
        "Lot_lev_2",
        "Lot_lev_11",
        "Elem_lev_edge",
        "Elem_lev_edison",
        "Elem_lev_parker",
        "Elem_lev_harris",
        "Elem_lev_adams",
        "Elem_lev_crest",
    ]
    assert set(df.columns) == set(expect_cols)

    expect = pandas.read_csv(os.path.join(dir_path, "homes_76_treated.csv"))
    assert expect.shape == df.shape
    assert set(expect.columns) == set(df.columns)
    assert (expect - df).abs().max().max() < 1.0e-7


def test_diabetes_example():
    dir_path = os.path.dirname(os.path.realpath(__file__))
    data = pandas.read_csv(os.path.join(dir_path, "diabetes_head.csv"))
    assert data.shape[0] == 1000

    # from AI200: day_04/ZZ_homework/soln_dont_peek/diabetes_soln.ipynb
    # https://archive.ics.uci.edu/ml/datasets/Diabetes+130-US+hospitals+for+years+1999-2008

    # sklearn.preprocessing.OneHotEncoder could
    # also perform this task well.

    # documentation:
    #  https://github.com/WinVector/pyvtreat/blob/main/Examples/Classification/Classification.md
    treatment = vtreat.BinomialOutcomeTreatment(
        cols_to_copy=["encounter_id", "patient_nbr", "readmitted"],
        outcome_name="readmitted",
        outcome_target=True,
        params=vtreat.vtreat_parameters(
            {"sparse_indicators": False, "filter_to_recommended": False,}
        ),
    )
    data_treated = treatment.fit_transform(data)

    assert data_treated.shape[0] == data.shape[0]

    expect = {
        "A1Cresult_lev_None",
        "A1Cresult_lev__gt_8",
        "A1Cresult_logit_code",
        "A1Cresult_prevalence_code",
        "acarbose_lev_No",
        "acarbose_logit_code",
        "acarbose_prevalence_code",
        "admission_source_id_lev_1",
        "admission_source_id_lev_7",
        "admission_source_id_logit_code",
        "admission_source_id_prevalence_code",
        "admission_type_id_lev_1",
        "admission_type_id_lev_2",
        "admission_type_id_lev_6",
        "admission_type_id_logit_code",
        "admission_type_id_prevalence_code",
        "age_lev__osq_40-50_cp_",
        "age_lev__osq_50-60_cp_",
        "age_lev__osq_60-70_cp_",
        "age_lev__osq_70-80_cp_",
        "age_lev__osq_80-90_cp_",
        "age_logit_code",
        "age_prevalence_code",
        "change_lev_Ch",
        "change_lev_No",
        "change_logit_code",
        "change_prevalence_code",
        "chlorpropamide_lev_No",
        "chlorpropamide_logit_code",
        "chlorpropamide_prevalence_code",
        "diabetesMed_lev_No",
        "diabetesMed_lev_Yes",
        "diabetesMed_logit_code",
        "diabetesMed_prevalence_code",
        "diag_1_is_bad",
        "diag_1_lev_414",
        "diag_1_logit_code",
        "diag_1_prevalence_code",
        "diag_2_is_bad",
        "diag_2_logit_code",
        "diag_2_prevalence_code",
        "diag_3_is_bad",
        "diag_3_lev_250",
        "diag_3_logit_code",
        "diag_3_prevalence_code",
        "discharge_disposition_id_lev_1",
        "discharge_disposition_id_lev_25",
        "discharge_disposition_id_logit_code",
        "discharge_disposition_id_prevalence_code",
        "encounter_id",
        "gender_lev_Female",
        "gender_lev_Male",
        "gender_logit_code",
        "gender_prevalence_code",
        "glimepiride_lev_No",
        "glimepiride_logit_code",
        "glimepiride_prevalence_code",
        "glipizide_lev_No",
        "glipizide_lev_Steady",
        "glipizide_logit_code",
        "glipizide_prevalence_code",
        "glyburide_lev_No",
        "glyburide_logit_code",
        "glyburide_prevalence_code",
        "insulin_lev_Down",
        "insulin_lev_No",
        "insulin_lev_Steady",
        "insulin_logit_code",
        "insulin_prevalence_code",
        "max_glu_serum_lev_None",
        "max_glu_serum_logit_code",
        "max_glu_serum_prevalence_code",
        "medical_specialty_is_bad",
        "medical_specialty_lev_Cardiology",
        "medical_specialty_lev_Family/GeneralPractice",
        "medical_specialty_lev_InternalMedicine",
        "medical_specialty_lev__NA_",
        "medical_specialty_logit_code",
        "medical_specialty_prevalence_code",
        "metformin_lev_No",
        "metformin_lev_Steady",
        "metformin_logit_code",
        "metformin_prevalence_code",
        "num_lab_procedures",
        "num_medications",
        "num_procedures",
        "number_diagnoses",
        "number_emergency",
        "number_inpatient",
        "number_outpatient",
        "patient_nbr",
        "pioglitazone_lev_No",
        "pioglitazone_logit_code",
        "pioglitazone_prevalence_code",
        "race_is_bad",
        "race_lev_AfricanAmerican",
        "race_lev_Caucasian",
        "race_logit_code",
        "race_prevalence_code",
        "readmitted",
        "repaglinide_lev_No",
        "repaglinide_logit_code",
        "repaglinide_prevalence_code",
        "revisit",
        "rosiglitazone_lev_No",
        "rosiglitazone_logit_code",
        "rosiglitazone_prevalence_code",
        "time_in_hospital",
        "tolazamide_lev_No",
        "tolazamide_logit_code",
        "tolazamide_prevalence_code",
        "tolbutamide_lev_No",
        "tolbutamide_logit_code",
        "tolbutamide_prevalence_code",
        "troglitazone_lev_No",
        "troglitazone_logit_code",
        "troglitazone_prevalence_code",
        "visit_number",
        "weight_is_bad",
        "weight_lev__NA_",
        "weight_logit_code",
        "weight_prevalence_code",
    }
    # assert set(data_treated.columns) == expect

    treatment = vtreat.BinomialOutcomeTreatment(
        cols_to_copy=["encounter_id", "patient_nbr", "readmitted"],
        outcome_name="readmitted",
        outcome_target=True,
        params=vtreat.vtreat_parameters(
            {"sparse_indicators": False, "filter_to_recommended": True,}
        ),
    )
    data_treated = treatment.fit_transform(data)

    assert data_treated.shape[0] == data.shape[0]
    assert data_treated.shape[1] >= 10

    transform_description = treatment.description_matrix()
    assert isinstance(transform_description, pandas.DataFrame)
    assert transform_description.shape[0] > 0
    assert transform_description.shape[1] > 0
