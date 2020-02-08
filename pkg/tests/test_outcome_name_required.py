
import pytest

import vtreat  # https://github.com/WinVector/pyvtreat

def test_outcome_name_required():
    with pytest.raises(Exception):
        vtreat.NumericOutcomeTreatment()

    with pytest.raises(Exception):
        vtreat.BinomialOutcomeTreatment(outcome_target=True)

    with pytest.raises(Exception):
        vtreat.vtreat_api.MultinomialOutcomeTreatment()
