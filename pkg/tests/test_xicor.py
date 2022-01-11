
import os
import yaml

import numpy
import pandas

from vtreat.stats_utils import xicor, xicor_for_frame


def test_xicor():
    dir_path = os.path.dirname(os.path.realpath(__file__))
    with open(os.path.join(dir_path, "xicor_examples.yaml"), "r") as in_f:
        examples = yaml.safe_load(in_f)

    x1, x1_std = xicor([1, 2, 3], [1, 2, 3])  # expect 0.25
    assert numpy.abs(x1 - 0.25) < 1e-5
    assert numpy.abs(x1_std) < 1e-5

    x2, x2_std = xicor([1, 2, 3], [3, 2, 1])  # expect 0.25
    assert numpy.abs(x2 - 0.25) < 1e-5
    assert numpy.abs(x2_std) < 1e-5

    x3, x3_std = xicor([1, 2, 3], [1, 3, 2])  # expect -0.125
    assert numpy.abs(x3 - -.125) < 1e-5
    assert numpy.abs(x3_std) < 1e-5

    # compare results
    n_reps = 100
    for ei in range(len(examples)):
        example = examples[ei]
        a = example['a']
        b = example['b']
        ref_xicor = example['xicor']
        our_xicor_mean, our_xicor_se = xicor(a, b, n_reps=n_reps)
        our_xicor_std = numpy.sqrt(n_reps) * our_xicor_se
        assert numpy.abs(numpy.mean(ref_xicor) - our_xicor_mean) < 0.05
        assert numpy.abs(numpy.std(ref_xicor) - our_xicor_std) < 0.05
        # print(f'ref: {np.mean(ref_xicor)} {np.std(ref_xicor)}, ours: {our_xicor_mean} {our_xicor_std}')
