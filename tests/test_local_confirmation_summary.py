import pytest

from analysis.compile_local_confirmation import paired_correctness


def test_paired_correctness_reconciles_all_four_cases_and_accuracy_delta():
    labels = [0, 1, 2, 3, 4]
    result = paired_correctness([0, 1, 0, 0, 4], [0, 0, 2, 0, 0], labels)
    assert result == dict(samples=5, both_correct=1, candidate_only_correct=2,
                          reference_only_correct=1, both_wrong=1, cpu_top1_difference_pp=20.)


def test_empty_predictions_do_not_generate_a_result():
    with pytest.raises(ValueError, match='nonempty'):
        paired_correctness([], [], [])
