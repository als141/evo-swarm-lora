import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest

from src.evalx.debate import majority_vote
from src.evalx.shapley import all_coalitions, shapley_values
from src.evalx.stats import bootstrap_accuracy_diff, mcnemar_exact
from src.evalx.tasks import extract_answer, is_correct


class TestExtractAnswer:
    def test_number_answer_line(self):
        assert extract_answer("Step 1... \nANSWER: 42", "number") == "42"

    def test_number_with_comma_and_dollar(self):
        assert extract_answer("ANSWER: $1,234", "number") == "1234"

    def test_number_decimal_integer_normalization(self):
        assert extract_answer("ANSWER: 12.0", "number") == "12"

    def test_number_fallback_last_number(self):
        assert extract_answer("The result is 7 so total is 21.", "number") == "21"

    def test_number_multiple_answer_lines_uses_last(self):
        text = "ANSWER: 5\nWait, correcting.\nANSWER: 8"
        assert extract_answer(text, "number") == "8"

    def test_letter_answer_line(self):
        assert extract_answer("Reasoning...\nANSWER: C", "letter") == "C"

    def test_letter_with_parenthesis(self):
        assert extract_answer("ANSWER: (B) because...", "letter") == "B"

    def test_letter_fallback(self):
        assert extract_answer("I think the answer is D .", "letter") == "D"

    def test_no_answer(self):
        assert extract_answer("I do not know.", "letter") is None

    def test_negative_number(self):
        assert extract_answer("ANSWER: -3", "number") == "-3"


class TestIsCorrect:
    def test_number_equivalence(self):
        assert is_correct("42", "42", "number")
        assert is_correct("42.0", "42", "number")
        assert not is_correct("41", "42", "number")

    def test_letter_case_insensitive(self):
        assert is_correct("c", "C", "letter")

    def test_none_prediction(self):
        assert not is_correct(None, "C", "letter")


class TestMajorityVote:
    def test_simple_majority(self):
        assert majority_vote(["A", "A", "B"]) == "A"

    def test_all_none(self):
        assert majority_vote([None, None, None]) is None

    def test_none_ignored(self):
        assert majority_vote(["B", None, "B"]) == "B"

    def test_tie_is_deterministic(self):
        first = majority_vote(["A", "B", "C"], tie_break_seed=7)
        second = majority_vote(["A", "B", "C"], tie_break_seed=7)
        assert first == second


class TestShapley:
    def test_all_coalitions_count(self):
        assert len(all_coalitions(["a", "b", "c"])) == 7

    def test_symmetric_agents(self):
        names = ["a", "b", "c"]
        values = {}
        for coalition in all_coalitions(names):
            values[frozenset(coalition)] = 0.3 * len(coalition)
        result = shapley_values(names, values)
        for v in result.values():
            assert v == pytest.approx(0.3)

    def test_efficiency_property(self):
        # Shapley 値の総和は v(全体連合) - v(∅) に一致する
        names = ["a", "b", "c"]
        values = {
            frozenset(["a"]): 0.5,
            frozenset(["b"]): 0.4,
            frozenset(["c"]): 0.3,
            frozenset(["a", "b"]): 0.65,
            frozenset(["a", "c"]): 0.6,
            frozenset(["b", "c"]): 0.5,
            frozenset(["a", "b", "c"]): 0.7,
        }
        result = shapley_values(names, values)
        assert sum(result.values()) == pytest.approx(0.7)

    def test_missing_coalition_raises(self):
        with pytest.raises(KeyError):
            shapley_values(["a", "b"], {frozenset(["a"]): 0.5})


class TestStats:
    def test_mcnemar_no_disagreement(self):
        pairs = [(True, True), (False, False)]
        assert mcnemar_exact(pairs)["p_value"] == 1.0

    def test_mcnemar_strong_difference(self):
        pairs = [(False, True)] * 30 + [(True, True)] * 70
        result = mcnemar_exact(pairs)
        assert result["p_value"] < 0.001
        assert result["c"] == 30

    def test_bootstrap_diff_sign(self):
        pairs = [(False, True)] * 40 + [(True, True)] * 60
        result = bootstrap_accuracy_diff(pairs, n_resamples=2000, seed=1)
        assert result["diff"] == pytest.approx(0.4)
        assert result["ci_lower"] > 0
