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

    def test_math_answer_line(self):
        assert extract_answer("Reasoning...\nANSWER: 2\\sqrt{5}", "math") == "2sqrt(5)"

    def test_math_boxed_fallback(self):
        assert extract_answer("thus \\boxed{\\frac{1}{3}} is it.", "math") == "1/3"

    def test_double_answer_prefix_letter(self):
        # "ANSWER: ANSWER: C" で単語 ANSWER の 'A' を誤抽出しない（run001 v3 回帰）
        assert extract_answer("ANSWER: ANSWER: C", "letter") == "C"

    def test_double_answer_prefix_math(self):
        assert extract_answer("ANSWER: ANSWER: 202", "math") == "202"


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


class TestAnswerFormatInstruction:
    def test_math_gets_math_format(self):
        from src.evalx.debate import answer_format_instruction

        # 旧実装は math が letter 書式に落ちるバグがあった（run001 v3 回帰）
        assert "<letter>" not in answer_format_instruction("math")
        assert "<final simplified answer>" in answer_format_instruction("math")

    def test_unknown_type_raises(self):
        import pytest

        from src.evalx.debate import answer_format_instruction

        with pytest.raises(ValueError):
            answer_format_instruction("essay")


class TestDebatePromptStyles:
    def test_standard_prompt_unchanged(self):
        # v3本実験と同一文言であることの回帰テスト（画像再ビルドで挙動が変わらないこと）
        from src.evalx.debate import DEBATE_UPDATE_INSTRUCTIONS

        assert DEBATE_UPDATE_INSTRUCTIONS["standard"] == (
            "\nCarefully examine the other agents' reasoning. Point out any errors, "
            "then provide your own updated step-by-step solution. "
            "You may keep or change your previous answer."
        )

    def test_conditional_requires_specific_error(self):
        from src.evalx.debate import DEBATE_UPDATE_INSTRUCTIONS

        text = DEBATE_UPDATE_INSTRUCTIONS["conditional"]
        assert "ONLY if" in text and "keep your original answer" in text

    def test_prompt_builder_uses_style(self):
        from src.evalx.debate import _debate_user_prompt
        from src.evalx.tasks import TaskItem

        item = TaskItem(item_id="x", question="Q?", gold="A")
        std = _debate_user_prompt(item, {"other": "sol"})
        cond = _debate_user_prompt(item, {"other": "sol"}, "conditional")
        assert "You may keep or change" in std and "ONLY if" in cond

    def test_anonymize_hides_agent_names(self):
        from src.evalx.debate import _debate_user_prompt
        from src.evalx.tasks import TaskItem

        item = TaskItem(item_id="x", question="Q?", gold="A")
        others = {"critic": "sol1", "explorer": "sol2"}
        plain = _debate_user_prompt(item, others)
        anon = _debate_user_prompt(item, others, anonymize=True, shuffle_seed=7)
        assert "Agent critic" in plain
        assert "critic" not in anon and "explorer" not in anon
        assert "--- Agent 1 ---" in anon and "--- Agent 2 ---" in anon
        assert "sol1" in anon and "sol2" in anon


class TestV3Aggregation:
    def test_weighted_vote_prefers_high_confidence(self):
        from src.evalx.debate import weighted_vote

        # 2票のBより高confidenceの1票Aが勝つ
        assert weighted_vote([("A", 0.9), ("B", 0.3), ("B", 0.3)]) == "A"

    def test_weighted_vote_none_confidence_counts_as_one(self):
        from src.evalx.debate import weighted_vote

        assert weighted_vote([("A", None), ("A", None), ("B", 0.9)]) == "A"

    def test_weighted_vote_all_none_answers(self):
        from src.evalx.debate import weighted_vote

        assert weighted_vote([(None, 0.5), (None, None)]) is None

    def test_genselect_parses_choice_and_respects_shuffle(self):
        from src.evalx.client import GenerationConfig
        from src.evalx.debate import genselect_adjudicate
        from src.evalx.tasks import TaskItem

        captured = {}

        class FakeClient:
            def chat(self, model, messages, config):
                captured["messages"] = messages
                return "Comparing candidates...\nBEST: 2"

        item = TaskItem(item_id="x", question="Q?", gold="A")
        candidates = [("A", "solution text A"), ("B", "solution text B"), ("C", "solution text C")]
        picked = genselect_adjudicate(
            FakeClient(), "base", item, candidates, GenerationConfig(), shuffle_seed=3
        )
        # 表示順は shuffle_seed=3 の並びなので、BEST:2 は表示2番目の候補の抽出回答を返す
        import random

        order = list(range(3))
        random.Random(3).shuffle(order)
        assert picked == candidates[order[1]][0]
        # 候補は匿名ラベルで提示される
        user_msg = captured["messages"][1]["content"]
        assert "--- Candidate 1 ---" in user_msg and "--- Candidate 3 ---" in user_msg

    def test_genselect_unparseable_returns_none(self):
        from src.evalx.client import GenerationConfig
        from src.evalx.debate import genselect_adjudicate
        from src.evalx.tasks import TaskItem

        class FakeClient:
            def chat(self, model, messages, config):
                return "I think candidate two is best."

        item = TaskItem(item_id="x", question="Q?", gold="A")
        picked = genselect_adjudicate(
            FakeClient(), "base", item, [("A", "sa"), ("B", "sb")], GenerationConfig()
        )
        assert picked is None


class TestCallLogger:
    def test_logs_full_context_and_response(self, tmp_path, monkeypatch):
        import json

        monkeypatch.setenv("EVALX_LOG_DIR", str(tmp_path))
        from src.evalx.client import _CallLogger

        logger = _CallLogger()
        record = {
            "model": "m",
            "messages": [{"role": "user", "content": "こんにちは"}],
            "response": "やあ",
        }
        logger.log(record)
        files = list(tmp_path.glob("calls_*.jsonl"))
        assert len(files) == 1
        loaded = json.loads(files[0].read_text(encoding="utf-8").strip())
        assert loaded["messages"][0]["content"] == "こんにちは"
        assert loaded["response"] == "やあ"

    def test_disabled_without_env(self, monkeypatch):
        monkeypatch.delenv("EVALX_LOG_DIR", raising=False)
        from src.evalx.client import _CallLogger

        logger = _CallLogger()
        logger.log({"model": "m"})  # パスなしでも例外を出さない
        assert logger._path is None
