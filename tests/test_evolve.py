import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest

from src.evalx.client import GenerationConfig
from src.evalx.tasks import TaskItem, TaskSpec
from src.evolve.loop import (
    CoalitionEvaluator,
    Individual,
    behavioral_distance,
    compute_candidate_fitness,
    sharing_penalty,
)


class FakeClient:
    """モデル名ごとに固定回答を返す ChatClient 互換のフェイク。

    debate では他者回答が提示されても各エージェントは自分の固定回答を維持する。
    """

    def __init__(self, answer_map):
        self._answer_map = answer_map

    def chat(self, model, messages, config):
        return f"Reasoning...\nANSWER: {self._answer_map[model]}"


def make_task(n=4):
    items = [TaskItem(item_id=f"q{i}", question=f"Question {i}", gold="A") for i in range(n)]
    return TaskSpec(name="fake", answer_type="letter", items=items)


def make_individual(name, role):
    return Individual(name=name, role=role, adapter_dir=f"/tmp/{name}", persona_prompt="")


class TestSharingPenalty:
    def test_no_neighbors(self):
        assert sharing_penalty([], sigma=0.3) == 1.0

    def test_identical_neighbor_halves(self):
        # 距離 0 の個体が 1 つ → niche count 2 → 0.5
        assert sharing_penalty([0.0], sigma=0.3) == pytest.approx(0.5)

    def test_distant_neighbor_no_penalty(self):
        assert sharing_penalty([0.9], sigma=0.3) == 1.0


class TestBehavioralDistance:
    def test_identical(self):
        result = {"per_item": {"q0": {"predicted": "A"}, "q1": {"predicted": "B"}}}
        assert behavioral_distance(result, result) == 0.0

    def test_all_different(self):
        a = {"per_item": {"q0": {"predicted": "A"}, "q1": {"predicted": "B"}}}
        b = {"per_item": {"q0": {"predicted": "C"}, "q1": {"predicted": "D"}}}
        assert behavioral_distance(a, b) == 1.0


class TestCandidateFitness:
    def test_shapley_reflects_contribution(self):
        """常に正解する候補は、常に誤答する候補より高い Shapley 値を得る。"""
        task = make_task(n=4)
        config = GenerationConfig(seed=1)

        reps = {
            "pragmatist": make_individual("rep_p", "pragmatist"),
            "explorer": make_individual("rep_e", "explorer"),
        }
        good = make_individual("good_critic", "critic")
        bad = make_individual("bad_critic", "critic")

        # 代表 2 体は誤答 B を出す。good は正解 A、bad は誤答 C。
        answer_map = {"rep_p": "B", "rep_e": "B", "good_critic": "A", "bad_critic": "C"}
        evaluator = CoalitionEvaluator(FakeClient(answer_map), task, config, rounds=1, tie_break_seed=0)

        record_good = compute_candidate_fitness(good, reps, evaluator)
        record_bad = compute_candidate_fitness(bad, reps, evaluator)

        assert record_good["solo_accuracy"] == 1.0
        assert record_bad["solo_accuracy"] == 0.0
        assert record_good["shapley"] > record_bad["shapley"]

    def test_shapley_efficiency(self):
        """効率性: Shapley 値の合計 = チーム精度（v(∅)=0 のため）。"""
        task = make_task(n=4)
        config = GenerationConfig(seed=1)
        reps = {
            "pragmatist": make_individual("rep_p", "pragmatist"),
            "explorer": make_individual("rep_e", "explorer"),
        }
        candidate = make_individual("cand", "critic")
        answer_map = {"rep_p": "A", "rep_e": "B", "cand": "A"}
        evaluator = CoalitionEvaluator(FakeClient(answer_map), task, config, rounds=1, tie_break_seed=0)

        record = compute_candidate_fitness(candidate, reps, evaluator)
        from src.evalx.shapley import shapley_values

        names = record["team_members"]
        coalition_values = {
            frozenset(k.split("+")): v for k, v in record["coalition_accuracy"].items()
        }
        values = shapley_values(names, coalition_values)
        assert sum(values.values()) == pytest.approx(record["team_accuracy"])

    def test_coalition_cache_reused(self):
        """代表ペアの評価は候補間で再利用される（27 → 共有分減の確認）。"""
        task = make_task(n=2)
        config = GenerationConfig(seed=1)
        reps = {
            "pragmatist": make_individual("rep_p", "pragmatist"),
            "explorer": make_individual("rep_e", "explorer"),
        }
        answer_map = {"rep_p": "A", "rep_e": "A", "c1": "A", "c2": "B"}
        evaluator = CoalitionEvaluator(FakeClient(answer_map), task, config, rounds=1, tie_break_seed=0)

        compute_candidate_fitness(make_individual("c1", "critic"), reps, evaluator)
        cache_size_after_first = len(evaluator.cache)
        compute_candidate_fitness(make_individual("c2", "critic"), reps, evaluator)
        cache_size_after_second = len(evaluator.cache)

        # 1 候補目: 7 連合。2 候補目: 代表のみの 3 連合はキャッシュ済みで +4 のみ
        assert cache_size_after_first == 7
        assert cache_size_after_second == 11
