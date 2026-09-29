"""v4 パイロット（1 ジョブ）。dev だけを使い、本実験の前に次を実測・決定する（test は一切使わない）。

A) スループット: ベースモデルのサンプリングを、クライアント並列数を変えて測る（予算計画の較正）
B) 議論プロトコル: 既定トリオで v4 / v4c / july を dev-A で比較し、事前登録の規則で1つ選ぶ
   （dev-A の macro 精度が最大のもの。v4 との差が 1pt 未満なら v4）
C) 世代0のペルソナ選定（docs/review_2026-09/persona_design.md §4）
   段階1: 候補10体（Belbin 9役割＋plain）と plain の追加系列2本の単独解答を dev-A で測り、能力税フィルタを掛ける
   段階2: AG(2,3) の釣り合い型12チーム＋既定トリオ＋plain×3 を dev-A で議論させ、役割の主効果を推定する
   段階3: 最終候補を dev-B で確認し、事前登録の決定規則で世代0の社会を決める。採用チームの厳密 Shapley も計算する
D) 2026-07 の c7 LoRA の動作確認
"""

from __future__ import annotations

import argparse
import collections
import itertools
import json
import random
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(1, str(ROOT / "vendor"))
sys.path.insert(2, str(ROOT / "scripts/v4"))

from src.evo4.game import banzhaf, shapley  # noqa: E402
from src.evo4.items import index_items  # noqa: E402
from src.evo4.llm import GenConfig, LLMClient  # noqa: E402
from src.evo4.prompts import PERSONAS_JULY  # noqa: E402
from src.evo4.scoring import extract_answer  # noqa: E402
from src.evo4.server import BASE_MODEL, VllmServer  # noqa: E402
from src.evo4.society import Agent, Society  # noqa: E402
from src.evo4.store import CallStore  # noqa: E402
from stats import paired_test  # noqa: E402

BENCHES = ("mmlu_pro", "supergpqa", "math")
SPLIT_SEED = 20260930


def split_dev(items: dict) -> tuple:
    """dev をベンチ別に層化して A（選抜用）と B（確認用）に半分ずつ分ける（固定 seed）。"""
    a, b = [], []
    for bench in BENCHES:
        ids = sorted(i for i, it in items.items() if it.bench == bench)
        random.Random(SPLIT_SEED).shuffle(ids)
        half = len(ids) // 2
        a += ids[:half]
        b += ids[half:]
    return sorted(a), sorted(b)


def macro_of(correct_by_item: dict, items: dict) -> float:
    by = collections.defaultdict(list)
    for item_id, ok in correct_by_item.items():
        by[items[item_id].bench].append(float(ok))
    return float(np.mean([np.mean(v) for v in by.values()]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--items", default=str(ROOT / "data/v4/items/dev.jsonl"))
    parser.add_argument("--pool", default=str(ROOT / "configs/v4/persona_pool.json"))
    parser.add_argument("--out", required=True)
    parser.add_argument("--store", default=None, help="全ジョブ共通の呼び出しストア（既定: <out>/store）")
    parser.add_argument("--adapters", required=True, help="c7 アダプタのディレクトリ（persona_a/b/c）")
    parser.add_argument("--sweep-workers", default="64,128,256")
    parser.add_argument("--sweep-items", type=int, default=120)
    parser.add_argument("--sweep-k", type=int, default=4)
    parser.add_argument("--n-lora", type=int, default=90)
    parser.add_argument("--candidate-protocols", default="v4t,v4ct",
                        help="採用候補（予算内で回せる切り詰め版）。最初のものが既定")
    parser.add_argument("--reference-protocols", default="v4,v4c,july",
                        help="参照として精度だけ測る全文版（第1パイロットの保存分を再利用）")
    parser.add_argument("--workers", type=int, default=192)
    parser.add_argument("--external-base-url", default=None, help="結合試験用: 既存サーバを使う")
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    summary_path = out / "pilot_summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    summary["nvidia_smi"] = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
                                            "--format=csv,noheader"], capture_output=True, text=True).stdout

    def save():
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=1))

    items = index_items([args.items])
    dev_a, dev_b = split_dev(items)
    bench_of = {i: items[i].bench for i in items}
    summary["dev_split"] = {"A": dev_a, "B": dev_b, "seed": SPLIT_SEED}
    pool = json.loads(Path(args.pool).read_text())
    prompts = dict(pool["personas"])
    for extra in pool["plain_team"]:
        prompts[extra] = ""  # plain の追加系列（乱数の系列だけが異なる）
    roles9 = [r for r in pool["personas"] if r != "plain"]
    default_trio = pool["default_trio"]
    plain_team = pool["plain_team"]

    server = VllmServer(max_loras=8, max_lora_rank=16, max_num_seqs=256)
    if args.external_base_url:
        llm = LLMClient(args.external_base_url)
    else:
        server.start()
        llm = LLMClient(server.base_url)
    store = CallStore(tempfile.mkdtemp(prefix="evo4_store_"), args.store or str(out / "store"), sync_every=60)
    config = GenConfig()
    print(f"[pilot] store has {len(store)} records", flush=True)

    def agent_of(name: str) -> Agent:
        return Agent(f"p.{name}", name, BASE_MODEL, prompts[name])

    def team_scores(society: Society, names, ids) -> dict:
        team = [agent_of(n) for n in names]
        return {i: float(society.coalition_result(team, i, 1)["correct"]) for i in ids}

    # ---------------- A) スループット
    base = Agent("base", "base", BASE_MODEL, "")
    sweep = summary.get("sweep", {})
    shuffled = sorted(items)
    random.Random(1).shuffle(shuffled)
    for idx, workers in enumerate(int(w) for w in args.sweep_workers.split(",")):
        subset = shuffled[idx * args.sweep_items:(idx + 1) * args.sweep_items]
        society = Society(llm, store, items, config, protocol="v4", workers=workers)
        tasks = society.sc_tasks(base, subset, gen_seed=1, k=args.sweep_k)
        if all(t.key in store for t in tasks):
            continue
        started = time.time()
        society.run_tasks(tasks, label=f"sweep_w{workers}")
        elapsed = time.time() - started
        tokens = sum((store.get(t.key) or {}).get("n_out") or 0 for t in tasks)
        sweep[str(workers)] = {"calls": len(tasks), "elapsed_s": round(elapsed, 1), "out_tokens": tokens,
                               "out_tok_per_s": round(tokens / elapsed, 1)}
        summary["sweep"] = sweep
        save()
        print(f"[pilot] sweep workers={workers}: {sweep[str(workers)]}", flush=True)

    # ---------------- B) 議論プロトコル（既定トリオ、dev-A）
    candidates_p = args.candidate_protocols.split(",")
    reference_p = [p for p in args.reference_protocols.split(",") if p]
    proto_acc = {}
    for protocol in candidates_p + reference_p:
        society = Society(llm, store, items, config, protocol=protocol, workers=args.workers)
        society.ensure_coalitions([[agent_of(n) for n in default_trio]], dev_a, gen_seed=1,
                                  label=f"protocol_{protocol}")
        proto_acc[protocol] = round(macro_of(team_scores(society, default_trio, dev_a), items), 4)
    default_p = candidates_p[0]
    best = max(candidates_p, key=lambda p: proto_acc[p])
    chosen = best if proto_acc[best] - proto_acc[default_p] >= 0.01 else default_p
    summary["protocol_acc_devA"] = proto_acc
    summary["protocol_candidates"] = candidates_p
    summary["protocol_chosen"] = chosen
    save()
    print(f"[pilot] protocol acc {proto_acc} -> chosen {chosen}", flush=True)
    society = Society(llm, store, items, config, protocol=chosen, workers=args.workers)

    # ---------------- C1) 段階1: 単独解答（dev-A）と能力税フィルタ
    candidates = roles9 + ["plain"] + [p for p in plain_team if p != "plain"]
    society.ensure_coalitions([[agent_of(n)] for n in candidates], dev_a, gen_seed=1, label="stage1_solo")
    solo = {n: team_scores(society, [n], dev_a) for n in candidates}
    solo_macro = {n: round(macro_of(v, items), 4) for n, v in solo.items()}
    tax = {}
    excluded = []
    for n in roles9:
        test = paired_test(solo[n], solo["plain"], bench_of, n_perm=2000, n_boot=4000)
        tax[n] = {"diff_vs_plain": round(test["diff_macro"], 4), "ci90": [round(x, 4) for x in test["ci90"]]}
        if test["diff_macro"] <= -0.02 and test["ci90"][1] < 0:
            excluded.append(n)

    def answers(name, ids):
        agent = agent_of(name)
        return {i: extract_answer((store.get(society.r0_key(agent, i, 1)) or {}).get("text", ""),
                                  items[i].answer_type) for i in ids}

    ans = {n: answers(n, dev_a) for n in candidates}
    other_plain = [p for p in plain_team if p != "plain"][0]
    base_disagree = float(np.mean([ans["plain"][i] != ans[other_plain][i] for i in dev_a]))
    diversity = {n: round(float(np.mean([ans[n][i] != ans["plain"][i] for i in dev_a])) - base_disagree, 4)
                 for n in roles9}
    summary["stage1"] = {"solo_macro": solo_macro, "capability_tax": tax, "excluded": excluded,
                         "plain_resample_disagreement": round(base_disagree, 4),
                         "excess_disagreement_vs_plain": diversity}
    save()
    print(f"[pilot] stage1 solo {solo_macro} excluded {excluded}", flush=True)

    # ---------------- C2) 段階2: 14チーム（dev-A）、役割の主効果
    design = [t["members"] for t in pool["design_teams"]]
    teams = {"+".join(sorted(t)): sorted(t) for t in design}
    teams["+".join(sorted(default_trio))] = sorted(default_trio)
    teams["+".join(plain_team)] = list(plain_team)
    society.ensure_coalitions([[agent_of(n) for n in t] for t in teams.values()], dev_a, gen_seed=1,
                              label="stage2_teams")
    team_acc_a = {k: round(macro_of(team_scores(society, t, dev_a), items), 4) for k, t in teams.items()}
    X = np.zeros((len(design), len(roles9)))
    y = np.zeros(len(design))
    for r, t in enumerate(design):
        for n in t:
            X[r, roles9.index(n)] = 1.0
        y[r] = team_acc_a["+".join(sorted(t))]
    coef, *_ = np.linalg.lstsq(np.hstack([np.ones((len(design), 1)), X]), y, rcond=None)
    effects = coef[1:] - coef[1:].mean()
    main_effect = {n: round(float(e), 4) for n, e in zip(roles9, effects)}
    balanced = [team_acc_a["+".join(sorted(t["members"]))] for t in pool["design_teams"] if t["type"] == "balanced"]
    homog = [team_acc_a["+".join(sorted(t["members"]))] for t in pool["design_teams"] if t["type"] == "homogeneous"]
    obs = float(np.mean(balanced) - np.mean(homog))
    labels = [1] * len(balanced) + [0] * len(homog)
    vals = balanced + homog
    rng = np.random.default_rng(0)
    perm = []
    for _ in range(20000):
        rng.shuffle(labels)
        b = [v for v, lab in zip(vals, labels) if lab == 1]
        h = [v for v, lab in zip(vals, labels) if lab == 0]
        perm.append(np.mean(b) - np.mean(h))
    p_balance = float((np.sum(np.abs(perm) >= abs(obs) - 1e-12) + 1) / (len(perm) + 1))
    summary["stage2"] = {"team_acc_devA": team_acc_a, "main_effect": main_effect,
                         "balance_minus_homogeneous": round(obs, 4), "balance_perm_p": round(p_balance, 4)}
    save()
    print(f"[pilot] stage2 teams {team_acc_a} main effects {main_effect}", flush=True)

    # ---------------- C3) 段階3: 最終候補を dev-B で確認し、決定規則で採否を決める
    allowed = [n for n in roles9 if n not in excluded]
    eligible = {k: t for k, t in teams.items() if k != "+".join(plain_team) and all(n in allowed for n in t)}
    f0 = "+".join(sorted(default_trio))
    f1 = max(eligible, key=lambda k: team_acc_a[k]) if eligible else f0
    f2 = "+".join(sorted(sorted(allowed, key=lambda n: -main_effect[n])[:3]))
    f3 = "+".join(sorted(sorted(roles9, key=lambda n: -solo_macro[n])[:3]))
    finalists = {"F0_default": f0, "F1_best_observed": f1, "F2_main_effect_top3": f2, "F3_solo_top3": f3,
                 "C_plain": "+".join(plain_team)}
    uniq = sorted(set(finalists.values()))
    society.ensure_coalitions([[agent_of(n) for n in k.split("+")] for k in uniq], dev_b, gen_seed=1,
                              label="stage3_B")
    scores_b = {k: team_scores(society, k.split("+"), dev_b) for k in uniq}
    acc_b = {k: round(macro_of(v, items), 4) for k, v in scores_b.items()}
    contenders = {name: key for name, key in finalists.items() if not name.startswith("C_")}
    star_name = max(contenders, key=lambda nm: (acc_b[contenders[nm]], nm == "F0_default"))
    star = contenders[star_name]
    decision = {"finalists": finalists, "acc_devB": acc_b, "best": star_name}
    if star != f0:
        test = paired_test(scores_b[star], scores_b[f0], bench_of)
        decision["best_vs_default"] = {"diff_macro": round(test["diff_macro"], 4),
                                       "ci90": [round(x, 4) for x in test["ci90"]], "p": test["p"]}
        adopt = star if (test["diff_macro"] >= 0.015 and test["ci90"][0] > 0) else f0
    else:
        adopt = f0
    decision["adopted"] = adopt
    decision["rule"] = "F* が既定トリオを +1.5pt 以上かつ 90%CI 下限 > 0 で上回るときだけ採用（それ以外は既定トリオ）"
    adopted = [agent_of(n) for n in adopt.split("+")]
    coalitions = [list(c) for size in (1, 2, 3) for c in itertools.combinations(adopted, size)]
    society.ensure_coalitions(coalitions, dev_b, gen_seed=1, label="stage3_shapley")
    v = {frozenset(): 0.0}
    for c in coalitions:
        v[frozenset(a.agent_id for a in c)] = macro_of(
            {i: float(society.coalition_result(c, i, 1)["correct"]) for i in dev_b}, items)
    players = [a.agent_id for a in adopted]
    decision["shapley_devB"] = {p: round(x, 4) for p, x in shapley(players, v).items()}
    decision["banzhaf_devB"] = {p: round(x, 4) for p, x in banzhaf(players, v).items()}
    decision["coalition_values_devB"] = {"+".join(sorted(k)): round(x, 4) for k, x in v.items() if k}
    summary["stage3"] = decision
    save()
    selected = {"personas": {n: prompts[n] for n in adopt.split("+")}, "order": adopt.split("+"),
                "protocol": chosen, "source": "pilot stage3 (docs/review_2026-09/persona_design.md §4.2)"}
    (out / "personas_selected.json").write_text(json.dumps(selected, ensure_ascii=False, indent=1))
    print(f"[pilot] stage3 acc_B {acc_b} -> adopted {adopt}", flush=True)

    # ---------------- D) c7 LoRA の動作確認（dev-A の一部）
    names = {"critic": "persona_a", "pragmatist": "persona_b", "explorer": "persona_c"}
    c7 = []
    for role, folder in names.items():
        lora = f"c7_{role}"
        if not args.external_base_url:
            server.load_lora(lora, str(Path(args.adapters) / folder))
        c7.append(Agent(f"c7.{role}", role, lora, PERSONAS_JULY[role]))
    society.ensure_coalitions([c7], dev_a[:args.n_lora], gen_seed=1, label="c7_check")

    store.sync()
    summary["n_records"] = len(store)
    summary["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    save()
    if not args.external_base_url:
        server.stop()
    print("[pilot] done", flush=True)


if __name__ == "__main__":
    main()
