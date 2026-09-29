"""【解析2】行動距離のうちサンプリングノイズ成分の分離。

問い: 進化ログの同役割個体間の不一致率（0.28〜0.45; 重みは cos≥0.9997 でほぼ同一,
同一 seed, max_tokens=512, MMLU-Pro 100問）は、同一モデルを単に再サンプリングした
ときの不一致率と区別できるか。

比較する「同一モデル・再サンプリング」の不一致率（すべて新環境 llm_calls 由来）:
  (a) 同一リクエスト（同一モデル・同一プロンプト・同一 seed）の反復: G2 の3エントリで
      round0 リクエストが完全一致 → vLLM の seed 再現性（common random numbers の成否）
  (b) ベースモデルの SC@9 サンプル同士（seed だけ違う）
  (c) 同一 LoRA エージェントの round0 を別 run-seed で同一問題に対して比較
  (d) 異なるペルソナ LoRA 間の round0 不一致（c7, c5）＝ペルソナ多様性＋サンプリング
いずれも max_tokens=512 打ち切り相当（先頭512トークンで再抽出）も併記する。
不一致の定義は進化ループの behavioral_distance と同じ（None 同士は一致、None と回答は不一致）。

実行: python3 scripts/analysis/ra2_sampling_disagreement.py
"""
import gzip
import hashlib
import itertools
import json
from collections import defaultdict

import numpy as np

from ra_common import OUT, R2, ROOT, dump_json, load_calls_index

TRUNCS = ["", "512", "2048"]


def pair_stats(groups, trunc=""):
    """groups: list of lists of call rows（同一問題の比較対象）。平均の不一致率と正誤不一致率。"""
    dis, cdis, none = [], [], []
    for g in groups:
        for a, b in itertools.combinations(g, 2):
            aa, bb = a.get(f"ans{trunc}"), b.get(f"ans{trunc}")
            dis.append(aa != bb)
            cdis.append(a.get(f"ok{trunc}") != b.get(f"ok{trunc}"))
        none.extend(r.get(f"ans{trunc}") is None for r in g)
    if not dis:
        return None
    return {"n_pairs": len(dis), "disagree": float(np.mean(dis)),
            "correct_discord": float(np.mean(cdis)), "none_rate": float(np.mean(none))}


def per_item_mean_disagree(groups, trunc=""):
    """問題ごとの平均不一致率を問題間で平均（問題あたりのペア数の偏りを除く）。"""
    vals = []
    for g in groups:
        pairs = list(itertools.combinations(g, 2))
        if pairs:
            vals.append(np.mean([a.get(f"ans{trunc}") != b.get(f"ans{trunc}") for a, b in pairs]))
    return float(np.mean(vals)) if vals else None


def main():
    calls = [c for c in load_calls_index() if c.get("item_id")]
    res = {}

    # (a) 同一リクエスト反復（G2: majority / weighted / genselect の round0 が同一）
    raw_key = {}
    for f in sorted((R2 / "g2_aggregation/llm_calls").glob("*.jsonl.gz")):
        with gzip.open(f, "rt") as fh:
            for i, line in enumerate(fh):
                r = json.loads(line)
                raw_key[(f.name, i)] = hashlib.sha1(
                    (r["model"] + str(r["seed"]) + json.dumps(r["messages"], ensure_ascii=False)).encode()
                ).hexdigest()
    same_req = defaultdict(list)
    for c in calls:
        if c["dir"] == "g2_aggregation" and c["kind"] == "team" and c["round"] == 0:
            same_req[raw_key[(c["file"], c["line"])]].append(c)
    grp = [g for g in same_req.values() if len(g) >= 2]
    res["a_same_request_repeat_sgpqa"] = {t or "full": pair_stats(grp, t) for t in TRUNCS}

    # (b) ベース SC@9 サンプル同士（同一 run-seed・同一問題の 9 サンプル）
    sc = defaultdict(list)
    for c in calls:
        if c["kind"] == "sc":
            sc[(c["bench"], c["run_seed"], c["item_id"])].append(c)
    res["b_base_sc_resample"] = {}
    for bench in ("mmlu_pro", "math500", "supergpqa"):
        g = [v for (b, _s, _i), v in sc.items() if b == bench]
        res["b_base_sc_resample"][bench] = {t or "full": pair_stats(g, t) for t in TRUNCS}
        res["b_base_sc_resample"][bench]["per_item_mean_full"] = per_item_mean_disagree(g)

    # (c) 同一 LoRA エージェントの round0、別 run-seed・同一問題
    r0 = defaultdict(list)
    for c in calls:
        if c["kind"] == "team" and c["round"] == 0 and c["dir"] != "g2_aggregation":
            r0[(c["family"], c["model"], c["bench"], c["item_id"])].append(c)
    res["c_same_agent_cross_seed"] = {}
    for fam in ("c7", "c5"):
        for bench in ("mmlu_pro", "math500", "supergpqa"):
            g = [v for (f, _m, b, _i), v in r0.items() if f == fam and b == bench
                 and len({x["run_seed"] for x in v}) >= 2]
            # 同一 seed の重複（無いはず）を除外して seed 間ペアだけにする
            g = [list({x["run_seed"]: x for x in v}.values()) for v in g]
            st = {t or "full": pair_stats(g, t) for t in TRUNCS}
            st["n_items"] = len(g)
            res["c_same_agent_cross_seed"][f"{fam}_{bench}"] = st

    # (d) 異なるペルソナ LoRA 間の round0 不一致（同一 run-seed・同一問題の3体）
    team_r0 = defaultdict(list)
    for c in calls:
        if c["kind"] == "team" and c["round"] == 0 and c["dir"] != "g2_aggregation":
            team_r0[(c["family"], c["bench"], c["run_seed"], c["item_id"])].append(c)
    res["d_inter_persona_r0"] = {}
    for fam in ("c7", "c5"):
        for bench in ("mmlu_pro", "math500", "supergpqa"):
            g = [v for (f, b, _s, _i), v in team_r0.items() if f == fam and b == bench]
            res["d_inter_persona_r0"][f"{fam}_{bench}"] = {t or "full": pair_stats(g, t) for t in TRUNCS}

    # 進化ログの行動距離（参照値）
    log = json.loads((ROOT / "results/evolution_run_log.json").read_text())
    dists = [rec["sharing_distances"][0] for g in log["generations"] for rl in g["roles"].values()
             for rec in rl["candidates"].values()]
    res["evolution_behavioral_distance"] = {
        "mean": float(np.mean(dists)), "min": float(np.min(dists)), "max": float(np.max(dists)),
        "n": len(dists) // 2}

    dump_json(res, OUT / "ra2_sampling_disagreement.json")

    def fmt(s):
        return (f"不一致={s['disagree']:.3f} 正誤不一致={s['correct_discord']:.3f} "
                f"None率={s['none_rate']:.3f} (ペア{s['n_pairs']})") if s else "—"

    print("== (a) 同一リクエスト（同一モデル・プロンプト・seed）の反復, G2 SuperGPQA")
    for t, s in res["a_same_request_repeat_sgpqa"].items():
        print(f"   [{t}] {fmt(s)}")
    print("== (b) ベースモデル SC サンプル同士（seed のみ異なる）")
    for bench, d in res["b_base_sc_resample"].items():
        for t in ("full", "512", "2048"):
            print(f"   {bench:10s} [{t}] {fmt(d[t])}")
    print("== (c) 同一 LoRA エージェント round0・別 run-seed・同一問題")
    for k, d in res["c_same_agent_cross_seed"].items():
        print(f"   {k:16s} 問題{d['n_items']:4d}  [full] {fmt(d['full'])}  [512] {fmt(d['512'])}")
    print("== (d) 異なるペルソナ間 round0（同一 run-seed）")
    for k, d in res["d_inter_persona_r0"].items():
        print(f"   {k:16s} [full] {fmt(d['full'])}  [512] {fmt(d['512'])}")
    print(f"== 進化ログの行動距離: {res['evolution_behavioral_distance']}")


if __name__ == "__main__":
    main()
