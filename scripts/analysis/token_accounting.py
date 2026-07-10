"""計算量の実測 — c7(チーム議論) と c2(SC@9) の総トークン数を llm_calls から集計（査読対応）。

「SC@9 は生成回数ベース（9生成 vs チーム6生成）の比較であり compute-matched とは
言えない」という査読指摘に対し、run002 の LLM 呼び出し完全ログ（全入力メッセージ+
全生成テキスト）から入力・出力トークン数を Qwen3-4B の実トークナイザで実測する。

- 対象: 新環境 6 シードの c7 / c2 全ジョブ（g3, final_c7, robust_c7, remeasure_v1,
  recheck_c2s1, robust_c2）。remeasure_v1 には c1/c5 も含まれるため model 名で分離。
- 入力はチャットテンプレート適用後のトークン数（実際にモデルへ入った長さ）。
- 出力は生成テキストのトークン数。

実行: uv run python scripts/analysis/token_accounting.py
"""
import glob
import gzip
import json
from collections import defaultdict
from pathlib import Path

from transformers import AutoTokenizer

ROOT = Path(__file__).parents[2]
R2 = ROOT / "results/gcs/run002"

# ジョブディレクトリ → 集計キーの割り当て規則
# model 名: チーム = r2_{critic,pragmatist,explorer}（c7）
#           SC/base solo = "base"（c2/c1）、旧チーム = ev_*（c5）
DIRS = ["g3_team_check", "final_c7", "robust_c7",
        "remeasure_v1", "recheck_c2s1", "robust_c2"]


def classify(dirname: str, model: str) -> str | None:
    if model.startswith("r2_"):
        return "c7_team"
    if model.startswith("ev_"):
        return "c5_old_team"
    if model == "Qwen/Qwen3-4B-Instruct-2507":
        # ベースモデル直呼び: remeasure_v1 には c1(solo,1生成/問) と c2(SC@9,9生成/問)
        # が混在するためファイル単位では分離できないが、c1=3,000call・c2=27,000call
        # と呼び出し数が既知なので合算値から差し引き可能。recheck/robust_c2 は c2 のみ。
        return "base_or_sc"
    return None


def main():
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B-Instruct-2507")
    stats = defaultdict(lambda: {"calls": 0, "in_tok": 0, "out_tok": 0})

    for d in DIRS:
        files = sorted(glob.glob(str(R2 / d / "llm_calls" / "*.jsonl.gz")))
        for f in files:
            with gzip.open(f, "rt") as fh:
                for line in fh:
                    rec = json.loads(line)
                    key = classify(d, rec["model"])
                    if key is None:
                        continue
                    key = f"{d}:{key}"
                    ids = tok.apply_chat_template(rec["messages"],
                                                  add_generation_prompt=True)
                    out = tok(rec["response"], add_special_tokens=False)["input_ids"]
                    s = stats[key]
                    s["calls"] += 1
                    s["in_tok"] += len(ids)
                    s["out_tok"] += len(out)
        print(f"[done] {d}")

    print()
    print(f"{'key':<34} {'calls':>8} {'入力tok':>14} {'出力tok':>14} {'合計tok':>14}")
    agg = defaultdict(lambda: {"calls": 0, "in_tok": 0, "out_tok": 0})
    for key, s in sorted(stats.items()):
        print(f"{key:<34} {s['calls']:>8,} {s['in_tok']:>14,} {s['out_tok']:>14,} "
              f"{s['in_tok']+s['out_tok']:>14,}")
        cond = "c7" if "c7_team" in key else ("c2c1" if "base_or_sc" in key else "c5")
        a = agg[cond]
        for k in ("calls", "in_tok", "out_tok"):
            a[k] += s[k]

    print()
    print("=== 条件別合計 ===")
    for cond, a in sorted(agg.items()):
        print(f"{cond:<8} calls={a['calls']:,} in={a['in_tok']:,} out={a['out_tok']:,} "
              f"total={a['in_tok']+a['out_tok']:,}")
    # 1問あたり平均（c7: 6シード×1000問=6000問分 / c2: 下で分離推定）
    if "c7" in agg:
        a = agg["c7"]
        print(f"\nc7: 1問あたり平均  calls={a['calls']/6000:.2f} "
              f"in={a['in_tok']/6000:,.0f} out={a['out_tok']/6000:,.0f} "
              f"total={(a['in_tok']+a['out_tok'])/6000:,.0f} tok")


if __name__ == "__main__":
    main()
