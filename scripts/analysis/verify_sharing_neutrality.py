"""fitness sharing が本実験の選抜に影響しなかったことの検証（査読対応）。

run001 進化ログの全世代・全役割について:
  (1) K=2 のサブ集団では行動距離が対称なため sharing 係数が役割内で常に同値
      → argmax が raw_fitness (Shapley) と一致し、選抜に影響しない（構造的無効）。
  (2) 加えて 18 ケース中 16 ケースで行動距離が niche 半径 sigma=0.3 を超え、
      係数自体が 1.0（割引が発動していない）。

実行: uv run python scripts/analysis/verify_sharing_neutrality.py
"""
import json
from pathlib import Path

LOG = Path(__file__).parents[2] / "results/gcs/run001/evolution/run_log.json"


def main() -> None:
    d = json.loads(LOG.read_text())
    n_cases = n_equal = n_rank_same = n_no_discount = 0
    for g in d["generations"]:
        for role, rl in g["roles"].items():
            cands = rl["candidates"]
            pens = [c["sharing_penalty"] for c in cands.values()]
            raws = {n: c["raw_fitness"] for n, c in cands.items()}
            fits = {n: c["fitness"] for n, c in cands.items()}
            n_cases += 1
            n_equal += len(set(round(p, 12) for p in pens)) == 1
            n_rank_same += max(raws, key=raws.get) == max(fits, key=fits.get)
            n_no_discount += all(abs(p - 1.0) < 1e-12 for p in pens)
            print(f"gen{g['generation']} {role:<11} penalties={pens} "
                  f"selected={rl['selected']}")
    print(f"\n全 {n_cases} ケース中:")
    print(f"  役割内で sharing 係数が同値（選抜順位に影響なし）: {n_equal}")
    print(f"  raw_fitness の順位 = sharing 後の順位:            {n_rank_same}")
    print(f"  係数=1.0（距離が sigma 超で割引不発動）:           {n_no_discount}")
    assert n_equal == n_rank_same == n_cases


if __name__ == "__main__":
    main()
