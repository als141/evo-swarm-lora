"""SuperGPQA 進行中エントリの暫定精度集計。

gold は c1_base_solo_supergpqa_s1.json の per_item から復元し、
progress キャッシュ (item_id, answer) と突合する。
"""
import json
import sys
from pathlib import Path

SP = Path(__file__).parent

# gold の復元
with (SP / "c1_base_solo_supergpqa_s1.json").open() as f:
    c1 = json.load(f)
per_item = c1["solo"]["base"]["per_item"]
gold = {}
if isinstance(per_item, dict):
    for iid, rec in per_item.items():
        gold[iid] = rec["gold"]
else:
    for rec in per_item:
        gold[rec["item_id"]] = rec["gold"]

print(f"gold items: {len(gold)}", file=sys.stderr)

# 確定済みエントリの再確認（c3, c3p）
for name, path in [
    ("c3_base_team_s1 (確定)", SP / "c3_base_team_supergpqa_s1.json"),
    ("c3p_prompt_team_s1 (確定)", SP / "c3p_prompt_persona_team_supergpqa_s1.json"),
]:
    with path.open() as f:
        d = json.load(f)
    team = d.get("team")
    if team:
        acc = team.get("accuracy")
        n = team.get("n")
        pi = team.get("per_item")
        vals = list(pi.values()) if isinstance(pi, dict) else (pi or [])
        nones = sum(1 for it in vals if it.get("predicted") is None)
        print(f"{name}: acc={acc:.3f} n={n} none={nones}")

# 進行中エントリの暫定値
for name, fn, kind in [
    ("c2_sc9_s1 (暫定)", "progress_sgpqa/c2_s1.jsonl", "sc"),
    ("c5_evolved_team_s1 (暫定)", "progress_sgpqa/c5_s1.jsonl", "team"),
]:
    path = SP / fn
    if not path.exists():
        continue
    correct = 0
    total = 0
    none_ct = 0
    unknown = 0
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            iid = rec["item_id"]
            if iid not in gold:
                unknown += 1
                continue
            total += 1
            ans = rec.get("answer")
            if ans is None:
                none_ct += 1
            elif str(ans).strip().upper() == str(gold[iid]).strip().upper():
                correct += 1
    acc = correct / total if total else 0.0
    print(f"{name}: acc={acc:.3f} ({correct}/{total}) none={none_ct} unknown_id={unknown}")
