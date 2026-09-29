"""v4 進化的社会学習のドライバ（1 系統 = 1 ジョブ。世代・工程単位で再開可能）。

世代 t（親 = 世代 t-1 の代表チーム）:
 1. debate  : 親チームが学習用問題 train_g{t-1} で議論する（round0 + round1）
 2. data    : 正解に至った発話から役割ごとの学習データを作る
              a（自己学習）: 自分の正解 round0 と正解 round1
              b（社会学習）: a ＋ 自分が誤り他者が正解した問題での他者の正解 round0（自分のペルソナで学ぶ）
 3. train   : 親アダプタから a, b を継続学習（vLLM を止めて GPU を使う）
 4. cross   : c = ΔW 空間で a と b を平均し rank 16 に SVD 再分解（兄弟交叉）
 5. select  : 候補 {親, a, b, c} から役割ごとに次の代表を選ぶ
              shapley/banzhaf: 親チームの文脈で全 7 連合を dev で実測し貢献度が最大の候補
              solo           : dev の単独精度が最大の候補（アブレーション A1）
              none           : 常に b（選抜なしの社会学習）
 6. test    : 新しい代表チームを test で評価する
dev 精度はベンチマーク等重み（macro）平均。
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import subprocess
import sys
import tempfile
import time
from itertools import combinations
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(1, str(ROOT / "vendor"))

from src.evo4.game import banzhaf, shapley  # noqa: E402
from src.evo4.items import index_items, load_items  # noqa: E402
from src.evo4.llm import GenConfig, LLMClient  # noqa: E402
from src.evo4.prompts import PERSONAS, r0_messages, r1_messages  # noqa: E402
from src.evo4.scoring import extract_answer, is_correct  # noqa: E402
from src.evo4.server import BASE_MODEL, VllmServer  # noqa: E402
from src.evo4.society import Agent, Society, stable_seed  # noqa: E402
from src.evo4.store import CallStore  # noqa: E402

ROLES = ("critic", "pragmatist", "explorer")  # --personas で上書きされる
BENCHES = ("mmlu_pro", "supergpqa", "math")


def agent_to_dict(agent: Agent, adapter: str | None) -> dict:
    return {"agent_id": agent.agent_id, "role": agent.role, "model": agent.model,
            "persona": agent.persona, "adapter": adapter}


def dict_to_agent(d: dict) -> Agent:
    return Agent(d["agent_id"], d["role"], d["model"], d["persona"])


class Runner:
    def __init__(self, args):
        global ROLES, PERSONAS
        if args.personas:
            spec = json.loads(Path(args.personas).read_text())
            PERSONAS = dict(spec["personas"])  # 役割名 -> ペルソナのプロンプト
            ROLES = tuple(spec.get("order", sorted(PERSONAS)))
        self.args = args
        self.out = Path(args.out)
        self.out.mkdir(parents=True, exist_ok=True)
        self.state_path = self.out / "state.json"
        self.state = json.loads(self.state_path.read_text()) if self.state_path.exists() else {
            "lineage": args.lineage, "select": args.select, "generations": {}}
        self.dev_ids = [i.item_id for i in load_items(args.dev)]
        self.test_ids = [i.item_id for i in load_items(args.test)]
        paths = [args.dev, args.test] + [str(Path(args.train_dir) / f"train_g{k}.jsonl")
                                         for k in range(args.generations)]
        self.items = index_items(paths)
        self.server = VllmServer(max_loras=args.max_loras, max_lora_rank=16, max_num_seqs=256)
        self.store = CallStore(tempfile.mkdtemp(prefix="evo4_store_"), args.store, sync_every=60)
        self.config = GenConfig()
        self.llm = None
        self.adapters: Dict[str, str] = {}  # vLLM モデル名 -> アダプタのパス

    # ------------------------------------------------------------------ utils
    def log(self, msg: str) -> None:
        print(f"[evolve:{self.args.lineage}] {msg}", flush=True)

    def save(self) -> None:
        tmp = self.state_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.state, ensure_ascii=False, indent=1))
        tmp.replace(self.state_path)

    def gen_state(self, t: int) -> dict:
        return self.state["generations"].setdefault(str(t), {"steps": {}})

    def society(self) -> Society:
        return Society(self.llm, self.store, self.items, self.config, protocol=self.args.protocol,
                       workers=self.args.workers, log=self.log)

    def ensure_server(self) -> None:
        if self.args.external_base_url:  # 結合試験用: 外部（偽）サーバを使い、起動・LoRA 登録はしない
            if self.llm is None:
                self.llm = LLMClient(self.args.external_base_url)
            return
        if self.server.proc is None:
            self.server.start()
            self.llm = LLMClient(self.server.base_url)
        for name, path in self.adapters.items():
            self.server.load_lora(name, path)

    def register(self, d: dict) -> Agent:
        if d.get("adapter"):
            self.adapters[d["model"]] = d["adapter"]
        return dict_to_agent(d)

    def macro_acc(self, coalition: List[Agent], ids: List[str]) -> Dict[str, float]:
        soc = self.society()
        per = {b: [] for b in BENCHES}
        for item_id in ids:
            per[self.items[item_id].bench].append(soc.coalition_result(coalition, item_id, 1)["correct"])
        out = {b: (sum(v) / len(v) if v else None) for b, v in per.items()}
        vals = [v for v in out.values() if v is not None]
        out["macro"] = sum(vals) / len(vals)
        out["micro"] = sum(sum(v) for v in per.values()) / sum(len(v) for v in per.values())
        return out

    # ------------------------------------------------------------------ generation 0
    def generation0(self) -> None:
        g = self.gen_state(0)
        if "reps" not in g:
            # パイロットの候補プールと同じ名前（p.<ペルソナ名>）にして、生成済みの呼び出しを再利用する
            g["reps"] = {r: agent_to_dict(Agent(f"p.{r}", r, BASE_MODEL, PERSONAS[r]), None) for r in ROLES}
            self.save()
        reps = [self.register(g["reps"][r]) for r in ROLES]
        if not g["steps"].get("dev"):
            self.ensure_server()
            coalitions = [[a] for a in reps] + [[a, b] for i, a in enumerate(reps) for b in reps[i + 1:]] + [reps]
            self.society().ensure_coalitions(coalitions, self.dev_ids, 1, label="g0_dev")
            g["dev_team"] = self.macro_acc(reps, self.dev_ids)
            g["steps"]["dev"] = True
            self.save()
        if not self.args.skip_test and not g["steps"].get("test"):
            self.ensure_server()
            self.society().ensure_coalitions([reps], self.test_ids, 1, label="g0_test")
            g["test_team"] = self.macro_acc(reps, self.test_ids)
            g["steps"]["test"] = True
            self.save()
        self.log(f"g0 dev={g.get('dev_team')} test={g.get('test_team')}")

    # ------------------------------------------------------------------ generation t
    def child_id(self, t: int, role: str, variant: str) -> str:
        prefix = "shared" if t <= self.args.share_until else self.args.lineage
        return f"{prefix}.g{t}.{role}.{variant}"

    def build_datasets(self, parents: List[Agent], ids: List[str]) -> Dict[str, Dict[str, List[List[dict]]]]:
        soc = self.society()
        data = {r: {"a": [], "b": []} for r in ROLES}
        for item_id in ids:
            item = self.items[item_id]
            r0 = {}
            for agent in parents:
                rec = self.store.get(soc.r0_key(agent, item_id, 1))
                ok = rec is not None and rec.get("finish") == "stop" and is_correct(
                    extract_answer(rec["text"], item.answer_type), item.gold, item.answer_type)
                r0[agent.role] = (rec, ok)
            for agent in parents:
                own_rec, own_ok = r0[agent.role]
                others = [o for o in parents if o.agent_id != agent.agent_id]
                if own_ok:
                    data[agent.role]["a"].append(
                        r0_messages(agent.persona, item.question, item.answer_type)
                        + [{"role": "assistant", "content": own_rec["text"]}])
                r1 = self.store.get(soc.r1_key(agent, others, item_id, 1))
                if r1 is not None and r1.get("finish") == "stop" and own_rec is not None and is_correct(
                        extract_answer(r1["text"], item.answer_type), item.gold, item.answer_type):
                    other_texts = [self.store.get(soc.r0_key(o, item_id, 1))["text"] for o in others]
                    data[agent.role]["a"].append(
                        r1_messages(self.args.protocol, agent.persona, item.question, item.answer_type,
                                    own_rec["text"], other_texts,
                                    stable_seed(item_id, 1, agent.role, "shuffle"))
                        + [{"role": "assistant", "content": r1["text"]}])
                if not own_ok:  # 社会学習: 自分が誤り、他者が正解した問題で他者の解き方を学ぶ
                    for other in others:
                        o_rec, o_ok = r0[other.role]
                        if o_ok:
                            data[agent.role]["b"].append(
                                r0_messages(agent.persona, item.question, item.answer_type)
                                + [{"role": "assistant", "content": o_rec["text"]}])
                            break
        rng = random.Random(stable_seed(self.args.lineage, "data", len(ids)))
        for role in ROLES:
            a = data[role]["a"]
            b = a + data[role]["b"]
            data[role]["a"] = rng.sample(a, min(len(a), self.args.max_examples))
            data[role]["b"] = rng.sample(b, min(len(b), self.args.max_examples))
        return data

    def generation(self, t: int) -> None:
        prev = self.gen_state(t - 1)
        g = self.gen_state(t)
        self.adapters = {}  # 過去世代のアダプタは載せない
        parents = [self.register(prev["reps"][r]) for r in ROLES]
        parent_adapter = {r: prev["reps"][r]["adapter"] for r in ROLES}
        train_ids = [i.item_id for i in load_items(str(Path(self.args.train_dir) / f"train_g{t - 1}.jsonl"))]

        # 1. debate
        if not g["steps"].get("debate"):
            self.ensure_server()
            self.society().ensure_coalitions([parents], train_ids, 1, label=f"g{t}_debate")
            g["train_team"] = self.macro_acc(parents, train_ids)
            g["steps"]["debate"] = True
            self.save()

        # 2-3. data + train（共有世代では既存アダプタを再利用）
        children: Dict[str, Dict[str, dict]] = g.setdefault("children", {})
        if not g["steps"].get("train"):
            data = None
            for role in ROLES:
                for variant in ("a", "b"):
                    if self.args.select == "none" and variant == "a" and t > self.args.share_until:
                        continue  # 選抜なし系統は b だけを使う
                    cid = self.child_id(t, role, variant)
                    adir = Path(self.args.adapters) / cid
                    if not (adir / "train_info.json").exists():
                        if data is None:
                            data = self.build_datasets(parents, train_ids)
                            g["data_counts"] = {r: {v: len(data[r][v]) for v in ("a", "b")} for r in ROLES}
                            self.save()
                        if self.server.proc is not None:
                            self.server.stop()
                        lr = self.args.lr_first if parent_adapter[role] is None else self.args.lr_next
                        ex_path = Path("/tmp") / f"{cid}.examples.jsonl"
                        with ex_path.open("w", encoding="utf-8") as handle:
                            for messages in data[role][variant]:
                                handle.write(json.dumps(messages, ensure_ascii=False) + "\n")
                        cmd = [sys.executable, str(ROOT / "scripts/v4/train_child.py"), "--examples", str(ex_path),
                               "--out", str(adir), "--seed", str(stable_seed(cid)), "--lr", str(lr)]
                        if parent_adapter[role]:
                            cmd += ["--parent", parent_adapter[role]]
                        if self.args.fake_train:  # 結合試験用: 小さなダミーのアダプタを書く
                            cmd = [sys.executable, str(ROOT / "tests/v4/fake_train.py"), str(ex_path), str(adir)]
                        self.log(f"training {cid} on {len(data[role][variant])} examples (lr={lr})")
                        subprocess.run(cmd, check=True)
                        info = json.loads((adir / "train_info.json").read_text())
                        self.log(f"trained {cid}: n={info['n_examples']} loss={info['final_loss']} "
                                 f"{info['elapsed_s']}s")
                    children.setdefault(role, {})[variant] = agent_to_dict(
                        Agent(cid, role, cid.replace(".", "_"), PERSONAS[role]), str(adir))
            g["steps"]["train"] = True
            self.save()

        # 4. cross（兄弟交叉 c = ΔW 平均）
        if not g["steps"].get("cross"):
            if self.args.select != "none" or t <= self.args.share_until:
                from src.models.lora_ops import delta_blend_lora
                for role in ROLES:
                    cid = self.child_id(t, role, "c")
                    adir = Path(self.args.adapters) / cid
                    if not (adir / "adapter_model.safetensors").exists():
                        tmp = Path("/tmp") / cid
                        if tmp.exists():
                            shutil.rmtree(tmp)
                        delta_blend_lora(children[role]["a"]["adapter"], children[role]["b"]["adapter"],
                                         str(tmp), alpha=0.5)
                        shutil.copytree(tmp, adir, dirs_exist_ok=True)
                    children[role]["c"] = agent_to_dict(
                        Agent(cid, role, cid.replace(".", "_"), PERSONAS[role]), str(adir))
            g["steps"]["cross"] = True
            self.save()

        # 5. select
        if not g["steps"].get("select"):
            for role in ROLES:
                for d in children.get(role, {}).values():
                    self.register(d)
            self.ensure_server()
            cand = {r: [parents[ROLES.index(r)]] + [dict_to_agent(d) for _, d in sorted(children.get(r, {}).items())]
                    for r in ROLES}
            fitness: Dict[str, dict] = {}
            if self.args.select in ("shapley", "banzhaf"):
                coalitions = []
                for role in ROLES:
                    ctx = [p for p in parents if p.role != role]
                    for c in cand[role]:
                        coalitions += [[c], [c, ctx[0]], [c, ctx[1]], [c, *ctx]]
                    coalitions += [[ctx[0]], [ctx[1]], ctx]
                self.society().ensure_coalitions(coalitions, self.dev_ids, 1, label=f"g{t}_select")
                for role in ROLES:
                    ctx = [p for p in parents if p.role != role]
                    for c in cand[role]:
                        players = [c.agent_id, ctx[0].agent_id, ctx[1].agent_id]
                        by_id = {c.agent_id: c, ctx[0].agent_id: ctx[0], ctx[1].agent_id: ctx[1]}
                        v = {frozenset(): 0.0}
                        for size in (1, 2, 3):
                            for combo in combinations(players, size):
                                v[frozenset(combo)] = self.macro_acc([by_id[p] for p in combo], self.dev_ids)["macro"]
                        fitness[c.agent_id] = {
                            "role": role, "shapley": shapley(players, v)[c.agent_id],
                            "banzhaf": banzhaf(players, v)[c.agent_id],
                            "solo": v[frozenset([c.agent_id])], "team": v[frozenset(players)]}
                key = self.args.select
            elif self.args.select == "solo":
                self.society().ensure_coalitions([[c] for r in ROLES for c in cand[r]], self.dev_ids, 1,
                                                 label=f"g{t}_solo")
                for role in ROLES:
                    for c in cand[role]:
                        fitness[c.agent_id] = {"role": role, "solo": self.macro_acc([c], self.dev_ids)["macro"]}
                key = "solo"
            else:
                key = None
            reps = {}
            for role in ROLES:
                if key is None:
                    chosen = children[role]["b"]
                else:
                    best = max(cand[role], key=lambda c: (fitness[c.agent_id][key], c.agent_id))
                    chosen = (prev["reps"][role] if best.agent_id == parents[ROLES.index(role)].agent_id
                              else next(d for d in children[role].values() if d["agent_id"] == best.agent_id))
                reps[role] = chosen
            g["fitness"] = fitness
            g["reps"] = reps
            g["steps"]["select"] = True
            self.save()
            self.log(f"g{t} selected: {[reps[r]['agent_id'] for r in ROLES]}")

        # 6. dev / test of new reps
        self.adapters = {}
        reps_agents = [self.register(g["reps"][r]) for r in ROLES]
        if not g["steps"].get("dev_team"):
            self.ensure_server()
            self.society().ensure_coalitions([reps_agents], self.dev_ids, 1, label=f"g{t}_devteam")
            g["dev_team"] = self.macro_acc(reps_agents, self.dev_ids)
            g["steps"]["dev_team"] = True
            self.save()
        if not self.args.skip_test and not g["steps"].get("test"):
            self.ensure_server()
            self.society().ensure_coalitions([reps_agents], self.test_ids, 1, label=f"g{t}_test")
            g["test_team"] = self.macro_acc(reps_agents, self.test_ids)
            g["steps"]["test"] = True
            self.save()
        self.log(f"g{t} dev={g.get('dev_team')} test={g.get('test_team')}")

    def run(self) -> None:
        started = time.time()
        self.generation0()
        for t in range(1, self.args.generations + 1):
            self.generation(t)
        self.store.sync()
        self.state["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
        self.state["elapsed_s"] = self.state.get("elapsed_s", 0) + round(time.time() - started)
        self.save()
        self.server.stop()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lineage", required=True)
    parser.add_argument("--select", choices=["shapley", "banzhaf", "solo", "none"], required=True)
    parser.add_argument("--generations", type=int, default=3)
    parser.add_argument("--out", required=True)
    parser.add_argument("--store", required=True)
    parser.add_argument("--adapters", required=True)
    parser.add_argument("--dev", default=str(ROOT / "data/v4/items/dev.jsonl"))
    parser.add_argument("--test", default=str(ROOT / "data/v4/items/test.jsonl"))
    parser.add_argument("--train-dir", default=str(ROOT / "data/v4/items"))
    parser.add_argument("--protocol", default="v4")
    parser.add_argument("--personas", default=None, help="世代0のペルソナ集合 JSON（{\"personas\": {役割: プロンプト}, \"order\": [...]}）")
    parser.add_argument("--workers", type=int, default=192)
    parser.add_argument("--max-loras", type=int, default=12)
    parser.add_argument("--max-examples", type=int, default=1000)
    parser.add_argument("--lr-first", type=float, default=1e-4)
    parser.add_argument("--lr-next", type=float, default=5e-5)
    parser.add_argument("--share-until", type=int, default=1)
    parser.add_argument("--skip-test", action="store_true")
    parser.add_argument("--external-base-url", default=None, help="結合試験用: 既存サーバを使う")
    parser.add_argument("--fake-train", action="store_true", help="結合試験用: 学習をダミーに置換")
    args = parser.parse_args()
    Runner(args).run()


if __name__ == "__main__":
    main()
