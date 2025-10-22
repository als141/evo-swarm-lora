以下は、**Qwen3‑4B** × **LoRA** × **3エージェント**で「協調実行→協調的最適化（進化）」までを**ローカルで再現・実験**できる、**研究用コードベース一式（Markdown でコピペ可）**です。

* 依存は **uv + Docker** を前提（Ubuntu + A6000 1枚でOK）。
* まずは **最小構成**（3人格LoRAを学習→Multi‑Agent Debate→適応度でLoRAを選抜・合成）で確実に回せる実装にしてあります。
* その後、**vLLMのMulti‑LoRA**や**LangGraph v1系**への拡張、**評価ハーネス（lm‑evaluation‑harness / lighteval）**まで含めた**完全研究手順**を添えます。
* さらに、ご提示の**gpt‑4.1‑mini 協調実行PoC**（OpenAI API）を**学術的に批判的検討**し、この実装方針にどう接続するかも最初に整理します。

---

## 0) 事前の批判的評価：既存PoC（gpt‑4.1‑mini協調実行のみ）をどう活かすか

**要旨**
いただいたPoCは、LangGraphでの非同期マルチエージェント議論・可視化・MMLU簡易評価までを素早く実装しています。ただし**「協調実行のみ」**の概念実証で、**協調的最適化（LoRAの進化・選抜・交配）**は未実装です。結果ログには**結論ノードのimport不整合**による停止が残っており（`cannot import name 'conclusion_node_streaming'`）、**再現性ある定量評価**までは到達していません。

**具体的所見（抜粋）**

* **エラー停止**: MMLU実行時の詳細ログに、`src.score.graph` から `conclusion_node_streaming` を import できず停止した記録。これにより、**議論は0ターン**で終了し、ベンチマークも無効化。
* **OpenAI埋め込み依存**: 収束度計算などで `OpenAIEmbeddings` を直接呼ぶ設計（ローカルOSS化には不向き）。
* **pyprojectのバージョン**: `langgraph>=0.4.8` と比較的古い指定。**LangGraph v1** 安定化（2025年10月）が進んでおり、最新系へ移行推奨。
* **単純ベースラインは有用**: PoC内の**シンプルMMLUスコアラー**はベースライン比較に有用（Structured Outputs で選択肢抽出）。ただしAPI依存。

**結論**
PoCの**「協調実行（議論）」アーキテクチャは活かしつつ**、本研究の中核である**LoRAエージェント群の進化（選抜・交配・突然変異）**を**ローカルOSS**で確実に回す設計へ**全面置換**するのが最短です。具体的には：

1. **Qwen3‑4B‑Instruct**（2025/07版, 256K長文対応）でローカル実行。
2. **PEFT(QLoRA) + TRL**で**3つの人格LoRA**を軽量学習（A6000×1で可）。
3. **Multi‑Agent Debate**は**Transformersローカル推論**でまず再現。次に**vLLMのMulti‑LoRA**へ拡張（同一ベース上でLoRAを差し替え並列化）。
4. **協調的最適化**は、**適応度 = 任务性能 + 協調寄与 + 新規性**で選抜し、**LoRAブレンド（交配）**と**部分再初期化（突然変異）**で次世代を生成。関連文献として**S‑LoRA/マルチLoRA提供系**やLoRA圧縮系、マルチエージェント議論の最新動向を参照。

---

## 1) 主要コンポーネントの技術選定（2025最新版に準拠）

* **ベースモデル**: Qwen3‑4B‑Instruct‑2507（Apache‑2.0, 256K, 最新テンプレ対応）。Transformersは**4.51以上**を要求。**vLLM>=0.8.5** or **SGLang>=0.4.6.post1**での提供が公式推奨。
* **推論(ローカル)**: まずは `transformers` + `peft` で**単一プロセス切替**（VRAM節約）。次に**vLLM Multi‑LoRA**で**同時多LoRA**を一括提供（`enable_lora`, `max_loras`, `max_lora_rank` など）。
* **学習(QLoRA)**: `transformers`(≥4.51) + `peft`(≥0.17) + `trl`(≥0.15) + `bitsandbytes`(≥0.48)。BF16対応の**PyTorch 2.6**を採用。
* **高速化**: 可能なら **FlashAttention‑3**（Ampere対応, 2.x/3.x）。ただしビルドが重いので任意。
* **評価**: `lm‑evaluation‑harness` か `lighteval`。OpenAI互換API（vLLM/TGI）経由評価も可。
* **LangGraph**: v1系列（2025年10月）。段階的に移行（現段階では自前オーケストレーションで最小構成を先に確実化）。
* **uv**: 公式ドキュメント通り使用（Python管理/lock/venv/ツール実行が高速）。

---

## 2) 研究プロトコル（最小実装での完全手順）

### 実験要件

* **1枚GPU(A6000, 48GB)**。QLoRA 4bit + rank低め（例 r=16～32）で十分収まります（VRAM ≒ 数GB + KV）。bitsandbytesは**CUDA 11.0〜12.5**対応。Dockerは**CUDA 12.4**ベース推奨。

### 研究ステップ（反復可能な最小構成）

1. **3人格データの準備**（小規模SFTデータ。公開人格種データや少量自作でOK。将来はNemotron‑Personas‑Japan等で拡張）
2. **Qwen3‑4BへQLoRAで3人格LoRAを学習**（SFTTrainer）。
3. **3エージェント・ディベート**（各ターンで**LoRA切替**して同一ベースに人格注入→発話→批判/支持→最終結論投票）。
4. **適応度計算**：

   * **タスク性能**（正答/ROUGE/F1など）、
   * **協調寄与**（他者の精度をどれだけ押し上げるか＝近似シャプレー：LOO増分）、
   * **新規性**（他2者の生成分布とのKLやエンベ間距離に基づく多様性）
     を正規化加重合成（重みは設定ファイル化）。
5. **協調的最適化**：

   * **選抜**（上位2つ）
   * **交配**（LoRA重みのαブレンド）
   * **突然変異**（一部層r再初期化/ドロップ）
     で**次世代LoRA**を生成。
     vLLMを使う場合は**Multi‑LoRA**の上で**`:<adapter_name>`**で切替・同時提供可能。
6. **評価**：`lm‑evaluation‑harness` / `lighteval` で**ベース/個別LoRA/アンサンブル**比較。

> **Multi‑Agent Debate の文献的裏付け**（MAD/DoT回避など）と、**進化的選択**（レプリケータ動力学/交配・圧縮LoRA）に関する近年の議論は以下を参照。

---

# 3) コードベース（丸ごとコピペで動かせる最小構成）

> まずは **ローカルTransformers/PEFT** の最小構成で動かし、**LoRA三人格→協調→選抜・交配**までを完了します。
> 次に vLLM への拡張（Multi‑LoRA 常駐）を使って**高速推論**へ移行します。

以下を**リポジトリ直下**に保存してください。

---

### `pyproject.toml`（uv 用・研究最小構成）

```toml
[project]
name = "evo-swarm-lora-agents"
version = "0.1.0"
description = "Evolutionary swarm intelligence for cooperative optimization of LoRA-LLM agent populations (Qwen3-4B)."
readme = "README.md"
requires-python = ">=3.12"

dependencies = [
  "torch>=2.6.0",                # BF16 / SDPA
  "transformers>=4.51.0",        # Qwen3 support (>=4.51)
  "accelerate>=1.0.0",
  "peft>=0.17.1",
  "bitsandbytes>=0.48.1",        # CUDA 11.0 - 12.5
  "trl>=0.15.0",                 # SFTTrainer 等
  "datasets>=3.0.0",
  "safetensors>=0.4.5",
  "huggingface-hub>=0.24.7",
  "evaluate>=0.4.2",
  "scikit-learn>=1.5.0",
  "numpy>=2.1.0",
  "pydantic>=2.8.0",
  "typer>=0.12.3",
  "tqdm>=4.66.4",
  "rich>=13.7.1",
  "jinja2>=3.1.4",
  "uvicorn>=0.30.0",
  "fastapi>=0.114.0",
]

[project.optional-dependencies]
dev = [
  "ruff>=0.6.9",
  "pytest>=8.3.1",
  "pytest-cov>=5.0.0",
  "ipykernel>=6.29.5"
]

[tool.uv]
# uv will create .venv by default on install
default-groups = ["dev"]

[tool.ruff]
line-length = 100
target-version = "py312"
```

**根拠**: Qwen3‑4B‑InstructはTransformers 4.51+が必要、推奨推論サーバは vLLM≥0.8.5 / SGLang≥0.4.6.post1。
bitsandbytes は CUDA 11.0–12.5対応。
PyTorch 2.6 GA（2025/01）。

---

### `Dockerfile`（GPU・本番計算用）

```dockerfile
# CUDA 12.4 + Ubuntu 22.04
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# OS deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl wget ca-certificates build-essential python3 python3-venv python3-pip \
    && rm -rf /var/lib/apt/lists/*

# uv install (official)
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"

WORKDIR /workspace
COPY pyproject.toml README.md ./

# Create venv & install deps with uv
RUN uv venv && . .venv/bin/activate && uv sync --no-dev

# Cache Hugging Face models (optional)
ENV HF_HOME=/workspace/.cache/huggingface
RUN mkdir -p $HF_HOME

# Add source
COPY src ./src
COPY data ./data
COPY scripts ./scripts

ENV PYTHONPATH=/workspace
CMD ["/bin/bash"]
```

**uvの公式インストール手順準拠**。

---

### `docker-compose.yml`（任意：vLLMを別サービスで）

```yaml
services:
  trainer:
    build: .
    image: evo-swarm-lora-agents:latest
    container_name: evo-trainer
    tty: true
    stdin_open: true
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
      - PYTHONUNBUFFERED=1
    volumes:
      - ./:/workspace
      - hf-cache:/workspace/.cache/huggingface
    deploy:
      resources:
        reservations:
          devices:
            - capabilities: [gpu]
    command: bash

  vllm:
    image: vllm/vllm-openai:latest
    container_name: vllm-qwen3
    ports: ["8000:8000"]
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
    volumes:
      - hf-cache:/root/.cache/huggingface
      - ./adapters:/adapters  # LoRA格納
    deploy:
      resources:
        reservations:
          devices:
            - capabilities: [gpu]
    command: >
      bash -lc "vllm serve Qwen/Qwen3-4B-Instruct-2507
      --port 8000
      --max-model-len 32768
      --tensor-parallel-size 1
      --enable-lora
      --max-loras 8
      --max-lora-rank 64
      --lora-modules persona_a:/adapters/persona_a persona_b:/adapters/persona_b persona_c:/adapters/persona_c"

volumes:
  hf-cache:
```

> vLLM の Multi‑LoRA では `--enable_lora`, `--max_loras`, `--max_lora_rank` 等を使います（Ray/Anyscale docs にも同様の指定が記載）。**OpenAI互換**で `model="Qwen/Qwen3-4B-Instruct-2507:persona_a"` の形式で利用可能。

---

### ディレクトリ構成

```
.
├─ README.md
├─ pyproject.toml
├─ Dockerfile
├─ docker-compose.yml
├─ data/
│  ├─ sft_persona_a.jsonl
│  ├─ sft_persona_b.jsonl
│  └─ sft_persona_c.jsonl
├─ adapters/          # 学習済みLoRAがここに生成される
│  ├─ persona_a/
│  │  ├─ adapter_config.json
│  │  └─ adapter_model.safetensors
│  ├─ persona_b/
│  └─ persona_c/
├─ scripts/
│  ├─ train_lora_persona.py
│  ├─ run_debate_local.py
│  ├─ evolve_loras.py
│  ├─ eval_with_harness.sh
│  └─ export_merge_utils.py
└─ src/
   ├─ agents/
   │  ├─ personalities.py
   │  └─ debate.py
   ├─ models/
   │  ├─ qwen_loader.py
   │  └─ lora_ops.py
   ├─ coop/
   │  ├─ fitness.py
   │  └─ voting.py
   └─ utils/
      ├─ prompts.py
      └─ io.py
```

---

### `data/sft_persona_*.jsonl`（最小SFTデータの雛形）

```jsonl
{"messages":[
  {"role":"system","content":"あなたは厳密な論理検証を重視する批判的思考の専門家です。"},
  {"role":"user","content":"ソクラテスは人間である。人間は死ぬ。ソクラテスは死ぬか？"},
  {"role":"assistant","content":"はい。三段論法により結論は「ソクラテスは死ぬ」です。"}
]}
{"messages":[
  {"role":"system","content":"あなたは厳密な論理検証を重視する批判的思考の専門家です。"},
  {"role":"user","content":"次の主張の前提の妥当性を検証してください: ..."},
  {"role":"assistant","content":"前提Aは観察データXにより支持されますが、Bは..."}
]}
```

> まずは**各人格50〜200対話**程度の最小SFTで可。将来は**Nemotron Personas‑Japan**, **OpenCharacter/PersonaHub**等で拡張し、日本語に最適化。

---

### `src/models/qwen_loader.py`

```python
from typing import Optional, List
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig

QWEN_NAME = "Qwen/Qwen3-4B-Instruct-2507"

def load_base(device_map: str = "auto", load_in_4bit: bool = True):
    tok = AutoTokenizer.from_pretrained(QWEN_NAME, trust_remote_code=True)
    kwargs = dict(
        torch_dtype=torch.bfloat16,
        device_map=device_map,
    )
    if load_in_4bit:
        kwargs.update(dict(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        ))
    model = AutoModelForCausalLM.from_pretrained(QWEN_NAME, **kwargs)
    model.config.use_cache = True
    return model, tok

def attach_lora(model, adapter_dir: str):
    # adapters/persona_x 内の adapter_config.json / adapter_model.safetensors を読む
    peft_cfg = PeftConfig.from_pretrained(adapter_dir)
    peft_model = PeftModel.from_pretrained(model, adapter_dir, is_trainable=False)
    peft_model.set_adapter(peft_cfg.peft_type)
    return peft_model

def generate(model, tok, messages: List[dict], max_new_tokens=512, temperature=0.7, top_p=0.9):
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tok([text], return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=max_new_tokens, temperature=temperature,
            do_sample=True, top_p=top_p, pad_token_id=tok.eos_token_id
        )
    gen_ids = out[0][len(inputs.input_ids[0]):]
    return tok.decode(gen_ids, skip_special_tokens=True)
```

> Qwen3の**Quickstart**はTokenizerの `apply_chat_template` を推奨。**Transformers 4.51+必須**。

---

### `scripts/train_lora_persona.py`（QLoRA学習）

```python
import os, argparse
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig
import torch

BASE = "Qwen/Qwen3-4B-Instruct-2507"

def load_sft_dataset(path):
    # JSONL {messages:[...]} 形式 → ChatTemplateでトークナイズ
    # datasetsのjsonl直接読み込み
    ds = load_dataset("json", data_files=path, split="train")
    return ds

def format_chat(example, tokenizer):
    # そのままmessagesを渡してテンプレ化
    text = tokenizer.apply_chat_template(example["messages"], tokenize=False, add_generation_prompt=False)
    return {"text": text}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)  # adapters/persona_x
    ap.add_argument("--r", type=int, default=32)
    ap.add_argument("--target", nargs="+", default=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"])
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--bsz", type=int, default=1)
    ap.add_argument("--grad_accum", type=int, default=16)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(BASE, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        BASE,
        torch_dtype=torch.bfloat16,
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        device_map="auto"
    )

    ds = load_sft_dataset(args.data)
    ds = ds.map(lambda ex: format_chat(ex, tok), remove_columns=ds.column_names)

    peft_cfg = LoraConfig(
        r=args.r, lora_alpha=args.r*2, lora_dropout=0.05,
        target_modules=args.target, bias="none", task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, peft_cfg)

    train_cfg = SFTConfig(
        output_dir=args.out,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.bsz,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        logging_steps=10,
        save_steps=200,
        bf16=True,
        optim="paged_adamw_32bit",
        max_seq_length=2048,
        gradient_checkpointing=True,
        packing=False
    )

    trainer = SFTTrainer(
        model=model,
        args=train_cfg,
        train_dataset=ds,
        tokenizer=tok,
        dataset_text_field="text",
    )
    trainer.train()
    # LoRAのみ出力 (adapter_config.json / adapter_model.safetensors)
    trainer.model.save_pretrained(args.out)
    tok.save_pretrained(args.out)

if __name__ == "__main__":
    main()
```

> TRLのSFTTrainerで**QLoRA**（4bit）学習。CLI利用も可。

---

### `src/agents/personalities.py`（3人格の定義）

```python
PERSONAS = {
  "persona_a": "あなたは厳密な検証を重視する批判的思考家。反証・例外・境界条件に敏感。",
  "persona_b": "あなたは応用志向の実務家。意思決定に役立つ実装可能性とコストを重視。",
  "persona_c": "あなたは創発を促す発想家。仮説生成と多角的比喩で発想を広げる。"
}
```

---

### `src/utils/prompts.py`（プロトコル）

```python
DEBATE_RULES = """# Debate Protocol
- 各エージェントは短くも具体的に反論/支持を述べる（<= 180 tokens）
- 事実主張には根拠（式・定義・出典）を添える
- 直前発言の論点逸脱を指摘し、論点合意を明示する
- 最終ラウンドでは「自分の最終回答」と「他者への重み」を出す
出力形式:
- 発話: <utterance>
- 要約: <summary>
- 自信: <0-1>
"""
JUDGE_PROMPT = """あなたは議長です。各発話の要点を統合し、重み付き投票で最終回答を導いてください。
- 各エージェントの自信×一貫性×寄与で重みを算出
- 結果は JSON で: {"answer": "...", "rationale": "..."}"""
```

---

### `src/coop/voting.py`（重み付き投票）

```python
from typing import List, Dict
import numpy as np

def soft_vote(candidates: List[Dict]):
    # candidates: [{"agent":"persona_a","answer":"C","confidence":0.7}, ...]
    # 同一選択肢の信頼度合算で多数決
    scores = {}
    for c in candidates:
        scores[c["answer"]] = scores.get(c["answer"], 0.0) + float(c["confidence"])
    answer = max(scores.items(), key=lambda x: x[1])[0]
    return answer, scores
```

---

### `src/coop/fitness.py`（適応度：性能 + 協調寄与 + 新規性）

```python
from typing import Dict, List
import numpy as np

def cooperation_gain(agent_name: str, solo_acc: float, team_acc: float) -> float:
    # 単独精度 → チーム最終精度の増分（近似シャプレー: 1人抜き比較などは将来拡張）
    return max(0.0, team_acc - solo_acc)

def novelty_score(emb_ref: np.ndarray, emb_agent: np.ndarray) -> float:
    # コサイン距離ベースの新規性（他者合成平均ベクトルとの距離）
    num = float(np.dot(emb_ref, emb_agent))
    denom = (np.linalg.norm(emb_ref) * np.linalg.norm(emb_agent) + 1e-8)
    cos = num / denom
    return 1.0 - cos  # 距離

def fitness(solo_acc: float, team_acc: float, novelty: float,
            w_perf=0.6, w_coop=0.3, w_nov=0.1) -> float:
    coop = cooperation_gain("", solo_acc, team_acc)
    return w_perf*solo_acc + w_coop*coop + w_nov*novelty
```

---

### `src/models/lora_ops.py`（LoRA交配・突然変異）

```python
from safetensors.torch import load_file, save_file
import torch, os, json, random
from typing import Dict

def alpha_blend_lora(parent_a_dir: str, parent_b_dir: str, out_dir: str, alpha: float = 0.5):
    os.makedirs(out_dir, exist_ok=True)
    A = load_file(os.path.join(parent_a_dir, "adapter_model.safetensors"))
    B = load_file(os.path.join(parent_b_dir, "adapter_model.safetensors"))
    C = {}
    for k in A.keys():
        if k in B:
            C[k] = (1-alpha)*A[k] + alpha*B[k]
    save_file(C, os.path.join(out_dir, "adapter_model.safetensors"))

    # configはAを継承（rank等は同一前提）
    with open(os.path.join(parent_a_dir, "adapter_config.json")) as f:
        cfg = json.load(f)
    with open(os.path.join(out_dir, "adapter_config.json"), "w") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)

def mutate_lora(in_dir: str, out_dir: str, ratio: float = 0.05, std: float = 0.01):
    os.makedirs(out_dir, exist_ok=True)
    W = load_file(os.path.join(in_dir, "adapter_model.safetensors"))
    M = {}
    for k, t in W.items():
        if random.random() < ratio:
            noise = torch.randn_like(t) * std
            M[k] = t + noise
        else:
            M[k] = t
    save_file(M, os.path.join(out_dir, "adapter_model.safetensors"))
    with open(os.path.join(in_dir, "adapter_config.json")) as f:
        cfg = json.load(f)
    with open(os.path.join(out_dir, "adapter_config.json"), "w") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)
```

> LoRAの**交配（重み線形合成）**と**突然変異（ランダム摂動）**の最小版。

---

### `scripts/run_debate_local.py`（3エージェント議論 → 最終回答）

```python
import argparse
from src.models.qwen_loader import load_base, attach_lora, generate
from src.agents.personalities import PERSONAS
from src.utils.prompts import DEBATE_RULES, JUDGE_PROMPT
from src.coop.voting import soft_vote

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--topic", required=True)
    ap.add_argument("--adapters", nargs=3, default=["adapters/persona_a","adapters/persona_b","adapters/persona_c"])
    ap.add_argument("--rounds", type=int, default=3)
    args = ap.parse_args()

    base, tok = load_base()
    agents = []
    for a in args.adapters:
        agents.append(attach_lora(base, a))

    history = []
    for r in range(args.rounds):
        votes = []
        for i, agent in enumerate(agents):
            persona = list(PERSONAS.values())[i]
            msgs = [{"role":"system","content": persona + "\n" + DEBATE_RULES}]
            if r==0:
                msgs.append({"role":"user","content": args.topic})
            else:
                msgs.append({"role":"user","content": "これまでの議論:\n" + "\n".join(history)})
            out = generate(agent, tok, msgs, max_new_tokens=256)
            history.append(f"[A{i}] {out}")
            votes.append({"agent": f"persona_{i}", "answer": out.strip()[:1], "confidence": 0.6})
        # 上は例；実際はJSON抽出にしてanswer/confidenceをparse推奨
    final, scores = soft_vote(votes)
    print("=== Debate Transcript ===")
    print("\n".join(history))
    print("=== Vote ===", scores)
    print("Final:", final)

if __name__ == "__main__":
    main()
```

※ 簡潔のため**出力整形**は省略。実運用では**JSON抽出**（構造化出力）にしてください。

---

### `scripts/evolve_loras.py`（適応度→選抜・交配・突然変異→次世代）

```python
import argparse, os, shutil
from src.models.lora_ops import alpha_blend_lora, mutate_lora

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parents", nargs=2, required=True)   # adapters/persona_a adapters/persona_b
    ap.add_argument("--child", required=True)              # adapters/persona_d
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--mut", type=float, default=0.05)
    args = ap.parse_args()

    tmp = args.child + "_tmp"
    if os.path.exists(tmp):
        shutil.rmtree(tmp)
    alpha_blend_lora(args.parents[0], args.parents[1], tmp, args.alpha)
    if args.mut > 0:
        mutate_lora(tmp, args.child, ratio=args.mut)
        shutil.rmtree(tmp)
    else:
        os.rename(tmp, args.child)

if __name__ == "__main__":
    main()
```

---

### `scripts/export_merge_utils.py`（LoRA→マージ（任意・検証用））

```python
import argparse, torch, os
from transformers import AutoModelForCausalLM
from peft import PeftModel

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="Qwen/Qwen3-4B-Instruct-2507")
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    base = AutoModelForCausalLM.from_pretrained(args.base, torch_dtype=torch.bfloat16, device_map="cpu")
    peft = PeftModel.from_pretrained(base, args.adapter)
    merged = peft.merge_and_unload()
    merged.save_pretrained(args.out)

if __name__ == "__main__":
    main()
```

---

### `scripts/eval_with_harness.sh`（評価ハーネス例：OpenAI互換API経由）

```bash
#!/usr/bin/env bash
# 例：vLLM起動済み（localhost:8000）で Qwen3 + LoRAをOpenAI互換で叩く
# EleutherAI lm-evaluation-harness / lighteval いずれも可
# 下はHarnessの例（2025時点での課題/Issue多数のため、最新README参照）
# https://github.com/EleutherAI/lm-evaluation-harness
set -e
export HF_ALLOW_CODE_EVAL=1

TASKS="arc_easy,winogrande,hellaswag"
MODEL="openai-chat-completions"
ARGS="model=Qwen/Qwen3-4B-Instruct-2507:persona_a,base_url=http://localhost:8000/v1/chat/completions"

python -m lm_eval --model ${MODEL} --model_args ${ARGS} \
  --tasks ${TASKS} --batch_size auto --apply_chat_template --output_path out/persona_a_eval.json
```

> Harness / LightEval は最新版の問題修正も追うこと。Issueも活発（2025）。

---

## 4) 実行手順（uv / Docker）

### ローカル（uv のみ）

```bash
# 依存同期
uv sync
# 3人格学習（最小SFT）
uv run python scripts/train_lora_persona.py --data data/sft_persona_a.jsonl --out adapters/persona_a
uv run python scripts/train_lora_persona.py --data data/sft_persona_b.jsonl --out adapters/persona_b
uv run python scripts/train_lora_persona.py --data data/sft_persona_c.jsonl --out adapters/persona_c
# 協調実行（ローカルTransformers）
uv run python scripts/run_debate_local.py --topic "消費税減税は日本経済にプラスか？"
# 適応度評価は別途（notebook or 評価スクリプト）、選抜後に交配
uv run python scripts/evolve_loras.py --parents adapters/persona_a adapters/persona_b --child adapters/persona_d --alpha 0.6 --mut 0.03
```

### Docker + vLLM（OpenAI互換API）

```bash
docker compose up -d --build
# vLLM が adapters/* を Multi-LoRAで提供
# 評価ハーネスを叩く
docker exec -it vllm-qwen3 bash -lc 'python - <<PY
from openai import OpenAI
c=OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")
r=c.chat.completions.create(model="Qwen/Qwen3-4B-Instruct-2507:persona_a",
    messages=[{"role":"user","content":"次の数列の一般項は？ 2,4,8,16, ..."}], temperature=0.2)
print(r.choices[0].message)
PY'
```

> vLLMのMulti‑LoRA 配備は、Ray/Anyscaleのドキュメントが詳しい（`enable_lora`, `max_loras`, `max_lora_rank`, `dynamic_lora_loading_path` 等）。

---

## 5) codex cli 用プロンプトパック（コピペ投入）

> 「**codex cli**」に貼るだけで、段階的にファイル生成・修正できる**タスク駆動プロンプト**を用意しました。
> `prompts/codex/*.md` として保存し、順に投入してください。

### `prompts/codex/00_project_bootstrap.md`

```
# GOAL
- Qwen3-4B + LoRA(QLoRA) + 3人格のマルチエージェント協調→協調的最適化の最小研究実装。
- uv / Docker 前提。A6000 1枚で実験可能。

# ACTIONS
1) 付与された pyproject.toml / Dockerfile / docker-compose.yml をベースに、venv同期とビルドを通す。
2) src/*, scripts/*, data/* を生成（不足があれば質問せず合理的に補完）。
3) run_debate_local.py が単独で動くことをCI風に検証用スクリプトを作る。

# ACCEPTANCE
- `uv sync` が成功。
- `python scripts/run_debate_local.py --topic "テスト"` がエラーなく終わる。
```

### `prompts/codex/01_train_three_loras.md`

```
# GOAL
- data/sft_persona_{a,b,c}.jsonl を読み込み、QLoRAで3人格LoRAを adapters/ に書き出す。

# ACTIONS
- scripts/train_lora_persona.py の引数やデフォルトを調整し、短時間で学習が終わるようにする（データが少量のため）。
- 学習ログとLoRAサイズを標準出力に表示。

# ACCEPTANCE
- adapters/persona_a|b|c に adapter_config.json / adapter_model.safetensors が生成。
- 3人格で run_debate_local.py が動作。
```

### `prompts/codex/02_multilora_vllm.md`

```
# GOAL
- docker-compose の vLLM サービスで Multi-LoRA を提供し、OpenAI互換API経由で人格切替を確認。

# ACTIONS
- 起動後の疎通スクリプトを scripts に追加（OpenAI SDK で :persona_x を付与して質問→応答）。
- harness用シェル eval_with_harness.sh を整備。

# ACCEPTANCE
- persona_a/b/c で応答が差別化される。
```

### `prompts/codex/03_evolutionary_loop.md`

```
# GOAL
- 3人格の適応度を評価→上位2つを選抜→LoRA交配（alpha_blend）→突然変異→評価→ログ保存のループを構築。

# ACTIONS
- src/coop/fitness.py を用い、タスク性能(正答率or Rouge等)、協調寄与(LOO増分近似)、新規性(埋め込み距離)を合成したfitnessを実装。
- scripts/evolve_loras.py を拡張し、世代ごとに adapters/gen_{t}/... を生成。

# ACCEPTANCE
- N世代回せる（最小2世代）。各世代のfitnessログと最良LoRAの推移が保存。
```

### `prompts/codex/04_langgraph_integration.md`

```
# GOAL
- 現行の自前オーケストレーションを LangGraph v1 に移行（将来、配信や分岐を強化）。

# ACTIONS
- v1の安定APIで、debateノード群（agent→metrics→judge）を状態遷移で表現。
- Qwenローカル推論ラッパーをLangGraphのコール可能に。

# ACCEPTANCE
- 既存run_debate_local.py と同等の結果が LangGraph 実装でも再現。
```

---

## 6) 研究評価・再現性・論文化のためのポイント

* **再現性セット**：

  1. データ（SFT + テスト）版管理、
  2. 乱数固定、
  3. 依存ロック（uv.lock）、
  4. 評価スクリプト（ハーネス出力の JSON 保存）
     を必ずリポジトリに含めます。

* **ベースライン**：

  * 単独ベース（Qwen3‑4B無改変）、
  * 単体LoRA（人格別）、
  * 協調（Multi‑Agent Debate）、
  * 進化世代（Gen‑t）
    を**同一ベンチで比較**。

* **メトリクス**：

  * **性能**：MC系は Accuracy、記述は Rouge‑L / BLEU / F1 を採用。
  * **協調寄与**：LOO近似で**各エージェントの限界貢献**を測定（擬シャプレー）。
  * **新規性**：生成埋め込みの**他者平均からの距離**。
  * **議論効率**：ターン数/到達確率/判定時間。

* **関連最新**

  * Qwen3‑4B‑Instruct‑2507（Transformers 4.51+ / vLLM≥0.8.5 / SGLang≥0.4.6.post1 推奨）。
  * **vLLM Multi‑LoRA**（`enable_lora`, `max_loras`, `max_lora_rank`・LRU管理）。
  * **PyTorch 2.6 GA**（2025/01）。
  * **bitsandbytes**（CUDA 11.0–12.5対応）。
  * **LangGraph v1** 安定化・移行ガイド（2025/10）。
  * **MAD / Degeneration‑of‑Thought** 議論手法、**LoRAマルチサーブ/圧縮**の最新。

---

## 7) A6000×1 での実行性について

* **Qwen3‑4B** + **QLoRA(4bit)** + **rank 16–32** で、**1枚48GB**なら**学習・推論とも十分**です（bf16計算 + 4bit重み + 勾配チェックポイント）。
* 4bitの**bitsandbytes**は CUDA 12.4 環境でも安定（適合範囲は11.0–12.5）。
* 高速化が必要になれば**FlashAttention‑3**の導入を検討（ビルド前提、任意）。

---

## 8) あなたのPoCとの橋渡し（移行計画）

* 既存PoCの**議論プロンプト設計とストリーミングUI**は流用価値大。ただし、**OpenAI依存のEmbeddingやAPI**は**ローカルQwen3**に置換します。`OpenAIEmbeddings`に依存していた収束度などのメトリクスは、**自前埋め込み**（Qwen3の`hidden_states`平均や軽量埋め込みモデル）に置換。
* `conclusion_node_streaming` import不整合など**グラフ構造のドリフト**があったため、**最小構成**を自前オーケストレーションで安定化→**LangGraph v1 へ段階移行**が安全です。

---

## 9) 参考文献・リンク（主要）

* **Qwen3‑4B‑Instruct‑2507（HFモデルカード）**：Transformers 4.51+ 要求、vLLM ≥0.8.5 / SGLang推奨、256K文脈、エージェント用Qwen‑Agentの案内あり。
* **vLLM Multi‑LoRA**（Ray/Anyscale Docs：`enable_lora`, `max_loras`, `max_lora_rank`, `:<adapter_name>` ルーティング）。
* **PyTorch 2.6 GA**（2025/01）。
* **bitsandbytes** CUDA 11.0–12.5。
* **LangGraph v1**（2025/10）。
* **lm‑evaluation‑harness** / **lighteval**。
* **Multi‑Agent Debate/MAD**：DoT問題と解決策など。
* **LoRA大量同時提供/圧縮**（S‑LoRA など）。

---

### 付録A：研究計画テンプレ（論文化向け章立て）

1. **動機**：巨大LLMの限界（スケーリング則限界・不透明性）と**個から集団**への発想。
2. **関連研究**：MAD/自己反省法/マルチエージェント、LoRAサービング/圧縮、進化的最適化。
3. **手法**：

   * **三段階**：人格LoRA初期化→**協調実行**→**協調的最適化（進化）**。
   * 適応度＝性能＋協調寄与＋新規性、**交配**（LoRAブレンド）＋**突然変異**（摂動）。
4. **実装**：Qwen3‑4B × QLoRA × vLLM Multi‑LoRA。
5. **評価**：ハーネス一式、ベース/個別/協調/世代比較、アブレーション（寄与/新規性重み）。
6. **結果**：少データLoRAで**協調による頑健化**と**進化での持続的改善**を示す。
7. **限界と今後**：人格干渉・通信コスト・LoRAマージの安定性・安全性。
8. **社会的波及**：**社会的合意形成・政策案評価**などコミュニケーション中心タスクでの有用性。

---

## 10) ここからの実行チェックリスト

1. `uv sync` が通る
2. `scripts/train_lora_persona.py` で 3人格LoRA が生成
3. `scripts/run_debate_local.py` が動作（3人格の差異が出る）
4. `docker compose up -d` → vLLM 経由で人格切替が可能
5. `scripts/evolve_loras.py` で新世代LoRAが作られる
6. ハーネスでベース/人格/協調/世代の横並び比較を保存

---

### （参考）既存PoCへの言及の出典

* **import不整合による停止**（`cannot import name 'conclusion_node_streaming'` で議論0ターン）。
* **OpenAI埋め込み依存**（収束度計算）。
* **pyprojectの依存**（langgraph 0.4.x）。
* **シンプルMMLUスコアラー**（Structured Outputs）。

---

この一式で、**ローカル最小構成→vLLM拡張→進化ループ**まで**A6000×1**で完結し、**論文化に耐える再現性**（データ/ロック/評価ログ）を確保できます。必要であれば**LangGraph v1** への段階移行テンプレも含めています。
実装・プロンプトの一部は簡潔化していますが、**codex cli**用タスクを用意しており、そのまま貼り付ければ**不足分は段階的に自動生成**できます。

**主要出典**：Qwen3‑4B‑Instruct‑2507モデルカード（4.51+ / vLLM≥0.8.5 / 256K）、vLLM Multi‑LoRA（Anyscale/Ray Docs）、PyTorch 2.6 GA、bitsandbytes CUDA対応、LangGraph v1、評価ハーネス群、MAD/LoRAサーブ文献。

---

> 以降は、**プロンプト設計**や**評価データ**を徐々に拡充し、**協調寄与**や**新規性**の定義をより厳密化（例：確率的出力分布からの**JSダイバージェンス**、反事実議論を導入）していくのが次の一手です。
