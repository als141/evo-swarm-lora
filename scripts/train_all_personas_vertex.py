import argparse
import json
import os
import random
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch


DEFAULT_PERSONAS = ["persona_a", "persona_b", "persona_c"]
PERSONA_DATA_TEMPLATE = "sft_{name}.jsonl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Vertex AI Custom Training entrypoint that sequentially trains persona LoRA adapters, "
            "collects metadata, and writes artifacts/metrics to the directories provided by AIP_* "
            "environment variables."
        )
    )
    parser.add_argument(
        "--personas",
        nargs="+",
        default=DEFAULT_PERSONAS,
        help="Persona identifiers. Each persona expects a dataset file named sft_<persona>.jsonl under --data-dir unless overridden via --datasets.",
    )
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=[],
        help="Override dataset mapping in the form persona=path. Entries that are not provided fall back to --data-dir/sft_<persona>.jsonl.",
    )
    parser.add_argument("--data-dir", default="data", help="Directory that stores persona SFT datasets.")
    parser.add_argument(
        "--output-root",
        default=None,
        help="Optional override for the base output directory. Defaults to AIP_MODEL_DIR or out/model.",
    )
    parser.add_argument(
        "--checkpoint-root",
        default=None,
        help="Optional override for the checkpoint directory. Defaults to AIP_CHECKPOINT_DIR or out/checkpoints.",
    )
    parser.add_argument(
        "--tensorboard-root",
        default=None,
        help="Optional override for the TensorBoard log directory. Defaults to AIP_TENSORBOARD_LOG_DIR or out/logs.",
    )
    parser.add_argument("--train-script", default="scripts/train_lora_persona.py", help="Training script path.")
    parser.add_argument("--epochs", type=int, default=1, help="Number of epochs for each persona.")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate for LoRA training.")
    parser.add_argument("--rank", type=int, default=32, help="LoRA rank (r).")
    parser.add_argument("--grad-accum", type=int, default=16, help="Gradient accumulation steps.")
    parser.add_argument("--bsz", type=int, default=1, help="Per-device batch size.")
    parser.add_argument("--seed", type=int, default=1234, help="Global random seed for reproducibility.")
    parser.add_argument(
        "--experiment",
        default=None,
        help="Optional Vertex AI Experiment name. When provided, parameters/metrics are logged if the SDK is available.",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="Optional Vertex AI Experiment run name. Defaults to persona-training-<timestamp>.",
    )
    parser.add_argument(
        "--project",
        default=None,
        help="Optional GCP project override for Vertex Experiments. Falls back to CLOUD_ML_PROJECT_ID.",
    )
    parser.add_argument(
        "--location",
        default=None,
        help="Optional GCP region override for Vertex Experiments. Falls back to CLOUD_ML_REGION.",
    )
    parser.add_argument(
        "--notes",
        default=None,
        help="Free-form notes stored in metadata.json for traceability (e.g., experiment rationale).",
    )
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cuda"):
        torch.backends.cuda.matmul.allow_tf32 = True
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = False


def _parse_dataset_overrides(overrides: Iterable[str]) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Invalid dataset override '{item}'. Expected format persona=path.")
        persona, path = item.split("=", 1)
        persona = persona.strip()
        if not persona:
            raise ValueError(f"Invalid dataset override '{item}'. Persona name is empty.")
        mapping[persona] = path.strip()
    return mapping


def _resolve_artifact_root(value: str | None, fallback: str, subdirectory: str) -> Tuple[Path, str]:
    uri = value or fallback
    if uri.startswith("gs://"):
        bucket_path = uri[5:]
        local_path = Path("/gcs") / bucket_path
    else:
        local_path = Path(uri)
    target = local_path / subdirectory if subdirectory else local_path
    target.mkdir(parents=True, exist_ok=True)
    return target, uri


def _dataset_path(persona: str, data_dir: Path, overrides: Dict[str, str]) -> Path:
    if persona in overrides:
        return Path(overrides[persona])
    filename = PERSONA_DATA_TEMPLATE.format(name=persona)
    return data_dir / filename


def _load_package_versions() -> Dict[str, str]:
    versions: Dict[str, str] = {
        "python": sys.version,
        "torch": torch.__version__,
    }
    try:
        import transformers

        versions["transformers"] = transformers.__version__
    except Exception:
        versions["transformers"] = "unavailable"
    try:
        import peft

        versions["peft"] = peft.__version__
    except Exception:
        versions["peft"] = "unavailable"
    try:
        import bitsandbytes as bnb

        versions["bitsandbytes"] = getattr(bnb, "__version__", "unavailable")
    except Exception:
        versions["bitsandbytes"] = "unavailable"
    return versions


def maybe_start_experiment(args: argparse.Namespace):
    if not args.experiment:
        return None

    project = args.project or os.environ.get("CLOUD_ML_PROJECT_ID")
    location = args.location or os.environ.get("CLOUD_ML_REGION")
    if not project or not location:
        print(
            "[warn] Vertex Experiment logging disabled: specify --project/--location or set CLOUD_ML_PROJECT_ID / CLOUD_ML_REGION."
        )
        return None

    try:
        from google.cloud import aiplatform
    except ImportError:
        print("[warn] Vertex Experiment logging disabled: google-cloud-aiplatform is not available.")
        return None

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    run_name = args.run_name or f"persona-training-{timestamp}"
    aiplatform.init(project=project, location=location)
    return aiplatform.start_run(run=run_name, experiment=args.experiment)


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)

    dataset_overrides = _parse_dataset_overrides(args.datasets)
    data_root = Path(args.data_dir)

    model_root, model_uri = _resolve_artifact_root(
        args.output_root or os.environ.get("AIP_MODEL_DIR"), fallback="out/model", subdirectory=""
    )
    checkpoint_root, checkpoint_uri = _resolve_artifact_root(
        args.checkpoint_root or os.environ.get("AIP_CHECKPOINT_DIR"), fallback="out/checkpoints", subdirectory=""
    )
    tensorboard_root, tensorboard_uri = _resolve_artifact_root(
        args.tensorboard_root or os.environ.get("AIP_TENSORBOARD_LOG_DIR"), fallback="out/logs", subdirectory=""
    )

    adapters_root = model_root / "adapters"
    configs_root = model_root / "configs"
    eval_root = model_root / "eval"
    debates_root = model_root / "debates"
    for path in (adapters_root, configs_root, eval_root, debates_root, checkpoint_root, tensorboard_root):
        path.mkdir(parents=True, exist_ok=True)

    run_context = maybe_start_experiment(args)
    if run_context is not None:
        params = {
            "personas": ",".join(args.personas),
            "lora_rank": args.rank,
            "learning_rate": args.lr,
            "epochs": args.epochs,
            "batch_size": args.bsz,
            "grad_accum": args.grad_accum,
            "seed": args.seed,
        }
        run_context.log_params(params)

    personas_metadata: List[Dict] = []

    for persona in args.personas:
        dataset_path = _dataset_path(persona, data_root, dataset_overrides)
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset for persona '{persona}' was not found at {dataset_path}")

        adapter_path = adapters_root / persona
        adapter_path.mkdir(parents=True, exist_ok=True)

        cmd = [
            sys.executable,
            args.train_script,
            "--data",
            str(dataset_path),
            "--out",
            str(adapter_path),
            "--r",
            str(args.rank),
            "--epochs",
            str(args.epochs),
            "--lr",
            str(args.lr),
            "--bsz",
            str(args.bsz),
            "--grad_accum",
            str(args.grad_accum),
        ]

        print(f"[info] Training persona '{persona}' with command: {' '.join(cmd)}")
        start_time = time.time()
        subprocess.run(cmd, check=True)
        end_time = time.time()
        duration = end_time - start_time

        persona_record = {
            "persona": persona,
            "dataset": str(dataset_path),
            "adapter_dir": str(adapter_path),
            "start_time_iso": datetime.fromtimestamp(start_time, tz=timezone.utc).isoformat(),
            "end_time_iso": datetime.fromtimestamp(end_time, tz=timezone.utc).isoformat(),
            "duration_seconds": duration,
        }
        personas_metadata.append(persona_record)

        if run_context is not None:
            run_context.log_metrics({f"{persona}_train_seconds": duration})

    metadata = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "seed": args.seed,
        "notes": args.notes,
        "personas": personas_metadata,
        "paths": {
            "model_root": str(model_root),
            "model_uri": model_uri,
            "checkpoint_root": str(checkpoint_root),
            "checkpoint_uri": checkpoint_uri,
            "tensorboard_root": str(tensorboard_root),
            "tensorboard_uri": tensorboard_uri,
        },
        "dependencies": _load_package_versions(),
        "training_hyperparameters": {
            "rank": args.rank,
            "learning_rate": args.lr,
            "epochs": args.epochs,
            "batch_size": args.bsz,
            "grad_accum": args.grad_accum,
        },
    }

    metadata_path = configs_root / "metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, ensure_ascii=False)
    print(f"[info] Wrote metadata to {metadata_path}")

    uv_lock = Path("uv.lock")
    if uv_lock.exists():
        target_lock = configs_root / "uv.lock"
        target_lock.write_bytes(uv_lock.read_bytes())
        print(f"[info] Persisted dependency lockfile to {target_lock}")

    if run_context is not None:
        run_context.log_metrics({"total_training_seconds": sum(p["duration_seconds"] for p in personas_metadata)})
        run_context.end_run()


if __name__ == "__main__":
    main()
