"""ジョブ内で vLLM の OpenAI 互換サーバを起動・停止し、LoRA を動的に登録する。

学習（GPU を占有する）と生成を同じジョブで交互に行うため、サーバはドライバが管理する。
"""

from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path
from typing import Dict, Optional

import httpx

BASE_MODEL = "Qwen/Qwen3-4B-Instruct-2507"


class VllmServer:
    def __init__(self, port: int = 8000, max_model_len: int = 49152, max_loras: int = 12,
                 max_lora_rank: int = 16, max_num_seqs: int = 256, gpu_util: float = 0.90,
                 log_path: str = "/tmp/vllm.log"):
        self.port = port
        self.args = [
            "python3", "-m", "vllm.entrypoints.openai.api_server",
            "--model", BASE_MODEL,
            "--port", str(port),
            "--max-model-len", str(max_model_len),
            "--dtype", "bfloat16",
            "--gpu-memory-utilization", str(gpu_util),
            "--max-num-seqs", str(max_num_seqs),
            "--enable-lora",
            "--max-loras", str(max_loras),
            "--max-lora-rank", str(max_lora_rank),
            "--disable-log-requests",
        ]
        extra = os.environ.get("VLLM_EXTRA_ARGS", "").split()
        self.args += extra
        self.log_path = log_path
        self.proc: Optional[subprocess.Popen] = None
        self.loaded: Dict[str, str] = {}

    @property
    def base_url(self) -> str:
        return f"http://localhost:{self.port}/v1"

    def start(self, timeout: float = 1800.0) -> None:
        env = dict(os.environ, VLLM_ALLOW_RUNTIME_LORA_UPDATING="True")
        log = open(self.log_path, "a")
        self.proc = subprocess.Popen(self.args, stdout=log, stderr=subprocess.STDOUT, env=env)
        started = time.time()
        while time.time() - started < timeout:
            if self.proc.poll() is not None:
                raise RuntimeError(f"vLLM exited early (code {self.proc.returncode}); see {self.log_path}\n"
                                   + Path(self.log_path).read_text()[-4000:])
            try:
                if httpx.get(f"http://localhost:{self.port}/health", timeout=5).status_code == 200:
                    print(f"[server] vLLM healthy after {time.time() - started:.0f}s", flush=True)
                    self.loaded = {}
                    return
            except httpx.HTTPError:
                pass
            time.sleep(5)
        raise RuntimeError("vLLM did not become healthy in time")

    def load_lora(self, name: str, path: str) -> None:
        if self.loaded.get(name) == path:
            return
        response = httpx.post(f"http://localhost:{self.port}/v1/load_lora_adapter",
                              json={"lora_name": name, "lora_path": str(path)}, timeout=300)
        if response.status_code not in (200, 201) and "already" not in response.text.lower():
            raise RuntimeError(f"load_lora {name} failed: {response.status_code} {response.text}")
        self.loaded[name] = path

    def unload_lora(self, name: str) -> None:
        if name not in self.loaded:
            return
        httpx.post(f"http://localhost:{self.port}/v1/unload_lora_adapter", json={"lora_name": name}, timeout=120)
        self.loaded.pop(name, None)

    def stop(self) -> None:
        if self.proc is None:
            return
        self.proc.terminate()
        try:
            self.proc.wait(timeout=120)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            self.proc.wait(timeout=60)
        self.proc = None
        self.loaded = {}
        time.sleep(10)  # GPU メモリの解放待ち
