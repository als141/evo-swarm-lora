import argparse
import ast
import json
import statistics
import sys
from pathlib import Path
from subprocess import CompletedProcess, run
from typing import Dict, List

ROOT = Path(__file__).resolve().parent.parent
RUN_DEBATE_SCRIPT = ROOT / "scripts" / "run_debate_local.py"


def parse_vote_block(text: str) -> Dict[str, float]:
    marker = "=== Vote Scores ==="
    idx = text.rfind(marker)
    if idx == -1:
        return {}
    tail = text[idx + len(marker) :].strip().splitlines()
    for line in tail:
        stripped = line.strip()
        if not stripped.startswith("{") or not stripped.endswith("}"):
            continue
        try:
            fixed = stripped.replace("'", '"')
            return json.loads(fixed)
        except json.JSONDecodeError:
            try:
                return ast.literal_eval(stripped)
            except (ValueError, SyntaxError):
                continue
    return {}


def extract_final_answer(text: str) -> str:
    marker = "Final Answer:"
    idx = text.rfind(marker)
    if idx == -1:
        return ""
    answer = text[idx + len(marker) :].strip()
    if answer in {"", "-", "→"}:
        lines = text.strip().splitlines()
        for line in reversed(lines):
            if line.strip() and not line.startswith(("===", "[A", "Final Answer")):
                answer = line.strip()
                break
    return answer


def unique_citations(text: str) -> int:
    import re

    citations = re.findall(r"（[^（）]*?(?:[0-9]{4}|p\.[0-9]+)[^（）]*?）", text)
    return len(set(citations))


def run_debate(adapters: List[str], topic: str, rounds: int) -> Dict[str, object]:
    cmd = [sys.executable, str(RUN_DEBATE_SCRIPT), "--topic", topic, "--rounds", str(rounds), "--adapters", *adapters]
    proc: CompletedProcess[str] = run(cmd, capture_output=True, text=True, cwd=str(ROOT))
    if proc.returncode != 0:
        raise RuntimeError(f"Debate execution failed: {proc.stderr.strip()}")
    output = proc.stdout
    votes = parse_vote_block(output)
    scores = list(votes.values())
    metrics = {
        "topic": topic,
        "rounds": rounds,
        "final_answer": extract_final_answer(output),
        "vote_count": len(scores),
        "avg_vote_score": statistics.mean(scores) if scores else 0.0,
        "unique_citations": unique_citations(output),
        "raw_output": output,
    }
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trio of persona adapters over multiple debate topics.")
    parser.add_argument(
        "--adapters",
        nargs=3,
        required=True,
        help="Paths to persona adapters (persona_a persona_b persona_c).",
    )
    parser.add_argument("--topics", nargs="*", help="List of debate topics.")
    parser.add_argument("--topics-file", help="Path to JSON file containing a list of topics.")
    parser.add_argument("--rounds", type=int, default=3, help="Number of debate rounds per topic.")
    parser.add_argument("--output", required=True, help="Path to write evaluation JSON.")
    parser.add_argument("--label", help="Optional label for this evaluation (e.g., generation name).")
    return parser.parse_args()


def load_topics(args: argparse.Namespace) -> List[str]:
    topics: List[str] = []
    if args.topics:
        topics.extend(args.topics)
    if args.topics_file:
        content = Path(args.topics_file).read_text(encoding="utf-8")
        topics.extend(json.loads(content))
    if not topics:
        raise ValueError("No topics provided. Use --topics or --topics-file.")
    return topics


def main() -> None:
    args = parse_args()
    adapters = [str(Path(p).resolve()) for p in args.adapters]
    topics = load_topics(args)

    results = []
    for topic in topics:
        metrics = run_debate(adapters, topic, args.rounds)
        results.append(metrics)

    avg_vote = statistics.mean(m["avg_vote_score"] for m in results) if results else 0.0
    avg_citations = statistics.mean(m["unique_citations"] for m in results) if results else 0.0
    avg_answer_len = statistics.mean(len(m["final_answer"]) for m in results) if results else 0.0
    fitness = avg_vote + 0.1 * avg_citations + 0.01 * avg_answer_len

    payload = {
        "label": args.label,
        "adapters": adapters,
        "rounds": args.rounds,
        "topics": topics,
        "per_topic": results,
        "summary": {
            "avg_vote_score": avg_vote,
            "avg_unique_citations": avg_citations,
            "avg_final_answer_length": avg_answer_len,
            "fitness": fitness,
        },
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[info] Wrote evaluation to {output_path}")


if __name__ == "__main__":
    main()
