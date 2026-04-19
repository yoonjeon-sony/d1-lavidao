import json
import os
import sys
from tqdm import tqdm
from pathlib import Path

# Run from repo root so lmms_eval is importable
sys.path.insert(0, str(Path(__file__).parent))

from lmms_eval.tasks.mmvet.utils import mmvet_process_results, mmvet_aggregate_results


def regrade(jsonl_path: str):
    assert os.environ.get("OPENAI_API_KEY"), "set OPENAI_API_KEY first"
    results = []
    with open(jsonl_path) as f:
        for line in f:
            row = json.loads(line)
            doc, resps = row["doc"], row["resps"]
            out = mmvet_process_results(doc, resps)  # routes through OpenAI Responses API
            results.append(out["gpt_eval_score"])

    # write re-graded per-sample scores next to the original
    out_path = jsonl_path.replace(".jsonl", ".regraded.jsonl")
    with open(out_path, "w") as f:
        for r in tqdm(results):
            f.write(json.dumps(r) + "\n")

    overall = mmvet_aggregate_results(results)
    print(f"Overall: {overall:.2f}  (n={len(results)})  wrote {out_path}")


if __name__ == "__main__":
    regrade(sys.argv[1])
