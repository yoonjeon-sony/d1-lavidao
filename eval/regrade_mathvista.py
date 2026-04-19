import copy
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

from tqdm import tqdm

# Run from repo root so lmms_eval is importable
sys.path.insert(0, str(Path(__file__).parent))

from lmms_eval.tasks.mathvista.utils import (
    mathvista_aggregate_results,
    mathvista_process_results,
)


def regrade(jsonl_path: str):
    # GPT extraction only runs when the heuristic path can't parse the answer.
    # Still require the key in case a batch falls through to GPT fallback.
    assert os.environ.get("OPENAI_API_KEY"), "set OPENAI_API_KEY first"

    results = []
    with open(jsonl_path) as f:
        for line in tqdm(list(f), desc="extract+normalize"):
            row = json.loads(line)
            doc, resps = row["doc"], row["resps"]
            out = mathvista_process_results(doc, resps)
            results.append(out["gpt_eval_score"])

    out_path = jsonl_path.replace(".jsonl", ".regraded.jsonl")
    with open(out_path, "w") as f:
        # copy because aggregate mutates `metadata` in-place
        for r in results:
            f.write(json.dumps(r) + "\n")

    args = SimpleNamespace(output_path=str(Path(jsonl_path).parent))
    accuracy = mathvista_aggregate_results(copy.deepcopy(results), args)
    if accuracy is None:
        print(f"Overall accuracy: 0 (aggregate returned None)  n={len(results)}  wrote {out_path}")
    else:
        print(f"Overall accuracy: {accuracy:.2f}%  (n={len(results)})  wrote {out_path}")


if __name__ == "__main__":
    regrade(sys.argv[1])
