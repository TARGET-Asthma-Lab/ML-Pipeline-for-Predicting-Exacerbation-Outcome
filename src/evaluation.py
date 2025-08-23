# src/evaluation.py
import argparse, json
from pathlib import Path

def main():
    ap = argparse.ArgumentParser(description="Summarize results by tag")
    ap.add_argument("--tag", required=True, help="Tag used during training (prefix ok)")
    ap.add_argument("--results", default="results")
    args = ap.parse_args()

    for p in sorted(Path(args.results).glob(f"metrics_{args.tag}*.json")):
        data = json.loads(Path(p).read_text())
        summ = data.get("summary", {})
        print(f"\n== {p.name} ==")
        for k, v in summ.items():
            print(f"{k}: {v}")
        if "mean_auc_from_roc" in data:
            print(f"mean_auc_from_roc: {data['mean_auc_from_roc']:.4f} ± {data.get('std_auc_from_roc', 0.0):.4f}")

if __name__ == "__main__":
    main()
