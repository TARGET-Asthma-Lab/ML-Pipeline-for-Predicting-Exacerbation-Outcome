# src/training.py
import argparse, subprocess, sys, shutil

def main():
    ap = argparse.ArgumentParser(description="Training entry point (routes to feedforward.py)")
    ap.add_argument("--data", nargs="+", required=True, help="CSV(s) or a single pattern with {i}")
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--target", default="Exacerbation.Outcome")
    ap.add_argument("--id-col", default="subject_id")
    ap.add_argument("--top-k", type=int, default=60)
    ap.add_argument("--folds", type=int, default=10)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--lr", type=float, default=7e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-2)
    ap.add_argument("--dropout", type=float, default=0.7)
    ap.add_argument("--tag", default="run")
    ap.add_argument("--results", default="results")
    ap.add_argument("--figures", default="figures")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--alpha", nargs="+")
    ap.add_argument("--beta", nargs="+")
    ap.add_argument("--raw-div")
    ap.add_argument("--raw-div-pca-var", type=float, default=None)
    args, unknown = ap.parse_known_args()

    py = sys.executable or shutil.which("python") or "python"
    cmd = [py, "src/feedforward.py",
           "--data", *args.data,
           "--runs", str(args.runs),
           "--target", args.target,
           "--id-col", args.id_col,
           "--top-k", str(args.top_k),
           "--folds", str(args.folds),
           "--epochs", str(args.epochs),
           "--lr", str(args.lr),
           "--weight-decay", str(args.weight_decay),
           "--dropout", str(args.dropout),
           "--tag", args.tag,
           "--results", args.results,
           "--figures", args.figures,
           "--seed", str(args.seed)]
    if args.alpha: cmd += ["--alpha", *args.alpha]
    if args.beta:  cmd += ["--beta", *args.beta]
    if args.raw_div: cmd += ["--raw-div", args.raw_div]
    if args.raw_div_pca_var is not None: cmd += ["--raw-div-pca-var", str(args.raw_div_pca_var)]
    cmd += unknown
    subprocess.check_call(cmd)

if __name__ == "__main__":
    main()
