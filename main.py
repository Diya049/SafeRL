"""
main.py — Safe RL Project Entry Point
======================================
Usage:
    python main.py              # train + analyse (skip tuning)
    python main.py --tune       # tune first, then train + analyse
    python main.py --tune-only  # just tune, no final training

Outputs land in  ./outputs/
"""

import argparse, sys

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--tune",      action="store_true",
                   help="Run Optuna tuning before final training")
    p.add_argument("--tune-only", action="store_true",
                   help="Only run Optuna tuning")
    p.add_argument("--trials",    type=int, default=40,
                   help="Optuna trials per method (default 40)")
    return p.parse_args()


def main():
    args = parse_args()

    if args.tune or getattr(args, "tune_only", False):
        print("\n" + "="*60)
        print("  PHASE 1 — Optuna Hyperparameter Tuning")
        print("="*60)
        import tune
        tune.tune(n_trials=args.trials)

    if getattr(args, "tune_only", False):
        print("Tuning complete. Re-run without --tune-only to train.")
        sys.exit(0)

    print("\n" + "="*60)
    print("  PHASE 2 — Full Training & Analysis")
    print("="*60)
    import analyze
    analyze.main() if hasattr(analyze, "main") else exec(
        open("analyze.py").read()
    )


if __name__ == "__main__":
    # If run without arguments just do full training + analysis
    import sys
    if len(sys.argv) == 1:
        import analyze
        h1, h2 = analyze.run_training()
        analyze.plot_learning_curves(h1, h2)
        analyze.plot_lambda(h1)
        analyze.pareto_analysis(h1, h2)
        analyze.plot_comparison_bar(h1, h2)
        analyze.save_best_episode_strip(h1, "Lagrangian_RL")
        analyze.save_best_episode_strip(h2, "Safety_Layer")
        analyze.save_best_episode_log(h1, "Lagrangian_RL")
        analyze.save_best_episode_log(h2, "Safety_Layer")
        analyze.print_summary(h1, h2)
        print("\nDone! All outputs in ./outputs/")
    else:
        main()
