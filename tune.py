"""
Hyperparameter Tuning
=====================
Uses Optuna when available; falls back to random search otherwise.
"""

import numpy as np
import json, os, warnings
warnings.filterwarnings("ignore")

import method1_lagrangian   as m1
import method2_safety_layer as m2
from config import LAG_COST_LIMIT

TUNE_EPISODES  = 400
TUNE_MAX_STEPS = 150
N_TRIALS       = 30
VIOL_BUDGET    = 0.15
OUTPUT_DIR     = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    HAS_OPTUNA = True
    print("Optuna found — using TPE sampler")
except ImportError:
    HAS_OPTUNA = False
    print("Optuna not installed — using random search fallback")


def score(avg_r, avg_viol):
    return avg_r - 500 * max(0, avg_viol - VIOL_BUDGET)

def eval_lagrangian(kwargs, seed):
    h, _ = m1.train(num_episodes=TUNE_EPISODES, max_steps=TUNE_MAX_STEPS,
                    agent_kwargs=kwargs, seed=seed)
    r = np.mean(h["reward"][-100:])
    v = np.mean(h["violation_rate"][-100:])
    return score(r, v), r, v

def eval_safety_layer(kwargs, seed):
    h, _ = m2.train(num_episodes=TUNE_EPISODES, max_steps=TUNE_MAX_STEPS,
                    agent_kwargs=kwargs, seed=seed)
    r = np.mean(h["reward"][-100:])
    v = np.mean(h["violation_rate"][-100:])
    return score(r, v), r, v

def _sample_lagrangian(rng):
    return dict(
        alpha      = float(np.exp(rng.uniform(np.log(0.01), np.log(0.5)))),
        gamma      = float(rng.uniform(0.90, 0.999)),
        eps_decay  = float(rng.uniform(0.990, 0.9995)),
        cost_limit = float(rng.uniform(0.01, 0.30)),
        lambda_lr  = float(rng.uniform(0.01, 0.30)),
        lambda_max = float(rng.uniform(1.0,  20.0)),
    )

def _sample_safety_layer(rng):
    return dict(
        alpha      = float(np.exp(rng.uniform(np.log(0.01), np.log(0.5)))),
        gamma      = float(rng.uniform(0.90, 0.999)),
        eps_decay  = float(rng.uniform(0.990, 0.9995)),
        sl_penalty = float(rng.uniform(5.0,  80.0)),
    )

def random_search(name, sampler_fn, eval_fn, n_trials):
    rng = np.random.default_rng(0)
    best_val, best_params = -np.inf, {}
    for i in range(n_trials):
        kw = sampler_fn(rng)
        val, avg_r, avg_v = eval_fn(kw, seed=i)
        marker = "*" if val > best_val else " "
        print(f"  [{marker}] trial {i+1:3d}/{n_trials} | "
              f"score={val:8.2f}  r={avg_r:7.2f}  viol={avg_v:.3f}")
        if val > best_val:
            best_val, best_params = val, kw
    return best_val, best_params

def _optuna_lag(trial):
    kw = dict(
        alpha      = trial.suggest_float("alpha",      0.01, 0.5, log=True),
        gamma      = trial.suggest_float("gamma",      0.90, 0.999),
        eps_decay  = trial.suggest_float("eps_decay",  0.990, 0.9995),
        cost_limit = trial.suggest_float("cost_limit", 0.01, 0.30),
        lambda_lr  = trial.suggest_float("lambda_lr",  0.01, 0.30),
        lambda_max = trial.suggest_float("lambda_max", 1.0,  20.0),
    )
    val, _, _ = eval_lagrangian(kw, seed=trial.number)
    return val

def _optuna_sl(trial):
    kw = dict(
        alpha      = trial.suggest_float("alpha",      0.01, 0.5, log=True),
        gamma      = trial.suggest_float("gamma",      0.90, 0.999),
        eps_decay  = trial.suggest_float("eps_decay",  0.990, 0.9995),
        sl_penalty = trial.suggest_float("sl_penalty", 5.0,  80.0),
    )
    val, _, _ = eval_safety_layer(kw, seed=trial.number)
    return val

def tune(n_trials=N_TRIALS, output_dir=OUTPUT_DIR):
    results = {}
    specs = [
        ("lagrangian",   _optuna_lag, _sample_lagrangian,   eval_lagrangian),
        ("safety_layer", _optuna_sl,  _sample_safety_layer, eval_safety_layer),
    ]
    for name, optuna_obj, sampler_fn, eval_fn in specs:
        print(f"\n{'='*55}\n  Tuning {name}  ({n_trials} trials)\n{'='*55}")
        if HAS_OPTUNA:
            study = optuna.create_study(
                direction="maximize",
                sampler=optuna.samplers.TPESampler(seed=0))
            study.optimize(optuna_obj, n_trials=n_trials, show_progress_bar=True)
            best_val, best_params = study.best_value, study.best_params
        else:
            best_val, best_params = random_search(name, sampler_fn, eval_fn, n_trials)
        print(f"\n  Best score: {best_val:.2f}")
        print(f"  Best params: {best_params}")
        results[name] = {"best_value": best_val, "best_params": best_params}

    for path in [os.path.join(output_dir, "best_params.json"), "best_params.json"]:
        with open(path, "w") as f:
            json.dump(results, f, indent=2, default=str)
    print(f"\nSaved → best_params.json")
    return results

if __name__ == "__main__":
    tune()
