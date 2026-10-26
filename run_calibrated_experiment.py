"""Evaluate frozen AMH/AN selection with a saved calibration artifact.

Generates genuinely new in-distribution AMH and AN observations plus modified
AMH observations containing lambda |grad phi|^2.  Every observation receives:

* the original relative AMH/AN winner;
* a calibrated logistic probability;
* an absolute AMH/AN goodness-of-fit p-value and possible NEITHER decision;
* conformal parameter intervals from the held-out calibration stage.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

import model_selection as ms
from amh_lambda_robustness import run_amh_lambda
from calibrate_selection import calibrated_selection


DEFAULT_AMH_MODEL = Path("amh_float32_32/model_20260705_152636.pth")
DEFAULT_AMH_SCALER = Path("amh_float32_32/scaler_20260705_152636.pkl")
DEFAULT_AN_MODEL = Path("an_psi_results_old/model_20260705_153631.pth")
DEFAULT_AN_SCALER = Path("an_psi_results_old/scaler_20260705_153631.pkl")


def flatten_result(dataset, sample, true_family, true_parameters,
                   raw, calibrated, lam=np.nan):
    row = {
        "dataset": dataset,
        "sample": sample,
        "true_family": true_family,
        "lam": lam,
        **true_parameters,
        "d_AMH": raw["scores"]["AMH"],
        "d_AN": raw["scores"]["AN"],
        "relative_winner": calibrated["relative_winner"],
        "final_decision": calibrated["final_decision"],
        "family_correct": calibrated["relative_winner"] == true_family,
        "accepted_correct": calibrated["final_decision"] == true_family,
        "separation": calibrated["separation"],
        "log_discrepancy_ratio": calibrated["log_discrepancy_ratio"],
        "p_amh": calibrated["p_amh"],
        "winner_probability": calibrated["winner_probability"],
        "p_gof_AMH": calibrated["p_gof"]["AMH"],
        "p_gof_AN": calibrated["p_gof"]["AN"],
        "normalized_d_AMH": calibrated["normalized_discrepancy"]["AMH"],
        "normalized_d_AN": calibrated["normalized_discrepancy"]["AN"],
        "AMH_adequate": calibrated["adequate"]["AMH"],
        "AN_adequate": calibrated["adequate"]["AN"],
        "winner_adequate": calibrated["adequate"][calibrated["relative_winner"]],
    }
    for model_name, intervals in calibrated["intervals"].items():
        for parameter, interval in intervals.items():
            prefix = f"{model_name}_{parameter}"
            row[f"{prefix}_pred"] = interval["point"]
            row[f"{prefix}_lower"] = interval["lower"]
            row[f"{prefix}_upper"] = interval["upper"]
            if model_name == true_family and parameter in true_parameters:
                truth = float(true_parameters[parameter])
                row[f"{prefix}_covered"] = interval["lower"] <= truth <= interval["upper"]
    return row


def classify_observation(vx, vy, artifact, realizations, selection_seed):
    np.random.seed(selection_seed)
    raw = ms.select_model(vx, vy, R=realizations, verbose=False)
    return raw, calibrated_selection(raw, artifact)


def sample_amh_parameters(rng, k_tilde_min=-1.2, k_tilde_max=-0.1):
    return {
        "k_tilde": float(rng.uniform(k_tilde_min, k_tilde_max)),
        "kp": float(rng.uniform(0.8, 1.2)),
        "b": float(rng.uniform(0.25, 1.25)),
    }


def sample_an_parameters(rng):
    return {
        "kappa": float(rng.uniform(0.5, 2.0)),
        "zeta": float(rng.uniform(-1.2, -0.1)),
    }


def run_experiment(artifact, samples, lambda_values, realizations, seed,
                   output_dir, amh_observation_config=None,
                   amh_k_tilde_min=-1.2, amh_k_tilde_max=-0.1):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    rows = []

    amh_parameters = [
        sample_amh_parameters(rng, amh_k_tilde_min, amh_k_tilde_max)
        for _ in range(samples)
    ]
    an_parameters = [sample_an_parameters(rng) for _ in range(samples)]
    cases = [("AMH", 0.0)] + [(f"AMH_lambda_{lam:+g}", float(lam))
                              for lam in lambda_values if lam != 0]
    total = samples * (len(cases) + 1)
    completed = 0

    # Use matched AMH parameters and observation seeds across lambda values.
    for dataset, lam in cases:
        for sample, parameters in enumerate(amh_parameters):
            observation_seed = seed + 10_000 * sample
            fields = run_amh_lambda(
                **parameters, lam=lam, seed=observation_seed,
                config=amh_observation_config,
            )
            completed += 1
            print(f"[{completed}/{total}] {dataset}, case {sample + 1}/{samples}", flush=True)
            if fields is None:
                rows.append({"dataset": dataset, "sample": sample,
                             "true_family": "AMH", "lam": lam,
                             **parameters, "status": "diverged"})
                continue
            vx, vy, _ = fields
            selection_seed = seed + 1_000_000 + sample
            raw, calibrated = classify_observation(
                vx, vy, artifact, realizations, selection_seed
            )
            row = flatten_result(dataset, sample, "AMH", parameters,
                                 raw, calibrated, lam=lam)
            row["status"] = "ok"
            rows.append(row)
            pd.DataFrame(rows).to_pickle(output_dir / "cases.pkl")

    for sample, parameters in enumerate(an_parameters):
        observation_seed = seed + 2_000_000 + 10_000 * sample
        np.random.seed(observation_seed)
        vx, vy = ms.run_an(parameters["kappa"], parameters["zeta"])
        completed += 1
        print(f"[{completed}/{total}] AN, case {sample + 1}/{samples}", flush=True)
        if vx is None:
            rows.append({"dataset": "AN", "sample": sample,
                         "true_family": "AN", **parameters, "status": "diverged"})
            continue
        selection_seed = seed + 3_000_000 + sample
        raw, calibrated = classify_observation(
            vx, vy, artifact, realizations, selection_seed
        )
        row = flatten_result("AN", sample, "AN", parameters, raw, calibrated)
        row["status"] = "ok"
        rows.append(row)
        pd.DataFrame(rows).to_pickle(output_dir / "cases.pkl")

    cases_frame = pd.DataFrame(rows)
    cases_frame.to_pickle(output_dir / "cases.pkl")
    cases_frame.to_csv(output_dir / "cases.csv", index=False)
    summary = summarize(cases_frame)
    summary.to_csv(output_dir / "summary.csv", index=False)
    interval_coverage = summarize_interval_coverage(cases_frame)
    interval_coverage.to_csv(output_dir / "parameter_interval_coverage.csv", index=False)
    return cases_frame, summary


def summarize(cases):
    valid = cases[cases["status"] == "ok"].copy()
    if valid.empty:
        return pd.DataFrame()
    return valid.groupby(["dataset", "true_family", "lam"], dropna=False,
                         as_index=False).agg(
        n=("sample", "size"),
        relative_family_accuracy=("family_correct", "mean"),
        accepted_correct_rate=("accepted_correct", "mean"),
        adequacy_acceptance_rate=("winner_adequate", "mean"),
        neither_rate=("final_decision", lambda value: float((value == "NEITHER").mean())),
        mean_separation=("separation", "mean"),
        mean_winner_probability=("winner_probability", "mean"),
        mean_p_gof_AMH=("p_gof_AMH", "mean"),
        mean_p_gof_AN=("p_gof_AN", "mean"),
    )


def summarize_interval_coverage(cases):
    valid = cases[cases["status"] == "ok"]
    rows = []
    for (dataset, family, lam), group in valid.groupby(
            ["dataset", "true_family", "lam"], dropna=False):
        prefix = f"{family}_"
        for column in (name for name in group if name.startswith(prefix)
                       and name.endswith("_covered")):
            values = group[column].dropna().astype(bool)
            if values.empty:
                continue
            parameter = column[len(prefix):-len("_covered")]
            rows.append({
                "dataset": dataset,
                "true_family": family,
                "lam": lam,
                "parameter": parameter,
                "n": len(values),
                "empirical_coverage": values.mean(),
            })
    return pd.DataFrame(rows)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration", type=Path,
                        default=Path("calibration/model_selection_calibration.joblib"))
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--lambdas", nargs="+", type=float,
                        default=[-1.0, -0.5, 0.5, 1.0])
    parser.add_argument("--realizations", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260928)
    parser.add_argument("--output-dir", type=Path,
                        default=Path("calibrated_experiment"))
    parser.add_argument("--amh-model", type=Path, default=DEFAULT_AMH_MODEL)
    parser.add_argument("--amh-scaler", type=Path, default=DEFAULT_AMH_SCALER)
    parser.add_argument("--an-model", type=Path, default=DEFAULT_AN_MODEL)
    parser.add_argument("--an-scaler", type=Path, default=DEFAULT_AN_SCALER)
    parser.add_argument("--grid-size", type=int, default=ms.AMH_CFG["Ng"])
    parser.add_argument("--time-steps", type=int, default=ms.AMH_CFG["Nt"])
    parser.add_argument("--dt", type=float, default=ms.AMH_CFG["dt"])
    parser.add_argument("--grid-spacing", type=float, default=ms.AMH_CFG["h"])
    parser.add_argument("--saved-frames", type=int, default=ms.AMH_CFG["Nf"])
    parser.add_argument("--amh-network-frames", type=int, default=ms.N_FRAMES_AMH)
    parser.add_argument("--an-network-frames", type=int, default=ms.N_FRAMES_AN)
    parser.add_argument(
        "--amh-observation-time-steps", type=int,
        help="AMH observation length if different from candidate resimulations."
    )
    parser.add_argument("--amh-k-tilde-min", type=float, default=-1.2)
    parser.add_argument("--amh-k-tilde-max", type=float, default=-0.1)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.amh_k_tilde_min >= args.amh_k_tilde_max:
        raise ValueError("--amh-k-tilde-min must be smaller than --amh-k-tilde-max")
    artifact = joblib.load(args.calibration)
    calibrated_r = artifact.get("selection_realizations")
    if calibrated_r is not None and args.realizations != calibrated_r:
        raise ValueError(
            f"The calibration discrepancies used R={calibrated_r}, but this run "
            f"requested R={args.realizations}. Use the same R so the absolute "
            "adequacy thresholds remain comparable."
        )
    ms.configure_system(
        Ng=args.grid_size,
        Nt=args.time_steps,
        dt=args.dt,
        h=args.grid_spacing,
        Nf=args.saved_frames,
        n_frames_amh=args.amh_network_frames,
        n_frames_an=args.an_network_frames,
    )
    ms.load_models(str(args.amh_model), str(args.amh_scaler),
                   str(args.an_model), str(args.an_scaler))
    amh_observation_config = dict(ms.AMH_CFG)
    if args.amh_observation_time_steps is not None:
        amh_observation_config["Nt"] = args.amh_observation_time_steps
    args.output_dir.mkdir(parents=True, exist_ok=True)
    configuration = vars(args).copy()
    configuration = {key: str(value) if isinstance(value, Path) else value
                     for key, value in configuration.items()}
    (args.output_dir / "config.json").write_text(
        json.dumps(configuration, indent=2) + "\n"
    )
    _, summary = run_experiment(
        artifact=artifact,
        samples=args.samples,
        lambda_values=args.lambdas,
        realizations=args.realizations,
        seed=args.seed,
        output_dir=args.output_dir,
        amh_observation_config=amh_observation_config,
        amh_k_tilde_min=args.amh_k_tilde_min,
        amh_k_tilde_max=args.amh_k_tilde_max,
    )
    print("\nSummary")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
