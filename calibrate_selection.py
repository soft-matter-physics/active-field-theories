"""Calibrate frozen AMH/AN model selection from held-out simulations.

This script does not retrain either neural network.  It consumes the scalar
outputs previously produced by ``model_selection.py`` together with their
matching held-out parameter-recovery data and saves one reusable artifact:

* a one-parameter logistic calibration of the signed discrepancy ratio;
* empirical, model-specific absolute goodness-of-fit reference distributions;
* finite-sample conformal intervals for every inferred parameter.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import expit
from sklearn.metrics import brier_score_loss, log_loss


EPS = 1e-12


PARAMETER_SPECS = {
    "AMH": {
        "k_tilde": ("kt", "k_tilde_pred", -1),
        "kp": ("kp", "kp_pred", 1),
        "b": ("b", "b_pred", 1),
    },
    "AN": {
        "kappa": ("kappa", "kappa_pred", 1),
        "zeta": ("zeta", "zeta_pred", -1),
    },
}


def finite_sample_quantile(values, coverage):
    """Conformal 'higher' quantile with finite-sample rank correction."""
    values = np.sort(np.asarray(values, dtype=float))
    if values.size == 0:
        raise ValueError("Cannot calculate a quantile from an empty sample.")
    rank = min(values.size, int(np.ceil((values.size + 1) * coverage)))
    return float(values[rank - 1])


def signed_log_ratio(results):
    return np.log(
        (results["d_AN"].to_numpy(float) + EPS)
        / (results["d_AMH"].to_numpy(float) + EPS)
    )


def _logistic_objective(slope, x, y, regularization_c):
    probability = np.clip(expit(slope * x), EPS, 1 - EPS)
    nll = -np.mean(y * np.log(probability) + (1 - y) * np.log(1 - probability))
    # Same scaling as an L2-regularized logistic regression with strength C.
    penalty = slope**2 / (2 * regularization_c * len(y))
    return float(nll + penalty)


def fit_symmetric_logistic(x, y, regularization_c=10.0):
    """Fit P(AMH|x)=sigmoid(slope*x), preserving the x=0 boundary."""
    result = minimize_scalar(
        lambda slope: _logistic_objective(slope, x, y, regularization_c),
        bounds=(1e-6, 100.0),
        method="bounded",
    )
    if not result.success:
        raise RuntimeError(f"Logistic calibration failed: {result.message}")
    return float(result.x)


def leave_one_out_logistic(x, y, regularization_c=10.0):
    probability = np.empty(len(y), dtype=float)
    for index in range(len(y)):
        keep = np.arange(len(y)) != index
        slope = fit_symmetric_logistic(x[keep], y[keep], regularization_c)
        probability[index] = expit(slope * x[index])
    return probability


def reliability_table(probability, labels, n_bins=4):
    frame = pd.DataFrame({"p_amh": probability, "is_amh": labels})
    # Equal-count bins are more stable than fixed-width bins for small data.
    frame["bin"] = pd.qcut(frame["p_amh"], q=min(n_bins, len(frame)),
                           duplicates="drop")
    return frame.groupby("bin", observed=True).agg(
        n=("is_amh", "size"),
        mean_predicted_p_amh=("p_amh", "mean"),
        observed_amh_fraction=("is_amh", "mean"),
    ).reset_index().astype({"bin": str})


def parameter_calibration(model_name, truth, selection, coverage):
    calibrated = {}
    for name, (true_column, predicted_column, sign) in PARAMETER_SPECS[model_name].items():
        true_value = np.abs(truth[true_column].to_numpy(float))
        predicted = np.abs(selection[predicted_column].to_numpy(float))
        if np.any(true_value <= 0) or np.any(predicted <= 0):
            raise ValueError(f"{model_name} {name} contains non-positive magnitudes.")
        residual = np.abs(np.log10(true_value) - np.log10(predicted))
        radius = finite_sample_quantile(residual, coverage)

        # Leave-one-out coverage is diagnostic only; the final interval uses
        # all calibration cases and retains split-conformal validity on new data.
        covered = []
        for index, score in enumerate(residual):
            radius_loo = finite_sample_quantile(np.delete(residual, index), coverage)
            covered.append(score <= radius_loo)

        calibrated[name] = {
            "log10_radius": radius,
            "multiplicative_factor": float(10**radius),
            "sign": int(sign),
            "n": int(len(residual)),
            "loo_coverage": float(np.mean(covered)),
            "median_absolute_log10_error": float(np.median(residual)),
        }
    return calibrated


def validate_alignment(truth, selection, model_name):
    if len(truth) != len(selection):
        raise ValueError(
            f"{model_name}: truth has {len(truth)} rows but selection results "
            f"have {len(selection)} rows; they must describe the same cases in the same order."
        )
    expected = selection.get("true_model")
    if expected is not None and not (expected.astype(str) == model_name).all():
        raise ValueError(f"{model_name}: true_model labels do not match the input file.")


def build_artifact(amh_truth, an_truth, amh_results, an_results,
                   parameter_coverage=0.90, adequacy_alpha=0.01,
                   realizations_used=5, logistic_c=10.0):
    validate_alignment(amh_truth, amh_results, "AMH")
    validate_alignment(an_truth, an_results, "AN")

    combined = pd.concat([amh_results, an_results], ignore_index=True)
    labels = np.r_[np.ones(len(amh_results)), np.zeros(len(an_results))]
    log_ratio = signed_log_ratio(combined)

    loo_probability = leave_one_out_logistic(log_ratio, labels, logistic_c)
    slope = fit_symmetric_logistic(log_ratio, labels, logistic_c)
    loo_predicted = (loo_probability >= 0.5).astype(int)
    logistic_diagnostics = {
        "leave_one_out_accuracy": float(np.mean(loo_predicted == labels)),
        "leave_one_out_brier": float(brier_score_loss(labels, loo_probability)),
        "leave_one_out_log_loss": float(log_loss(labels, loo_probability, labels=[0, 1])),
        "slope": slope,
        "regularization_c": float(logistic_c),
        "n": int(len(labels)),
    }

    amh_reference = np.sort(amh_results["d_AMH"].to_numpy(float))
    an_reference = np.sort(an_results["d_AN"].to_numpy(float))
    adequacy_coverage = 1.0 - adequacy_alpha

    artifact = {
        "version": 1,
        "selection_realizations": int(realizations_used),
        "logistic": {
            "form": "P(AMH|ell) = sigmoid(slope * ell)",
            "ell": "ln((d_AN + eps) / (d_AMH + eps))",
            **logistic_diagnostics,
        },
        "adequacy": {
            "alpha": float(adequacy_alpha),
            "AMH": {
                "reference_discrepancies": amh_reference,
                "threshold": finite_sample_quantile(amh_reference, adequacy_coverage),
                "leave_one_out_acceptance_rate": leave_one_out_adequacy_rate(
                    amh_reference, adequacy_alpha
                ),
            },
            "AN": {
                "reference_discrepancies": an_reference,
                "threshold": finite_sample_quantile(an_reference, adequacy_coverage),
                "leave_one_out_acceptance_rate": leave_one_out_adequacy_rate(
                    an_reference, adequacy_alpha
                ),
            },
        },
        "parameter_intervals": {
            "coverage": float(parameter_coverage),
            "AMH": parameter_calibration(
                "AMH", amh_truth, amh_results, parameter_coverage
            ),
            "AN": parameter_calibration(
                "AN", an_truth, an_results, parameter_coverage
            ),
        },
    }
    return artifact, reliability_table(loo_probability, labels)


def goodness_of_fit_pvalue(discrepancy, reference):
    """Upper-tail conformal p-value; small values indicate inadequate fit."""
    reference = np.asarray(reference, dtype=float)
    return float((1 + np.count_nonzero(reference >= discrepancy)) / (len(reference) + 1))


def leave_one_out_adequacy_rate(values, alpha):
    values = np.asarray(values, dtype=float)
    accepted = []
    for index, value in enumerate(values):
        reference = np.delete(values, index)
        accepted.append(goodness_of_fit_pvalue(value, reference) >= alpha)
    return float(np.mean(accepted))


def calibrated_selection(raw_result, artifact):
    """Add calibrated probability, separation, adequacy, and intervals."""
    d_amh = float(raw_result["scores"]["AMH"])
    d_an = float(raw_result["scores"]["AN"])
    ell = float(np.log((d_an + EPS) / (d_amh + EPS)))
    p_amh = float(expit(artifact["logistic"]["slope"] * ell))
    relative_winner = "AMH" if d_amh < d_an else "AN"
    probability_winner = p_amh if relative_winner == "AMH" else 1 - p_amh
    separation = abs(d_an - d_amh) / (d_an + d_amh + EPS)

    p_gof = {}
    adequate = {}
    normalized_discrepancy = {}
    for model_name, discrepancy in (("AMH", d_amh), ("AN", d_an)):
        reference = artifact["adequacy"][model_name]["reference_discrepancies"]
        threshold = artifact["adequacy"][model_name].get("threshold")
        if threshold is None:
            threshold = finite_sample_quantile(
                reference, 1.0 - artifact["adequacy"]["alpha"]
            )
        normalized_discrepancy[model_name] = float(discrepancy / threshold)
        p_gof[model_name] = goodness_of_fit_pvalue(discrepancy, reference)
        adequate[model_name] = p_gof[model_name] >= artifact["adequacy"]["alpha"]
    eligible = [name for name in ("AMH", "AN") if adequate[name]]
    if not eligible:
        final_decision = "NEITHER"
    elif len(eligible) == 1:
        final_decision = eligible[0]
    else:
        final_decision = relative_winner

    intervals = {}
    points = {"AMH": raw_result["amh_params"], "AN": raw_result["an_params"]}
    for model_name, parameters in points.items():
        intervals[model_name] = {}
        for name, info in artifact["parameter_intervals"][model_name].items():
            point = float(parameters[name])
            magnitude = abs(point)
            factor = info["multiplicative_factor"]
            lower_magnitude, upper_magnitude = magnitude / factor, magnitude * factor
            if info["sign"] < 0:
                lower, upper = -upper_magnitude, -lower_magnitude
            else:
                lower, upper = lower_magnitude, upper_magnitude
            intervals[model_name][name] = {
                "point": point, "lower": lower, "upper": upper
            }

    return {
        "relative_winner": relative_winner,
        "final_decision": final_decision,
        "p_amh": p_amh,
        "winner_probability": probability_winner,
        "separation": float(separation),
        "log_discrepancy_ratio": ell,
        "p_gof": p_gof,
        "normalized_discrepancy": normalized_discrepancy,
        "adequate": adequate,
        "eligible_models": eligible,
        "intervals": intervals,
    }


def artifact_summary(artifact):
    rows = {
        "logistic_slope": artifact["logistic"]["slope"],
        "logistic_loo_accuracy": artifact["logistic"]["leave_one_out_accuracy"],
        "logistic_loo_brier": artifact["logistic"]["leave_one_out_brier"],
        "logistic_loo_log_loss": artifact["logistic"]["leave_one_out_log_loss"],
        "amh_adequacy_threshold": artifact["adequacy"]["AMH"]["threshold"],
        "an_adequacy_threshold": artifact["adequacy"]["AN"]["threshold"],
        "amh_adequacy_loo_acceptance": artifact["adequacy"]["AMH"][
            "leave_one_out_acceptance_rate"
        ],
        "an_adequacy_loo_acceptance": artifact["adequacy"]["AN"][
            "leave_one_out_acceptance_rate"
        ],
    }
    for model_name in ("AMH", "AN"):
        for parameter, info in artifact["parameter_intervals"][model_name].items():
            prefix = f"{model_name.lower()}_{parameter}"
            rows[f"{prefix}_factor"] = info["multiplicative_factor"]
            rows[f"{prefix}_loo_coverage"] = info["loo_coverage"]
    return rows


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--amh-data", type=Path,
                        default=Path("amh_float32_32/results_df_20260705_152636.pkl"))
    parser.add_argument("--an-data", type=Path,
                        default=Path("an_psi_results_old/results_df_20260705_153631.pkl"))
    parser.add_argument("--amh-results", type=Path,
                        default=Path("model_selecction_results/amh.pkl"))
    parser.add_argument("--an-results", type=Path,
                        default=Path("model_selecction_results/an.pkl"))
    parser.add_argument("--parameter-coverage", type=float, default=0.90)
    parser.add_argument("--adequacy-alpha", type=float, default=0.01)
    parser.add_argument("--realizations-used", type=int, default=5,
                        help="R used when the saved selection-result files were created.")
    parser.add_argument("--logistic-c", type=float, default=10.0,
                        help="Weak L2 regularization; prevents complete-separation overconfidence.")
    parser.add_argument("--output", type=Path,
                        default=Path("calibration/model_selection_calibration.joblib"))
    return parser.parse_args()


def main():
    args = parse_args()
    if args.logistic_c <= 0:
        raise ValueError("logistic C must be positive.")
    for probability, name in ((args.parameter_coverage, "parameter coverage"),
                              (args.adequacy_alpha, "adequacy alpha")):
        if not 0 < probability < 1:
            raise ValueError(f"{name} must be strictly between zero and one.")

    amh_truth = pd.read_pickle(args.amh_data)
    an_truth = pd.read_pickle(args.an_data)
    amh_results = pd.read_pickle(args.amh_results)
    an_results = pd.read_pickle(args.an_results)
    artifact, reliability = build_artifact(
        amh_truth, an_truth, amh_results, an_results,
        parameter_coverage=args.parameter_coverage,
        adequacy_alpha=args.adequacy_alpha,
        realizations_used=args.realizations_used,
        logistic_c=args.logistic_c,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, args.output)
    reliability.to_csv(args.output.with_name("logistic_reliability.csv"), index=False)
    summary = artifact_summary(artifact)
    args.output.with_name("calibration_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )

    print(f"Saved calibration artifact to {args.output}")
    print(json.dumps(summary, indent=2))
    print("\nLeave-one-out logistic reliability")
    print(reliability.to_string(index=False))


if __name__ == "__main__":
    main()
