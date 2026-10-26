"""Out-of-distribution AMH/AN selection test with an Active Model B term.

The trained networks and the candidate simulators are deliberately left
unchanged.  Only the *observed* AMH trajectory is generated with

    mu = a phi + b phi**3 - kp nabla**2 phi + lam |nabla phi|**2.

Thus ``lam=0`` is an exact baseline from the training model, while non-zero
values test whether the original pipeline still recognises modified scalar
active matter as AMH.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.fft as sp_fft

import model_selection as ms


DEFAULT_AMH_MODEL = Path("amh_float32_32/model_20260705_152636.pth")
DEFAULT_AMH_SCALER = Path("amh_float32_32/scaler_20260705_152636.pkl")
DEFAULT_AN_MODEL = Path("an_psi_results_old/model_20260705_153631.pth")
DEFAULT_AN_SCALER = Path("an_psi_results_old/scaler_20260705_153631.pkl")


class FPSNumpyLambda(ms.FPS_Numpy):
    """The paper's AMH solver with the AMB lambda chemical-potential term."""

    def __init__(self, p):
        super().__init__(p)
        self.lam = np.float32(p.get("lam", 0.0))

    def step(self, u_hat):
        # Preserve the original solver exactly for the lambda=0 control.
        if self.lam == 0:
            return super().step(u_hat)

        uc = sp_fft.ifft2(u_hat)
        vxg, vyg = self.flow(u_hat)
        ux = sp_fft.ifft2(self.Dx * u_hat)
        uy = sp_fft.ifft2(self.Dy * u_hat)

        advection = -self.k_tilde * sp_fft.fft2(vxg * ux + vyg * uy)
        cubic = -self.q2b * sp_fft.fft2(uc**3)

        # Since d_t phi contains nabla^2 mu, the Fourier contribution from
        # mu_lambda = lambda |nabla phi|^2 is
        # -q^2 lambda FFT[ (d_x phi)^2 + (d_y phi)^2 ].
        amb_term = -self.q2 * self.lam * sp_fft.fft2(ux**2 + uy**2)
        n_hat = advection + cubic + amb_term

        linear = -self.q2 * (self.a + self.kp * self.q2)
        denom = np.where(linear == 0, np.float32(1.0), linear)
        propagator = np.where(
            linear == 0,
            self.dt,
            (self.eA - np.float32(1.0)) / denom,
        )
        return u_hat * self.eA + n_hat * propagator


def run_amh_lambda(k_tilde, kp, b, lam, seed, config=None):
    """Generate one unseen AMH observation; return ``None`` on divergence."""
    cfg = dict(ms.AMH_CFG if config is None else config)
    p = dict(
        Ng=cfg["Ng"],
        Nt=cfg["Nt"],
        dt=cfg["dt"],
        h=cfg["h"],
        a=-float(b),
        b=float(b),
        kp=float(kp),
        k_tilde=float(k_tilde),
        lam=float(lam),
        Nf=cfg["Nf"],
    )
    np.random.seed(seed)
    with np.errstate(over="ignore", invalid="ignore"):
        vx, vy, phi = FPSNumpyLambda(p).run_full()
    if not (np.isfinite(vx).all() and np.isfinite(vy).all()
            and np.isfinite(phi).all()):
        return None
    return vx, vy, phi


def sample_parameter_sets(n_samples, seed):
    """Sample once, then reuse every parameter set at every lambda value."""
    rng = np.random.default_rng(seed)
    return [
        dict(
            k_tilde=float(rng.uniform(-1.2, -0.1)),
            kp=float(rng.uniform(0.8, 1.2)),
            b=float(rng.uniform(0.25, 1.25)),
        )
        for _ in range(n_samples)
    ]


def run_experiment(lambda_values, n_samples=20, realizations=5, seed=42,
                   save_fields=False, output_dir="amh_lambda_robustness"):
    """Run matched modified-AMH cases through the frozen selection pipeline."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    params = sample_parameter_sets(n_samples, seed)
    rows = []

    total = len(lambda_values) * n_samples
    completed = 0
    for lam_index, lam in enumerate(lambda_values):
        for sample_index, true_p in enumerate(params):
            observation_seed = seed + 10_000 * sample_index
            out = run_amh_lambda(**true_p, lam=lam, seed=observation_seed)
            completed += 1
            print(
                f"[{completed:>3}/{total}] lambda={lam:+.3g}, "
                f"case={sample_index + 1}/{n_samples}",
                flush=True,
            )
            if out is None:
                rows.append(dict(
                    lam=lam, sample=sample_index, observation_seed=observation_seed,
                    **true_p, status="diverged", correct=False,
                ))
                continue

            vx, vy, phi = out
            if save_fields:
                np.savez_compressed(
                    output_dir / f"fields_lam{lam:+.3g}_sample{sample_index:03d}.npz",
                    vx=vx, vy=vy, phi=phi,
                )

            # This seed controls only the unchanged lambda=0 candidate
            # resimulations used by Algorithm 1.
            selection_seed = seed + 1_000_000 + 10_000 * lam_index + sample_index
            np.random.seed(selection_seed)
            result = ms.select_model(vx, vy, R=realizations, verbose=False)
            rows.append(dict(
                lam=lam,
                sample=sample_index,
                observation_seed=observation_seed,
                selection_seed=selection_seed,
                **true_p,
                status="ok",
                winner=result["model"],
                correct=result["model"] == "AMH",
                confidence=result["confidence"],
                c_AMH=result["conf"]["AMH"],
                c_AN=result["conf"]["AN"],
                d_AMH=result["scores"]["AMH"],
                d_AN=result["scores"]["AN"],
                log10_dAN_over_dAMH=np.log10(
                    result["scores"]["AN"] / result["scores"]["AMH"]
                ),
                k_tilde_pred=result["amh_params"]["k_tilde"],
                kp_pred=result["amh_params"]["kp"],
                b_pred=result["amh_params"]["b"],
                kappa_pred=result["an_params"]["kappa"],
                zeta_pred=result["an_params"]["zeta"],
            ))

            # Checkpoint after every successful case so partial progress survives.
            pd.DataFrame(rows).to_pickle(output_dir / "cases.pkl")

    cases = pd.DataFrame(rows)
    if "winner" not in cases:
        cases["winner"] = np.nan
    baseline = (
        cases[(cases["lam"] == 0) & (cases["status"] == "ok")]
        .drop_duplicates("sample")
        .set_index("sample")["winner"]
    )
    cases["baseline_winner"] = cases["sample"].map(baseline)
    cases["flipped_from_lambda0"] = np.where(
        (cases["status"] == "ok") & cases["baseline_winner"].notna(),
        cases["winner"] != cases["baseline_winner"],
        np.nan,
    )
    cases.to_pickle(output_dir / "cases.pkl")
    cases.to_csv(output_dir / "cases.csv", index=False)
    summary = summarize(cases)
    summary.to_csv(output_dir / "summary.csv", index=False)
    plot_summary(summary, output_dir / "robustness_vs_lambda.png")
    return cases, summary


def summarize(cases):
    valid = cases[cases["status"] == "ok"].copy()
    if valid.empty:
        return pd.DataFrame()
    valid["kt_abs_error_pct"] = (
        100 * np.abs(np.abs(valid["k_tilde_pred"]) - np.abs(valid["k_tilde"]))
        / np.abs(valid["k_tilde"])
    )
    valid["kp_error_pct"] = 100 * np.abs(valid["kp_pred"] - valid["kp"]) / valid["kp"]
    valid["b_error_pct"] = 100 * np.abs(valid["b_pred"] - valid["b"]) / valid["b"]
    summary = valid.groupby("lam", as_index=False).agg(
        n=("correct", "size"),
        amh_classification_rate=("correct", "mean"),
        mean_amh_probability=("c_AMH", "mean"),
        mean_log10_dAN_over_dAMH=("log10_dAN_over_dAMH", "mean"),
        median_log10_dAN_over_dAMH=("log10_dAN_over_dAMH", "median"),
        kt_mape=("kt_abs_error_pct", "mean"),
        kp_mape=("kp_error_pct", "mean"),
        b_mape=("b_error_pct", "mean"),
    )
    attempts = cases.groupby("lam", as_index=False).agg(
        n_attempted=("status", "size"),
        n_diverged=("status", lambda value: int((value == "diverged").sum())),
    )
    summary = attempts.merge(summary, on="lam", how="left")

    # Wilson 95% interval remains well behaved for rates of exactly zero or one.
    z = 1.959963984540054
    p = summary["amh_classification_rate"]
    n = summary["n"]
    denominator = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denominator
    half_width = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denominator
    summary["classification_rate_ci95_low"] = centre - half_width
    summary["classification_rate_ci95_high"] = centre + half_width

    if "flipped_from_lambda0" in valid:
        flip_rate = valid.groupby("lam")["flipped_from_lambda0"].mean()
        summary["flip_rate_from_lambda0"] = summary["lam"].map(flip_rate)
    return summary


def plot_summary(summary, path):
    if summary.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    rate = summary["amh_classification_rate"].to_numpy()
    rate_err = np.vstack([
        rate - summary["classification_rate_ci95_low"].to_numpy(),
        summary["classification_rate_ci95_high"].to_numpy() - rate,
    ])
    axes[0].errorbar(summary["lam"], rate, yerr=rate_err, fmt="o-", capsize=3,
                     label="AMH decision (95% CI)")
    axes[0].plot(summary["lam"], summary["mean_amh_probability"], "s--", label="mean AMH weight")
    axes[0].set(xlabel=r"$\lambda$", ylabel="fraction / weight", ylim=(-0.05, 1.05))
    axes[0].legend(frameon=False)
    axes[1].plot(summary["lam"], summary["mean_log10_dAN_over_dAMH"], "o-")
    axes[1].axhline(0, color="black", lw=1, ls="--")
    axes[1].set(xlabel=r"$\lambda$", ylabel=r"mean $\log_{10}(d_{AN}/d_{AMH})$")
    fig.savefig(path, dpi=200)
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lambdas", nargs="+", type=float,
                        default=[-1.0, -0.5, -0.25, 0.0, 0.25, 0.5, 1.0])
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--realizations", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="amh_lambda_robustness")
    parser.add_argument("--save-fields", action="store_true")
    parser.add_argument("--amh-model", type=Path, default=DEFAULT_AMH_MODEL)
    parser.add_argument("--amh-scaler", type=Path, default=DEFAULT_AMH_SCALER)
    parser.add_argument("--an-model", type=Path, default=DEFAULT_AN_MODEL)
    parser.add_argument("--an-scaler", type=Path, default=DEFAULT_AN_SCALER)
    return parser.parse_args()


def main():
    args = parse_args()
    model_paths = [args.amh_model, args.amh_scaler, args.an_model, args.an_scaler]
    missing = [str(path) for path in model_paths if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing trained model artifact(s): " + ", ".join(missing))

    ms.load_models(*(str(path) for path in model_paths))
    config = vars(args).copy()
    config["lambdas"] = config.pop("lambdas")
    for key in ("amh_model", "amh_scaler", "an_model", "an_scaler"):
        config[key] = str(config[key])
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")

    _, summary = run_experiment(
        lambda_values=args.lambdas,
        n_samples=args.samples,
        realizations=args.realizations,
        seed=args.seed,
        save_fields=args.save_fields,
        output_dir=args.output_dir,
    )
    print("\nSummary")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
