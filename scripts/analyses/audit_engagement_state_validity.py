"""Audit external validity of behavior-only GLM-HMM states for LAB-TASKS-445."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import statsmodels.formula.api as smf

import _bootstrap  # noqa: F401

from behavior_analyses.engagement_states import TASK_MODE
from behavior_analyses.io import get_chipmunk_table


KEY = ["subject_name", "session_name", "dataset_name", "trial_num"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject", required=True)
    parser.add_argument("--posterior-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--posterior-threshold", type=float, default=0.70)
    parser.add_argument("--bootstrap-samples", type=int, default=2_000)
    parser.add_argument("--seed", type=int, default=445)
    args = parser.parse_args()
    if not 0 < args.posterior_threshold < 1:
        parser.error("--posterior-threshold must be between zero and one")
    return args


def fetch_external_trials(subject: str) -> pd.DataFrame:
    Chipmunk = get_chipmunk_table()
    relation = (
        (Chipmunk() & {"setting_task_mode": TASK_MODE, "subject_name": subject})
        * Chipmunk.Trial()
        * Chipmunk.TrialParameters()
    )
    fields = [
        *KEY,
        "early_withdrawal",
        "t_gocue",
        "t_response",
    ]
    return pd.DataFrame(dict(zip(fields, relation.fetch(*fields))))


def clustered_mean(frame, value, rng, n_bootstrap):
    finite = frame.loc[np.isfinite(frame[value]), ["session_name", value]]
    grouped = finite.groupby("session_name")[value].agg(["sum", "count"])
    if grouped.empty:
        return np.nan, np.nan, np.nan, 0, 0
    draws = rng.integers(0, len(grouped), size=(n_bootstrap, len(grouped)))
    estimates = grouped["sum"].to_numpy()[draws].sum(axis=1) / grouped[
        "count"
    ].to_numpy()[draws].sum(axis=1)
    return (
        float(finite[value].mean()),
        *np.quantile(estimates, [0.025, 0.975]).tolist(),
        len(finite),
        len(grouped),
    )


def state_summary(confident, rng, n_bootstrap):
    variables = {
        "response_latency_s": "response_latency_s",
        "current_reward_rate": "rewarded",
        "previous_reward_rate": "previous_reward",
        "trial_position_fraction": "trial_position_fraction",
        "absolute_stimulus_evidence_hz": "absolute_stimulus_evidence_hz",
    }
    rows = []
    for state, trials in confident.groupby("inferred_state", sort=True):
        for label, column in variables.items():
            estimate, lower, upper, n_trials, n_sessions = clustered_mean(
                trials, column, rng, n_bootstrap
            )
            rows.append(
                {
                    "state": int(state),
                    "external_variable": label,
                    "estimate": estimate,
                    "ci_lower": lower,
                    "ci_upper": upper,
                    "n_trials": n_trials,
                    "n_sessions": n_sessions,
                }
            )
    return pd.DataFrame(rows)


def regression_row(name, fit, term, n_trials, n_sessions):
    lower, upper = fit.conf_int().loc[term]
    return {
        "contrast": name,
        "estimate": float(fit.params[term]),
        "ci_lower": float(lower),
        "ci_upper": float(upper),
        "p_value": float(fit.pvalues[term]),
        "n_trials": n_trials,
        "n_sessions": n_sessions,
        "model": fit.model.formula,
        "uncertainty": "session-clustered 95% CI",
    }


def controlled_contrasts(confident):
    rows = []
    latency = confident.loc[
        np.isfinite(confident["response_latency_s"])
        & confident["response_latency_s"].gt(0)
    ].copy()
    latency["log_response_latency"] = np.log(latency["response_latency_s"])
    latency_fit = smf.ols(
        "log_response_latency ~ high_sensory_state + stimulus_evidence_hz + "
        "C(choice_right) + trial_position_fraction + C(session_name)",
        data=latency,
    ).fit(cov_type="cluster", cov_kwds={"groups": latency["session_name"]})
    rows.append(
        regression_row(
            "high-sensory state effect on log response latency",
            latency_fit,
            "high_sensory_state",
            len(latency),
            latency["session_name"].nunique(),
        )
    )

    reward_fit = smf.ols(
        "rewarded ~ high_sensory_state + absolute_stimulus_evidence_hz + "
        "C(choice_right) + trial_position_fraction + C(session_name)",
        data=confident,
    ).fit(cov_type="cluster", cov_kwds={"groups": confident["session_name"]})
    rows.append(
        regression_row(
            "high-sensory state effect on current reward probability",
            reward_fit,
            "high_sensory_state",
            len(confident),
            confident["session_name"].nunique(),
        )
    )

    position_fit = smf.ols(
        "high_sensory_state ~ trial_position_fraction + C(session_name)",
        data=confident,
    ).fit(cov_type="cluster", cov_kwds={"groups": confident["session_name"]})
    rows.append(
        regression_row(
            "within-session high-sensory occupancy drift",
            position_fit,
            "trial_position_fraction",
            len(confident),
            confident["session_name"].nunique(),
        )
    )
    return pd.DataFrame(rows)


def withdrawal_contrasts(raw, occupancy, rng, n_bootstrap):
    withdrawal = (
        raw.groupby("session_name", as_index=False)
        .agg(
            withdrawal_rate=("early_withdrawal", "mean"),
            n_raw_trials=("trial_num", "size"),
        )
        .merge(occupancy, on="session_name", validate="one_to_many")
    )
    rows = []
    for state, sessions in withdrawal.groupby("state", sort=True):
        estimate = spearmanr(
            sessions["posterior_occupancy"], sessions["withdrawal_rate"]
        ).statistic
        boot = []
        for _ in range(n_bootstrap):
            sample = sessions.iloc[rng.integers(0, len(sessions), len(sessions))]
            if (
                sample["posterior_occupancy"].nunique() > 1
                and sample["withdrawal_rate"].nunique() > 1
            ):
                boot.append(
                    spearmanr(
                        sample["posterior_occupancy"], sample["withdrawal_rate"]
                    ).statistic
                )
        lower, upper = np.quantile(boot, [0.025, 0.975])
        rows.append(
            {
                "contrast": "session occupancy vs withdrawal rate",
                "state": int(state),
                "estimate": float(estimate),
                "ci_lower": float(lower),
                "ci_upper": float(upper),
                "n_sessions": len(sessions),
                "uncertainty": "session bootstrap 95% CI",
            }
        )
    return withdrawal, pd.DataFrame(rows)


def save_figure(output_dir, parameters, summary, withdrawal):
    import matplotlib.pyplot as plt

    states = parameters.sort_values("ordered_state")
    colors = plt.cm.tab10.colors
    figure, axes = plt.subplots(2, 2, figsize=(9, 7))

    for state in states.itertuples():
        axes[0, 0].plot(
            ["Bias", "Evidence"],
            [state.intercept, state.stimulus_evidence_hz],
            marker="o",
            label=f"State {state.ordered_state}",
        )
    axes[0, 0].axhline(0, color="0.7", linewidth=0.8)
    axes[0, 0].set(title="Choice-GLM identity", ylabel="Coefficient (log odds)")
    axes[0, 0].legend(frameon=False)

    latency = summary.loc[summary["external_variable"].eq("response_latency_s")]
    axes[0, 1].errorbar(
        latency["state"],
        latency["estimate"],
        yerr=[
            latency["estimate"] - latency["ci_lower"],
            latency["ci_upper"] - latency["estimate"],
        ],
        fmt="o",
        capsize=4,
    )
    axes[0, 1].set(
        title="Response latency at posterior ≥ 0.70",
        xlabel="Ordered state",
        ylabel="Mean seconds (session-bootstrap CI)",
        xticks=latency["state"],
    )

    for state, sessions in withdrawal.groupby("state", sort=True):
        axes[1, 0].scatter(
            sessions["posterior_occupancy"],
            sessions["withdrawal_rate"],
            s=12,
            alpha=0.55,
            color=colors[int(state)],
            label=f"State {int(state)}",
        )
    axes[1, 0].set(
        title="Session occupancy and withdrawal",
        xlabel="Mean posterior occupancy",
        ylabel="Early-withdrawal rate",
    )
    axes[1, 0].legend(frameon=False)

    confident = summary.loc[summary["external_variable"].eq("trial_position_fraction")]
    axes[1, 1].errorbar(
        confident["state"],
        confident["estimate"],
        yerr=[
            confident["estimate"] - confident["ci_lower"],
            confident["ci_upper"] - confident["estimate"],
        ],
        fmt="o",
        capsize=4,
    )
    axes[1, 1].set(
        title="Within-session trial position",
        xlabel="Ordered state",
        ylabel="Mean full-session fraction",
        xticks=confident["state"],
    )
    figure.suptitle("Behavior-only state validity audit", fontsize=12)
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    figure.savefig(output_dir / "behavioral_state_validity.pdf", bbox_inches="tight")
    figure.savefig(output_dir / "behavioral_state_validity.svg", bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    posteriors = pd.read_csv(args.posterior_dir / "trial_state_posteriors.csv")
    parameters = pd.read_csv(args.posterior_dir / "selected_state_parameters.csv")
    occupancy = pd.read_csv(args.posterior_dir / "session_state_occupancy.csv")
    if set(posteriors["subject_name"].unique()) != {args.subject}:
        raise ValueError("Posterior table subject does not match --subject")

    raw = fetch_external_trials(args.subject)
    trials = posteriors.merge(
        raw[[*KEY, "t_gocue", "t_response"]],
        on=KEY,
        how="left",
        validate="one_to_one",
    )
    bounds = raw.groupby("session_name")["trial_num"].agg(["min", "max"])
    trials = trials.join(bounds, on="session_name")
    trials["trial_position_fraction"] = (
        (trials["trial_num"] - trials["min"]) / (trials["max"] - trials["min"])
    ).fillna(0.0)
    trials["response_latency_s"] = pd.to_numeric(
        trials["t_response"], errors="coerce"
    ) - pd.to_numeric(trials["t_gocue"], errors="coerce")
    trials.loc[trials["response_latency_s"].le(0), "response_latency_s"] = np.nan
    trials["absolute_stimulus_evidence_hz"] = trials["stimulus_evidence_hz"].abs()
    high_sensory_state = int(
        parameters.loc[parameters["stimulus_evidence_hz"].idxmax(), "ordered_state"]
    )
    trials["high_sensory_state"] = (
        trials["inferred_state"].eq(high_sensory_state).astype(int)
    )
    confident = trials.loc[
        trials["max_state_probability"].ge(args.posterior_threshold)
    ].copy()

    rng = np.random.default_rng(args.seed)
    summary = state_summary(confident, rng, args.bootstrap_samples)
    controls = controlled_contrasts(confident)
    withdrawal, withdrawal_summary = withdrawal_contrasts(
        raw, occupancy, rng, args.bootstrap_samples
    )
    availability = pd.DataFrame(
        [
            ("response_latency", "available", "t_response - t_gocue"),
            (
                "withdrawal_rate",
                "available_session_level_only",
                "excluded trials have no state posterior",
            ),
            ("reward_history", "descriptive_only", "previous reward defines states"),
            ("trial_position", "available", "full-session trial-number fraction"),
            ("motion", "unavailable", "not present in the Chipmunk behavior schema"),
        ],
        columns=["external_variable", "status", "reason"],
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output_dir / "state_external_variable_summary.csv", index=False)
    controls.to_csv(args.output_dir / "controlled_contrasts.csv", index=False)
    withdrawal_summary.to_csv(
        args.output_dir / "withdrawal_occupancy_contrasts.csv", index=False
    )
    availability.to_csv(
        args.output_dir / "external_variable_availability.csv", index=False
    )
    save_figure(args.output_dir, parameters, summary, withdrawal)

    metadata = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "task": "LAB-TASKS-445",
        "subject_name": args.subject,
        "posterior_threshold": args.posterior_threshold,
        "high_sensory_state": high_sensory_state,
        "n_confident_trials": len(confident),
        "n_dropped_trials": len(trials) - len(confident),
        "bootstrap_samples": args.bootstrap_samples,
        "seed": args.seed,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    verdict = (
        "The K=2 high-sensory state remains a candidate rather than an engaged "
        "label. Available response-latency, reward, withdrawal, and trial-position "
        "checks are reported here, but motion is absent from the Chipmunk behavior "
        "schema and withdrawal can only be tested at session level. Do not authorize "
        "the neural claim until those results are reviewed and the motion gap is "
        "resolved or explicitly accepted.\n"
    )
    (args.output_dir / "verdict.txt").write_text(verdict, encoding="utf-8")
    print(summary.to_string(index=False))
    print("\n" + controls.to_string(index=False))
    print("\n" + withdrawal_summary.to_string(index=False))
    print("\n" + verdict)


if __name__ == "__main__":
    main()
