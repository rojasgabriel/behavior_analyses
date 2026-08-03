"""Fit and compare behavior-only GLM-HMM engagement states for LAB-TASKS-445."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from itertools import groupby
import json
from pathlib import Path

import numpy as np
import pandas as pd

import _bootstrap  # noqa: F401

from behavior_analyses.engagement_states import (
    MIN_SESSIONS_FOR_CV,
    MIN_VALID_TRIALS,
    MODEL_INPUT_COLUMNS,
    build_design_matrix,
    classify_trials,
    fit_glm_hmm_sessions,
    ordered_state_parameters,
    session_folds,
    session_log_likelihood,
    session_sequences,
    smoothed_session_probabilities,
)
from engagement_state_preflight import fetch_trials, git_value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--subject")
    parser.add_argument("--synthetic-recovery", action="store_true")
    parser.add_argument("--states", nargs="+", type=int, default=[1, 2, 3, 4])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--cv-seed", type=int, default=445)
    parser.add_argument("--em-iters", type=int, default=50)
    parser.add_argument("--m-step-iters", type=int, default=50)
    args = parser.parse_args()
    if bool(args.subject) == args.synthetic_recovery:
        parser.error("choose exactly one of --subject or --synthetic-recovery")
    return args


def parameter_rows(scope, subject, num_states, fold, seed, params):
    order, coefficients, transitions = ordered_state_parameters(params)
    states = [
        {
            "scope": scope,
            "subject_name": subject,
            "num_states": num_states,
            "fold": fold,
            "seed": seed,
            "ordered_state": state,
            "original_state": int(order[state]),
            "intercept": float(coefficients[state, 0]),
            **{
                name: float(coefficients[state, i + 1])
                for i, name in enumerate(MODEL_INPUT_COLUMNS)
            },
        }
        for state in range(num_states)
    ]
    transition_rows = [
        {
            "scope": scope,
            "subject_name": subject,
            "num_states": num_states,
            "fold": fold,
            "seed": seed,
            "from_state": source,
            "to_state": target,
            "probability": float(transitions[source, target]),
        }
        for source in range(num_states)
        for target in range(num_states)
    ]
    return states, transition_rows


def cross_validate(subject, sequences, args):
    fits = []
    state_rows = []
    transition_rows = []
    folds = session_folds(sequences, args.folds, args.cv_seed)
    for num_states in args.states:
        for fold, (train_indices, test_indices) in enumerate(folds):
            train = [sequences[i] for i in train_indices]
            test = [sequences[i] for i in test_indices]
            for seed in args.seeds:
                print(f"K={num_states} fold={fold} seed={seed}", flush=True)
                model, params, trace = fit_glm_hmm_sessions(
                    train,
                    num_states,
                    seed,
                    args.em_iters,
                    args.m_step_iters,
                )
                train_ll = session_log_likelihood(model, params, train)
                test_ll = session_log_likelihood(model, params, test)
                fits.append(
                    {
                        "subject_name": subject,
                        "num_states": num_states,
                        "fold": fold,
                        "seed": seed,
                        "n_train_sessions": len(train),
                        "n_train_trials": sum(len(item[1]) for item in train),
                        "n_test_sessions": len(test),
                        "n_test_trials": sum(len(item[1]) for item in test),
                        "initial_train_log_likelihood": float(trace[0]),
                        "final_train_log_likelihood": train_ll,
                        "test_log_likelihood": test_ll,
                        "test_log_likelihood_per_trial": test_ll
                        / sum(len(item[1]) for item in test),
                        "em_trace_non_decreasing": bool(
                            np.all(np.diff(trace) >= -1e-3)
                        ),
                    }
                )
                states, transitions = parameter_rows(
                    "cross_validation", subject, num_states, fold, seed, params
                )
                state_rows.extend(states)
                transition_rows.extend(transitions)

    fits = pd.DataFrame(fits)
    seed_scores = (
        fits.groupby(["subject_name", "num_states", "seed"], as_index=False)
        .agg(
            test_log_likelihood=("test_log_likelihood", "sum"),
            n_test_trials=("n_test_trials", "sum"),
        )
        .assign(
            test_log_likelihood_per_trial=lambda frame: (
                frame["test_log_likelihood"] / frame["n_test_trials"]
            )
        )
    )
    summary = (
        seed_scores.groupby(["subject_name", "num_states"], as_index=False)
        .agg(
            mean_test_log_likelihood_per_trial=(
                "test_log_likelihood_per_trial",
                "mean",
            ),
            standard_deviation=("test_log_likelihood_per_trial", "std"),
            n_seeds=("seed", "nunique"),
        )
        .fillna({"standard_deviation": 0.0})
    )
    summary["standard_error"] = summary["standard_deviation"] / np.sqrt(
        summary["n_seeds"]
    )
    best = summary.loc[summary["mean_test_log_likelihood_per_trial"].idxmax()]
    threshold = best["mean_test_log_likelihood_per_trial"] - best["standard_error"]
    selected_k = int(
        summary.loc[
            summary["mean_test_log_likelihood_per_trial"] >= threshold,
            "num_states",
        ].min()
    )
    summary["selected_by_one_standard_error"] = summary["num_states"].eq(selected_k)
    return (
        fits,
        seed_scores,
        summary,
        pd.DataFrame(state_rows),
        pd.DataFrame(transition_rows),
        selected_k,
    )


def fit_selected(subject, sequences, selected_k, args):
    candidates = []
    fit_rows = []
    state_rows = []
    transition_rows = []
    for seed in args.seeds:
        print(f"full K={selected_k} seed={seed}", flush=True)
        model, params, trace = fit_glm_hmm_sessions(
            sequences,
            selected_k,
            seed,
            args.em_iters,
            args.m_step_iters,
        )
        final_ll = session_log_likelihood(model, params, sequences)
        candidates.append(
            (
                final_ll,
                seed,
                model,
                params,
                trace,
            )
        )
        fit_rows.append(
            {
                "subject_name": subject,
                "num_states": selected_k,
                "seed": seed,
                "initial_log_likelihood": float(trace[0]),
                "final_log_likelihood": final_ll,
                "em_trace_non_decreasing": bool(np.all(np.diff(trace) >= -1e-3)),
            }
        )
        states, transitions = parameter_rows(
            "full_data", subject, selected_k, -1, seed, params
        )
        state_rows.extend(states)
        transition_rows.extend(transitions)
    full_ll, seed, model, params, trace = max(candidates, key=lambda item: item[0])
    selected_states, selected_transitions = parameter_rows(
        "full_data", subject, selected_k, -1, seed, params
    )
    return (
        model,
        params,
        seed,
        full_ll,
        trace,
        pd.DataFrame(selected_states),
        pd.DataFrame(selected_transitions),
        pd.DataFrame(fit_rows),
        pd.DataFrame(state_rows),
        pd.DataFrame(transition_rows),
    )


def posterior_table(design, sequences, model, params):
    order, _, _ = ordered_state_parameters(params)
    tables = []
    probabilities = smoothed_session_probabilities(model, params, sequences)
    for (session_name, _, _), posterior in zip(sequences, probabilities):
        trials = design.loc[design["session_name"].astype(str).eq(session_name)].copy()
        posterior = posterior[:, order]
        if len(trials) != len(posterior):
            raise AssertionError(f"Posterior rows do not match {session_name}")
        for state in range(posterior.shape[1]):
            trials[f"state_{state}_probability"] = posterior[:, state]
        trials["inferred_state"] = posterior.argmax(axis=1)
        trials["max_state_probability"] = posterior.max(axis=1)
        trials["posterior_entropy_bits"] = -np.sum(
            posterior * np.log2(np.clip(posterior, 1e-12, 1.0)),
            axis=1,
        )
        for threshold in (0.70, 0.80, 0.90):
            trials[f"state_at_{threshold:.2f}"] = np.where(
                trials["max_state_probability"] >= threshold,
                trials["inferred_state"],
                -1,
            )
        tables.append(trials)
    return pd.concat(tables, ignore_index=True)


def diagnostic_tables(posteriors, state_parameters):
    probability_columns = [
        column
        for column in posteriors.columns
        if column.startswith("state_") and column.endswith("_probability")
    ]
    occupancy = (
        posteriors.groupby(["subject_name", "session_name"], as_index=False)[
            probability_columns
        ]
        .mean()
        .melt(
            id_vars=["subject_name", "session_name"],
            var_name="state",
            value_name="posterior_occupancy",
        )
    )
    occupancy["state"] = occupancy["state"].str.extract(r"state_(\d+)").astype(int)

    indexed = posteriors.copy()
    session_position = indexed.groupby("session_name").cumcount()
    session_size = indexed.groupby("session_name")["trial_num"].transform("size")
    indexed["trial_index_decile"] = np.minimum(
        (10 * session_position / session_size).astype(int) + 1,
        10,
    )
    trial_index = (
        indexed.groupby("trial_index_decile", as_index=False)[probability_columns]
        .mean()
        .melt(
            id_vars="trial_index_decile",
            var_name="state",
            value_name="posterior_occupancy",
        )
    )
    trial_index["state"] = trial_index["state"].str.extract(r"state_(\d+)").astype(int)

    dwell_rows = []
    for session_name, trials in posteriors.groupby("session_name", sort=True):
        for state, run in groupby(trials["inferred_state"].to_numpy()):
            dwell_rows.append(
                {
                    "subject_name": trials["subject_name"].iloc[0],
                    "session_name": session_name,
                    "state": int(state),
                    "dwell_trials": sum(1 for _ in run),
                }
            )
    dwell = pd.DataFrame(dwell_rows)

    evidence = np.linspace(
        posteriors["stimulus_evidence_hz"].quantile(0.01),
        posteriors["stimulus_evidence_hz"].quantile(0.99),
        101,
    )
    psychometric_rows = []
    for state in state_parameters.itertuples():
        right_probability = 1 / (
            1 + np.exp(-(state.intercept + state.stimulus_evidence_hz * evidence))
        )
        psychometric_rows.extend(
            {
                "state": state.ordered_state,
                "stimulus_evidence_hz": x,
                "right_choice_probability": probability,
                "history": "previous choice = 0; previous reward = 0",
            }
            for x, probability in zip(evidence, right_probability)
        )
    return occupancy, trial_index, dwell, pd.DataFrame(psychometric_rows)


def save_figures(
    output_dir,
    subject,
    seed_scores,
    model_summary,
    state_parameters,
    transitions,
    occupancy,
    trial_index,
    dwell,
    psychometric,
    n_trials,
    n_sessions,
    selected_k,
):
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
        }
    )
    colors = plt.get_cmap("tab10").colors

    figure, axis = plt.subplots(figsize=(5.2, 3.6))
    for _, seed in seed_scores.groupby("seed"):
        axis.plot(
            seed["num_states"],
            seed["test_log_likelihood_per_trial"],
            color="0.75",
            marker="o",
            markersize=3,
            linewidth=0.8,
        )
    axis.errorbar(
        model_summary["num_states"],
        model_summary["mean_test_log_likelihood_per_trial"],
        yerr=model_summary["standard_error"],
        color="black",
        marker="o",
        capsize=3,
        linewidth=1.2,
        label="Mean +/- SE across seeds",
    )
    axis.axvline(selected_k, color="0.5", linestyle="--", linewidth=0.8)
    axis.set(
        title="Held-out whole-session choice likelihood by state count",
        xlabel="Number of latent states (K)",
        ylabel="Log likelihood per valid trial (nats)",
        xticks=sorted(model_summary["num_states"].unique()),
    )
    axis.legend(frameon=False)
    axis.text(
        0,
        -0.25,
        f"{subject}; {n_trials:,} valid trials across {n_sessions} sessions; "
        "preliminary state-count audit",
        transform=axis.transAxes,
        fontsize=8,
    )
    figure.savefig(output_dir / "model_selection.pdf", bbox_inches="tight")
    figure.savefig(output_dir / "model_selection.svg", bbox_inches="tight")
    plt.close(figure)

    figure, axes = plt.subplots(2, 3, figsize=(11, 7))
    state_names = sorted(state_parameters["ordered_state"].unique())
    coefficient_names = ["intercept", *MODEL_INPUT_COLUMNS]
    x = np.arange(len(coefficient_names))
    for state in state_names:
        row = state_parameters.loc[state_parameters["ordered_state"].eq(state)].iloc[0]
        axes[0, 0].plot(
            x,
            [row[name] for name in coefficient_names],
            marker="o",
            color=colors[state],
            label=f"State {state}",
        )
    axes[0, 0].axhline(0, color="0.7", linewidth=0.8)
    axes[0, 0].set(
        title="State-specific choice GLM coefficients",
        ylabel="Coefficient (log odds)",
        xticks=x,
        xticklabels=["Bias", "Evidence", "Previous\nchoice", "Previous\nreward"],
    )
    axes[0, 0].legend(frameon=False)

    for state, values in psychometric.groupby("state"):
        axes[0, 1].plot(
            values["stimulus_evidence_hz"],
            values["right_choice_probability"],
            color=colors[state],
            label=f"State {state}",
        )
    axes[0, 1].axhline(0.5, color="0.75", linewidth=0.8)
    axes[0, 1].axvline(0, color="0.75", linewidth=0.8)
    axes[0, 1].set(
        title="State-specific psychometric functions",
        xlabel="Signed visual evidence (Hz)",
        ylabel="Predicted P(right choice)",
        ylim=(0, 1),
    )

    transition_matrix = transitions.pivot(
        index="from_state", columns="to_state", values="probability"
    ).to_numpy()
    image = axes[0, 2].imshow(
        transition_matrix,
        vmin=0,
        vmax=1,
        cmap="Greys",
        aspect="equal",
    )
    for source in state_names:
        for target in state_names:
            axes[0, 2].text(
                target,
                source,
                f"{transition_matrix[source, target]:.2f}",
                ha="center",
                va="center",
                color=(
                    "white" if transition_matrix[source, target] > 0.55 else "black"
                ),
            )
    axes[0, 2].set(
        title="Stationary transition matrix",
        xlabel="Next state",
        ylabel="Current state",
        xticks=state_names,
        yticks=state_names,
    )
    figure.colorbar(image, ax=axes[0, 2], fraction=0.046, pad=0.04)

    for state, values in dwell.groupby("state"):
        ordered = np.sort(values["dwell_trials"].to_numpy())
        axes[1, 0].plot(
            ordered,
            np.arange(1, len(ordered) + 1) / len(ordered),
            color=colors[state],
            label=f"State {state}",
        )
    axes[1, 0].set(
        title="Most-likely-state dwell distribution",
        xlabel="Consecutive trials",
        ylabel="Empirical cumulative probability",
        ylim=(0, 1),
    )

    rng = np.random.default_rng(445)
    for state, values in occupancy.groupby("state"):
        jitter = rng.normal(0, 0.04, len(values))
        axes[1, 1].scatter(
            state + jitter,
            values["posterior_occupancy"],
            color=colors[state],
            alpha=0.55,
            s=10,
        )
        axes[1, 1].plot(
            [state - 0.18, state + 0.18],
            [values["posterior_occupancy"].median()] * 2,
            color="black",
            linewidth=1.3,
        )
    axes[1, 1].set(
        title="Session-level state occupancy",
        xlabel="Ordered state",
        ylabel="Mean posterior probability",
        xticks=state_names,
        ylim=(0, 1),
    )

    for state, values in trial_index.groupby("state"):
        axes[1, 2].plot(
            values["trial_index_decile"],
            values["posterior_occupancy"],
            marker="o",
            color=colors[state],
            label=f"State {state}",
        )
    axes[1, 2].set(
        title="State occupancy across session progress",
        xlabel="Within-session trial-index decile",
        ylabel="Mean posterior probability",
        xticks=np.arange(1, 11),
        ylim=(0, 1),
    )

    figure.suptitle(
        f"Preliminary behavior-only state diagnostics: {subject} (K={selected_k})",
        fontsize=12,
    )
    figure.text(
        0.5,
        0.01,
        f"{n_trials:,} valid trials; {n_sessions} sessions; forward-backward "
        "posteriors; psychometric curves fix history at zero",
        ha="center",
        fontsize=8,
    )
    figure.tight_layout(rect=(0, 0.04, 1, 0.95))
    figure.savefig(output_dir / "state_diagnostics.pdf", bbox_inches="tight")
    figure.savefig(output_dir / "state_diagnostics.svg", bbox_inches="tight")
    plt.close(figure)


def simulate_sessions(seed=445, n_sessions=6, trials_per_session=200):
    rng = np.random.default_rng(seed)
    transition = np.array([[0.97, 0.03], [0.05, 0.95]])
    coefficients = np.array(
        [
            [0.0, 2.5, 0.25, 0.20],
            [1.5, 0.1, -0.20, -0.10],
        ]
    )
    sequences = []
    true_states = {}
    for session_index in range(n_sessions):
        states = np.empty(trials_per_session, dtype=np.int32)
        choices = np.empty(trials_per_session, dtype=np.int32)
        inputs = np.zeros((trials_per_session, 3), dtype=np.float32)
        state = rng.integers(2)
        previous_choice = 0.0
        previous_reward = 0.0
        for trial in range(trials_per_session):
            if trial:
                state = rng.choice(2, p=transition[state])
            inputs[trial] = [rng.normal(), previous_choice, previous_reward]
            linear = coefficients[state, 0] + coefficients[state, 1:] @ inputs[trial]
            choice = int(rng.random() < 1 / (1 + np.exp(-linear)))
            reward = int(rng.random() < (0.75 if choice == state else 0.35))
            states[trial] = state
            choices[trial] = choice
            previous_choice = 2 * choice - 1
            previous_reward = reward
        name = f"synthetic_{session_index:02d}"
        sequences.append((name, choices, inputs))
        true_states[name] = states
    return sequences, true_states, coefficients, transition


def run_synthetic(args):
    sequences, true_states, true_coefficients, true_transition = simulate_sessions()
    rows = []
    for num_states in (1, 2):
        for seed in args.seeds:
            print(f"synthetic K={num_states} seed={seed}", flush=True)
            model, params, trace = fit_glm_hmm_sessions(
                sequences,
                num_states,
                seed,
                args.em_iters,
                args.m_step_iters,
            )
            order, coefficients, transition = ordered_state_parameters(params)
            accuracy = np.nan
            if num_states == 2:
                probabilities = smoothed_session_probabilities(model, params, sequences)
                inferred = np.concatenate(
                    [posterior[:, order].argmax(axis=1) for posterior in probabilities]
                )
                truth = np.concatenate([true_states[name] for name, _, _ in sequences])
                accuracy = float((inferred == truth).mean())
            rows.append(
                {
                    "num_states": num_states,
                    "seed": seed,
                    "n_trials": sum(len(item[1]) for item in sequences),
                    "log_likelihood": session_log_likelihood(model, params, sequences),
                    "log_likelihood_per_trial": session_log_likelihood(
                        model, params, sequences
                    )
                    / sum(len(item[1]) for item in sequences),
                    "state_accuracy": accuracy,
                    "parameter_rmse": (
                        float(np.sqrt(np.mean((coefficients - true_coefficients) ** 2)))
                        if num_states == 2
                        else np.nan
                    ),
                    "transition_rmse": (
                        float(np.sqrt(np.mean((transition - true_transition) ** 2)))
                        if num_states == 2
                        else np.nan
                    ),
                    "em_trace_non_decreasing": bool(np.all(np.diff(trace) >= -1e-3)),
                }
            )
    results = pd.DataFrame(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.output_dir / "synthetic_recovery.csv", index=False)
    save_synthetic_figure(args.output_dir, results)
    best_k1 = results.loc[results["num_states"].eq(1), "log_likelihood_per_trial"].max()
    best_k2 = results.loc[results["num_states"].eq(2), "log_likelihood_per_trial"].max()
    best_accuracy = results.loc[results["num_states"].eq(2), "state_accuracy"].max()
    verdict = (
        "Synthetic recovery check only. Across the requested initializations, "
        f"the best K=2 fit improved log likelihood by {best_k2 - best_k1:.6f} "
        f"nats/trial over K=1 and recovered ordered state labels at "
        f"{best_accuracy:.1%} accuracy. This validates the session-reset fitting "
        "path; it does not select K for GRB006.\n"
    )
    (args.output_dir / "synthetic_recovery_verdict.txt").write_text(
        verdict,
        encoding="utf-8",
    )
    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "n_sessions": len(sequences),
        "n_trials": sum(len(item[1]) for item in sequences),
        "true_coefficients": true_coefficients.tolist(),
        "true_transition_matrix": true_transition.tolist(),
        "best_k2_state_accuracy": float(
            results.loc[results["num_states"].eq(2), "state_accuracy"].max()
        ),
        "best_k2_parameter_rmse": float(
            results.loc[results["num_states"].eq(2), "parameter_rmse"].min()
        ),
    }
    (args.output_dir / "synthetic_recovery_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    print(results.to_string(index=False))


def save_synthetic_figure(output_dir, results):
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
        }
    )
    figure, axes = plt.subplots(1, 2, figsize=(8, 3.4))
    for seed, values in results.groupby("seed"):
        axes[0].plot(
            values["num_states"],
            values["log_likelihood_per_trial"],
            color="0.55",
            marker="o",
            linewidth=0.8,
            label=f"Seed {seed}",
        )
    axes[0].set(
        title="Synthetic choice likelihood by fitted state count",
        xlabel="Number of latent states (K)",
        ylabel="Log likelihood per trial (nats)",
        xticks=[1, 2],
    )
    axes[0].legend(frameon=False)

    recovered = results.loc[results["num_states"].eq(2)]
    axes[1].scatter(
        recovered["seed"],
        recovered["state_accuracy"],
        color="black",
        s=25,
    )
    axes[1].axhline(0.5, color="0.65", linestyle="--", linewidth=0.8)
    axes[1].set(
        title="Two-state label recovery across initializations",
        xlabel="Initialization seed",
        ylabel="Ordered-state accuracy",
        ylim=(0, 1),
        xticks=sorted(recovered["seed"].unique()),
    )
    figure.suptitle("Synthetic persistent-state recovery check", fontsize=12)
    figure.text(
        0.5,
        0.01,
        f"{results['n_trials'].iloc[0]:,} trials across 6 separate sessions; "
        "preliminary implementation check",
        ha="center",
        fontsize=8,
    )
    figure.tight_layout(rect=(0, 0.05, 1, 0.93))
    figure.savefig(output_dir / "synthetic_recovery.pdf", bbox_inches="tight")
    figure.savefig(output_dir / "synthetic_recovery.svg", bbox_inches="tight")
    plt.close(figure)


def run_subject(args):
    classified = classify_trials(fetch_trials([args.subject]))
    design = build_design_matrix(classified)
    sequences = session_sequences(design)
    if len(design) < MIN_VALID_TRIALS:
        raise ValueError(
            f"{args.subject} has {len(design)} valid trials; need {MIN_VALID_TRIALS}"
        )
    if len(sequences) < max(MIN_SESSIONS_FOR_CV, args.folds):
        raise ValueError(
            f"{args.subject} has {len(sequences)} sessions; need at least "
            f"{max(MIN_SESSIONS_FOR_CV, args.folds)}"
        )

    (
        fits,
        seed_scores,
        model_summary,
        cv_states,
        cv_transitions,
        selected_k,
    ) = cross_validate(args.subject, sequences, args)
    (
        model,
        params,
        selected_seed,
        full_ll,
        trace,
        selected_states,
        selected_transitions,
        full_seed_fits,
        full_seed_states,
        full_seed_transitions,
    ) = fit_selected(args.subject, sequences, selected_k, args)
    posteriors = posterior_table(design, sequences, model, params)
    occupancy, trial_index, dwell, psychometric = diagnostic_tables(
        posteriors, selected_states
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "cross_validation_fits.csv": fits,
        "cross_validation_seed_scores.csv": seed_scores,
        "model_selection.csv": model_summary,
        "cross_validation_state_parameters.csv": cv_states,
        "cross_validation_transition_matrices.csv": cv_transitions,
        "selected_state_parameters.csv": selected_states,
        "selected_transition_matrix.csv": selected_transitions,
        "full_data_fits.csv": full_seed_fits,
        "full_data_state_parameters.csv": full_seed_states,
        "full_data_transition_matrices.csv": full_seed_transitions,
        "trial_state_posteriors.csv": posteriors,
        "session_state_occupancy.csv": occupancy,
        "trial_index_state_occupancy.csv": trial_index,
        "state_dwell_times.csv": dwell,
        "state_psychometric_curves.csv": psychometric,
    }
    for name, table in outputs.items():
        table.to_csv(args.output_dir / name, index=False)
    save_figures(
        args.output_dir,
        args.subject,
        seed_scores,
        model_summary,
        selected_states,
        selected_transitions,
        occupancy,
        trial_index,
        dwell,
        psychometric,
        len(design),
        len(sequences),
        selected_k,
    )

    threshold_counts = {
        f"{threshold:.2f}": int(
            (posteriors["max_state_probability"] >= threshold).sum()
        )
        for threshold in (0.70, 0.80, 0.90)
    }
    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "task": "LAB-TASKS-445",
        "subject_name": args.subject,
        "git_branch": git_value("branch", "--show-current"),
        "git_commit": git_value("rev-parse", "HEAD"),
        "n_sessions": len(sequences),
        "n_trials": len(design),
        "candidate_states": args.states,
        "seeds": args.seeds,
        "n_folds": args.folds,
        "cv_seed": args.cv_seed,
        "em_iterations": args.em_iters,
        "m_step_iterations": args.m_step_iters,
        "model_selection_policy": (
            "smallest K within one standard error of the best mean held-out "
            "whole-session log likelihood per trial across seeds"
        ),
        "selected_k": selected_k,
        "selected_seed": selected_seed,
        "full_data_log_likelihood": full_ll,
        "full_data_trace_non_decreasing": bool(np.all(np.diff(trace) >= -1e-3)),
        "posterior_method": "Dynamax forward-backward smoothing",
        "posterior_threshold_counts": threshold_counts,
        "mean_posterior_entropy_bits": float(
            posteriors["posterior_entropy_bits"].mean()
        ),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    selected_score = model_summary.loc[
        model_summary["num_states"].eq(selected_k),
        "mean_test_log_likelihood_per_trial",
    ].item()
    baseline_score = model_summary.loc[
        model_summary["num_states"].eq(1),
        "mean_test_log_likelihood_per_trial",
    ].item()
    verdict = (
        f"Preliminary state-count audit only. Whole-session validation selected "
        f"K={selected_k} by the one-standard-error rule; its mean held-out log "
        f"likelihood was {selected_score:.6f} nats/trial versus {baseline_score:.6f} "
        "for K=1. Review repeated-fit parameter spread, posterior entropy, occupancy, "
        "and dwell diagnostics before design lock. This audit does not make the "
        "neural-decoding claim.\n"
    )
    (args.output_dir / "verdict.txt").write_text(verdict, encoding="utf-8")
    print(model_summary.to_string(index=False))
    print(f"\nSelected K={selected_k}, seed={selected_seed}")


def main() -> None:
    args = parse_args()
    if args.synthetic_recovery:
        run_synthetic(args)
    else:
        run_subject(args)


if __name__ == "__main__":
    main()
