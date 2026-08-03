"""Behavior-only data contract for GLM-HMM engagement-state inference."""

from __future__ import annotations

import numpy as np
import pandas as pd


MIN_VALID_TRIALS = 3_000
MIN_SESSIONS_FOR_CV = 5
TASK_MODE = "discrimination"
BOUNDARY_ATOL_HZ = 1e-9
MODEL_INPUT_COLUMNS = (
    "stimulus_evidence_hz",
    "previous_choice",
    "previous_reward",
)

EXCLUSION_REASONS = (
    "early_withdrawal",
    "missing_choice",
    "no_choice",
    "missing_stimulus",
    "missing_reward",
)

REQUIRED_COLUMNS = {
    "subject_name",
    "session_name",
    "dataset_name",
    "trial_num",
    "response",
    "with_choice",
    "early_withdrawal",
    "rewarded",
    "stim_rate_vision",
    "category_boundary",
}


def classify_trials(trials: pd.DataFrame) -> pd.DataFrame:
    """Apply mutually exclusive exclusions and retain valid binary choices."""
    missing = REQUIRED_COLUMNS.difference(trials.columns)
    if missing:
        raise ValueError(f"Missing trial columns: {sorted(missing)}")

    out = trials.copy()
    response = pd.to_numeric(out["response"], errors="coerce")
    stimulus = pd.to_numeric(out["stim_rate_vision"], errors="coerce")
    boundary = pd.to_numeric(out["category_boundary"], errors="coerce")
    rewarded = pd.to_numeric(out["rewarded"], errors="coerce")
    early = pd.to_numeric(out["early_withdrawal"], errors="coerce").eq(1)
    with_choice = pd.to_numeric(out["with_choice"], errors="coerce").eq(1)

    reason = pd.Series("valid", index=out.index, dtype="object")
    reason.loc[early] = "early_withdrawal"
    reason.loc[reason.eq("valid") & response.isna()] = "missing_choice"
    reason.loc[reason.eq("valid") & ~(with_choice & response.isin([-1, 1]))] = (
        "no_choice"
    )
    reason.loc[
        reason.eq("valid") & (~np.isfinite(stimulus) | ~np.isfinite(boundary))
    ] = "missing_stimulus"
    reason.loc[reason.eq("valid") & ~np.isfinite(rewarded)] = "missing_reward"

    out["response"] = response
    out["rewarded"] = rewarded
    out["stim_rate_vision"] = stimulus
    out["category_boundary"] = boundary
    out["exclusion_reason"] = reason
    out["is_valid"] = reason.eq("valid")
    out["is_boundary"] = out["is_valid"] & np.isclose(
        stimulus - boundary, 0.0, atol=BOUNDARY_ATOL_HZ
    )
    return out


def build_design_matrix(classified_trials: pd.DataFrame) -> pd.DataFrame:
    """Build the locked behavior matrix, resetting history at each session."""
    valid = classified_trials.loc[classified_trials["is_valid"]].copy()
    valid = valid.sort_values(
        ["subject_name", "session_name", "dataset_name", "trial_num"]
    ).reset_index(drop=True)

    datasets_per_session = valid.groupby(
        ["subject_name", "session_name"]
    ).dataset_name.nunique()
    if (datasets_per_session > 1).any():
        sessions = datasets_per_session[datasets_per_session > 1].index.tolist()
        raise ValueError(f"Multiple datasets in a session: {sessions[:5]}")

    session = valid.groupby(["subject_name", "session_name"], sort=False)
    valid["intercept"] = 1.0
    valid["stimulus_evidence_hz"] = (
        valid["stim_rate_vision"] - valid["category_boundary"]
    )
    valid["choice_right"] = valid["response"].eq(1).astype(int)
    valid["previous_choice"] = session["response"].shift().fillna(0.0)
    valid["previous_reward"] = session["rewarded"].shift().fillna(0.0)
    valid["sequence_start"] = session.cumcount().eq(0)

    design_columns = [
        "intercept",
        "stimulus_evidence_hz",
        "previous_choice",
        "previous_reward",
        "choice_right",
    ]
    if not np.isfinite(valid[design_columns].to_numpy(dtype=float)).all():
        raise ValueError("Non-finite value in engagement-state design matrix")
    starts = valid.loc[valid["sequence_start"]]
    if not (
        starts["previous_choice"].eq(0).all() and starts["previous_reward"].eq(0).all()
    ):
        raise AssertionError("History did not reset at every session boundary")
    return valid


def summarize_feasibility(
    classified_trials: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return per-session and per-subject exclusion/feasibility counts."""
    counted = classified_trials.copy()
    for reason in ("valid", *EXCLUSION_REASONS):
        counted[f"n_{reason}"] = counted["exclusion_reason"].eq(reason)
    counted["n_boundary_valid"] = counted["is_boundary"]

    count_columns = [
        "n_valid",
        *(f"n_{reason}" for reason in EXCLUSION_REASONS),
        "n_boundary_valid",
    ]
    session = (
        counted.groupby(["subject_name", "session_name"], as_index=False)
        .agg(
            n_trials=("trial_num", "size"),
            n_datasets=("dataset_name", "nunique"),
            **{name: (name, "sum") for name in count_columns},
        )
        .sort_values(["subject_name", "session_name"])
    )
    session["n_nonboundary_valid"] = session["n_valid"] - session["n_boundary_valid"]

    subject = (
        session.groupby("subject_name", as_index=False)
        .agg(
            n_sessions=("session_name", "nunique"),
            n_sessions_with_valid=("n_valid", lambda values: int((values > 0).sum())),
            n_trials=("n_trials", "sum"),
            **{name: (name, "sum") for name in count_columns},
            n_nonboundary_valid=("n_nonboundary_valid", "sum"),
        )
        .sort_values(["n_valid", "subject_name"], ascending=[False, True])
    )
    subject["meets_trial_gate"] = subject["n_valid"] >= MIN_VALID_TRIALS
    subject["meets_session_gate"] = (
        subject["n_sessions_with_valid"] >= MIN_SESSIONS_FOR_CV
    )
    subject["feasible_for_k_comparison"] = (
        subject["meets_trial_gate"] & subject["meets_session_gate"]
    )
    return session, subject


def session_sequences(
    design: pd.DataFrame,
) -> list[tuple[str, np.ndarray, np.ndarray]]:
    """Return one binary-choice GLM-HMM sequence per session."""
    subjects = design["subject_name"].unique()
    if len(subjects) != 1:
        raise ValueError("Fit one subject at a time")

    sequences = []
    for session_name, trials in design.groupby("session_name", sort=True):
        choices = trials["choice_right"].to_numpy(dtype=np.int32)
        inputs = trials[list(MODEL_INPUT_COLUMNS)].to_numpy(dtype=np.float32)
        sequences.append((str(session_name), choices, inputs))
    return sequences


def session_folds(
    sequences: list[tuple[str, np.ndarray, np.ndarray]],
    n_splits: int = 5,
    seed: int = 0,
) -> list[tuple[list[int], list[int]]]:
    """Split complete sessions into deterministic cross-validation folds."""
    if not 2 <= n_splits <= len(sequences):
        raise ValueError("n_splits must be between 2 and the number of sessions")

    shuffled = np.random.default_rng(seed).permutation(len(sequences))
    test_folds = np.array_split(shuffled, n_splits)
    all_indices = np.arange(len(sequences))
    return [
        (
            np.setdiff1d(all_indices, test_indices).tolist(),
            test_indices.tolist(),
        )
        for test_indices in test_folds
    ]


def fit_glm_hmm_sessions(
    sequences: list[tuple[str, np.ndarray, np.ndarray]],
    num_states: int,
    seed: int,
    num_em_iters: int = 50,
    m_step_num_iters: int = 50,
):
    """Fit a Dynamax categorical GLM-HMM to variable-length sessions."""
    import jax.numpy as jnp
    import jax.random as jr
    from dynamax.hidden_markov_model import CategoricalRegressionHMM

    if not sequences:
        raise ValueError("At least one session is required")

    model = CategoricalRegressionHMM(
        num_states=num_states,
        num_classes=2,
        input_dim=len(MODEL_INPUT_COLUMNS),
        m_step_num_iters=m_step_num_iters,
    )
    if num_states == 1:
        initialize = {
            "initial_probs": jnp.ones(1),
            "transition_matrix": jnp.ones((1, 1)),
        }
    else:
        initialize = {
            "initial_probs": jnp.full(num_states, 1 / num_states),
            "transition_matrix": jnp.full(
                (num_states, num_states),
                0.05 / (num_states - 1),
            )
            .at[jnp.diag_indices(num_states)]
            .set(0.95),
        }
    params, props = model.initialize(key=jr.key(seed), **initialize)
    m_step_state = model.initialize_m_step_state(params, props)

    log_probs = []
    for _ in range(num_em_iters):
        posterior, choices, inputs, boundaries, starts = _block_posterior(
            model, params, sequences
        )
        keep_transition = (
            jnp.ones(len(choices) - 1, dtype=bool).at[boundaries].set(False)
        )
        initial_stats = posterior.smoothed_probs[starts]
        transition_stats = posterior.trans_probs[keep_transition].sum(axis=0)[None, ...]
        emission_stats = (
            posterior.smoothed_probs[None, ...],
            choices[None, ...],
            inputs[None, ...],
        )
        log_probs.append(float(posterior.marginal_loglik))
        params, m_step_state = model.m_step(
            params,
            props,
            (initial_stats, transition_stats, emission_stats),
            m_step_state,
        )
    return model, params, np.asarray(log_probs)


def _block_posterior(model, params, sequences):
    """Run one exact Dynamax smoother with explicit session resets."""
    import jax.numpy as jnp
    from dynamax.hidden_markov_model.inference import hmm_two_filter_smoother

    lengths = np.asarray([len(sequence[1]) for sequence in sequences])
    if (lengths < 1).any():
        raise ValueError("Every session must contain at least one trial")

    choices = jnp.concatenate([jnp.asarray(sequence[1]) for sequence in sequences])
    inputs = jnp.concatenate([jnp.asarray(sequence[2]) for sequence in sequences])
    starts = jnp.asarray(np.r_[0, np.cumsum(lengths)[:-1]])
    boundaries = jnp.asarray(np.cumsum(lengths)[:-1] - 1)

    transition = params.transitions.transition_matrix
    transitions = jnp.broadcast_to(transition, (len(choices) - 1, *transition.shape))
    reset = jnp.broadcast_to(params.initial.probs, transition.shape)
    transitions = transitions.at[boundaries].set(reset)
    log_likelihoods = model.emission_component._compute_conditional_logliks(
        params.emissions, choices, inputs
    )
    posterior = hmm_two_filter_smoother(
        params.initial.probs,
        transitions,
        log_likelihoods,
    )
    return posterior, choices, inputs, boundaries, starts


def session_log_likelihood(model, params, sequences) -> float:
    """Sum marginal log likelihoods without joining session boundaries."""
    posterior, *_ = _block_posterior(model, params, sequences)
    return float(posterior.marginal_loglik)


def smoothed_session_probabilities(model, params, sequences) -> list[np.ndarray]:
    """Return exact smoothed probabilities split back into sessions."""
    posterior, *_ = _block_posterior(model, params, sequences)
    offsets = np.r_[0, np.cumsum([len(sequence[1]) for sequence in sequences])]
    probabilities = np.asarray(posterior.smoothed_probs)
    return [probabilities[start:stop] for start, stop in zip(offsets[:-1], offsets[1:])]


def ordered_state_parameters(params) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return identifiable binary-logit coefficients in a stable state order."""
    weights = np.asarray(params.emissions.weights)
    biases = np.asarray(params.emissions.biases)
    coefficients = weights[:, 1] - weights[:, 0]
    intercepts = biases[:, 1] - biases[:, 0]
    order = np.lexsort((-intercepts, -coefficients[:, 0]))
    transition_matrix = np.asarray(params.transitions.transition_matrix)
    return (
        order,
        np.column_stack([intercepts[order], coefficients[order]]),
        transition_matrix[np.ix_(order, order)],
    )
