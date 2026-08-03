"""Audit the behavior contract and trial-count gate for LAB-TASKS-445."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess

import pandas as pd

import _bootstrap  # noqa: F401

from behavior_analyses.engagement_states import (
    BOUNDARY_ATOL_HZ,
    MIN_SESSIONS_FOR_CV,
    MIN_VALID_TRIALS,
    TASK_MODE,
    build_design_matrix,
    classify_trials,
    summarize_feasibility,
)
from behavior_analyses.io import get_chipmunk_table


REPO_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--subjects", nargs="+")
    return parser.parse_args()


def fetch_trials(subjects: list[str] | None = None) -> pd.DataFrame:
    Chipmunk = get_chipmunk_table()
    relation = (
        (Chipmunk() & {"setting_task_mode": TASK_MODE})
        * Chipmunk.Trial()
        * Chipmunk.TrialParameters()
    )
    if subjects:
        relation &= [{"subject_name": subject} for subject in subjects]

    fields = [
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
    ]
    return pd.DataFrame(dict(zip(fields, relation.fetch(*fields))))


def git_value(*args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else "unavailable"


def contract(
    classified: pd.DataFrame, design: pd.DataFrame, subject: pd.DataFrame
) -> dict:
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "task": "LAB-TASKS-445",
        "source": {
            "repository": REPO_ROOT.name,
            "git_branch": git_value("branch", "--show-current"),
            "git_commit": git_value("rev-parse", "HEAD"),
            "working_tree_dirty": bool(git_value("status", "--porcelain")),
            "task_mode": TASK_MODE,
        },
        "feasibility": {
            "minimum_valid_trials_per_mouse": MIN_VALID_TRIALS,
            "minimum_sessions_for_whole_session_cv": MIN_SESSIONS_FOR_CV,
            "n_subjects": int(subject.shape[0]),
            "n_feasible_subjects": int(subject["feasible_for_k_comparison"].sum()),
            "n_trials": int(classified.shape[0]),
            "n_valid_trials": int(classified["is_valid"].sum()),
            "n_design_rows": int(design.shape[0]),
        },
        "design_matrix": {
            "sample_axis": "one row per retained binary-choice trial",
            "sequences": "one sequence per mouse and session",
            "outcome": "choice_right: 0=left response (-1), 1=right response (+1)",
            "inputs": [
                "intercept: Dynamax emission bias",
                "stimulus_evidence_hz: stim_rate_vision - category_boundary",
                "previous_choice: previous retained response, coded -1/0/+1",
                "previous_reward: previous retained reward, coded 0/1",
            ],
            "history_policy": (
                "carry the previous retained valid trial through excluded trials; "
                "set previous_choice and previous_reward to 0 at every session start"
            ),
            "boundary_policy": (
                f"include boundary trials as 0 Hz evidence "
                f"(absolute tolerance {BOUNDARY_ATOL_HZ:g} Hz)"
            ),
            "exclusions_are_mutually_exclusive": [
                "early_withdrawal",
                "missing_choice",
                "no_choice",
                "missing_stimulus",
                "missing_reward",
            ],
        },
        "paper_departures": [
            (
                "Use separate previous-choice and previous-reward covariates instead "
                "of Ashwood's previous-choice and win-stay/lose-switch terms; locked "
                "in the Notion task."
            ),
            (
                "Remove invalid-choice trials before fitting instead of retaining "
                "violation trials with masked emissions; locked in the Notion task."
            ),
            (
                "Use neutral zero history at each session start rather than copying "
                "or sampling an initial previous choice."
            ),
        ],
        "implementation_gate": {
            "candidate": "Dynamax CategoricalRegressionHMM",
            "status": (
                "installed; variable-length sessions use Dynamax forward-backward "
                "E-steps and pooled native M-steps"
            ),
            "rejected_first_pass": (
                "Ashwood's zashwood/ssm fork is paper-faithful but minimally "
                "supported and has no release"
            ),
        },
    }


def main() -> None:
    args = parse_args()
    classified = classify_trials(fetch_trials(args.subjects))
    design = build_design_matrix(classified)
    session, subject = summarize_feasibility(classified)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    session_path = args.output_dir / "session_feasibility.csv"
    subject_path = args.output_dir / "subject_feasibility.csv"
    contract_path = args.output_dir / "design_matrix_contract.json"
    session.to_csv(session_path, index=False)
    subject.to_csv(subject_path, index=False)
    contract_path.write_text(
        json.dumps(contract(classified, design, subject), indent=2) + "\n",
        encoding="utf-8",
    )

    print(subject.to_string(index=False))
    print(f"\nWrote {session_path}")
    print(f"Wrote {subject_path}")
    print(f"Wrote {contract_path}")


if __name__ == "__main__":
    main()
