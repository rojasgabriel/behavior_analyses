import marimo

__generated_with = "0.19.7"
app = marimo.App()


@app.cell
def _():
    from labdata.schema import DecisionTask  # type: ignore
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt

    return DecisionTask, np, pd, plt


@app.cell
def _(DecisionTask, np, pd):
    def calculate_session_ew_rate(response_values):
        """Calculate early withdrawal rate for a session."""
        is_ew_trial = ~(np.isin(np.array(response_values), [-1, 1]))
        return (
            (is_ew_trial).sum() / is_ew_trial.shape[0]
            if is_ew_trial.shape[0] > 0
            else 0
        )

    mice = "GRB045"
    ses_back = 10
    data_full = pd.DataFrame(DecisionTask.TrialSet() & f"subject_name = '{mice}'").tail(
        ses_back
    )

    # Apply session filter to remove poor quality sessions
    # Calculate early withdrawal rate for each session to use as a quality filter
    session_ew_rates = [
        calculate_session_ew_rate(ses.response_values) for ses in data_full.itertuples()
    ]

    # Add early withdrawal rate to dataframe for filtering
    data_with_quality_metrics = data_full.copy()
    data_with_quality_metrics["ew_rate_session"] = session_ew_rates

    # Apply session filter: exclude sessions with >80% early withdrawal rate (can be adjusted)
    # This ensures multisession graphs only include valid sessions
    ew_threshold = 0.8
    data = data_with_quality_metrics[
        data_with_quality_metrics["ew_rate_session"] <= ew_threshold
    ].copy()

    sesdata = data[data.session_name == data.session_name.iloc[-1]]
    return calculate_session_ew_rate, data, data_full, ew_threshold, ses_back, sesdata


@app.cell
def _(calculate_session_ew_rate, data):
    # Recalculate early withdrawal rate for filtered sessions
    ew_rate = [
        calculate_session_ew_rate(ses.response_values) for ses in data.itertuples()
    ]
    return (ew_rate,)


@app.cell
def _(np, sesdata):
    valid_trials = sesdata.response_values.apply(lambda x: np.isin(x, [-1, 1])).values[
        0
    ]  # removing early withdrawal trials
    stims = sesdata.intensity_values.values[0][valid_trials]
    responses = np.array(sesdata.response_values.values[0])[
        valid_trials
    ]  # this one is saved as a list in the database
    correct = sesdata.correct_values.values[0][valid_trials]
    react_times = sesdata.reaction_times.values[0][valid_trials]
    react_times = react_times[react_times < 2]

    unique_stims = np.unique(sesdata.intensity_values.values[0])
    frac_correct = []
    p_right = []
    for ustim in unique_stims:
        mask = stims == ustim
        right_mask = responses[mask] == 1
        frac_correct.append(correct[mask].sum() / correct[mask].shape[0])
        p_right.append(right_mask.sum() / right_mask.shape[0])
    return frac_correct, p_right, react_times, unique_stims


@app.cell
def _(
    data,
    ew_rate,
    frac_correct,
    np,
    p_right,
    plt,
    react_times,
    ses_back,
    sesdata,
    unique_stims,
):
    fig, ax = plt.subplots(3, 2, figsize=(6, 7))

    ### fraction correct per stim ###
    ax[0, 0].plot(unique_stims, frac_correct, "o-", color="dodgerblue")
    ax[0, 0].set_xlabel("stim intensities")
    ax[0, 0].set_ylabel("fraction correct")
    ax[0, 0].set_ylim((0.3, 1))

    ### p(right) per stim ###
    ax[1, 0].plot(unique_stims, p_right, "o-", color="dodgerblue")
    ax[1, 0].set_xlabel("stim intensities")
    ax[1, 0].set_ylabel("p(right)")
    ax[1, 0].set_ylim((0, 1))

    ### performance on easy trials ###
    # Use actual number of sessions in filtered data, not ses_back
    n_sessions = len(data)
    xvalues = np.arange(0, n_sessions, 1)
    ax[0, 1].plot(
        xvalues,
        data["performance_easy"][::-1],
        color="violet",
        marker="o",
        linestyle="-",
        label="easy",
        alpha=0.8,
    )
    ax[0, 1].plot(
        xvalues,
        data["performance"][::-1],
        color="grey",
        marker="o",
        linestyle="-",
        label="all",
        alpha=0.8,
    )
    ax[0, 1].set_ylabel("performance")
    ax[0, 1].set_xlabel("sessions back")
    ax[0, 1].legend()
    ax[0, 1].set_ylim((0.3, 1))

    ### early withdrawal rate ###
    ax[1, 1].hlines(0.5, xvalues[0], xvalues[-1], color="k", linestyle="--")
    ax[1, 1].plot(xvalues, ew_rate, color="crimson", marker="o", alpha=0.8)
    ax[1, 1].set_xlabel("sessions back")
    ax[1, 1].set_ylabel("e.w. rate")
    ax[1, 1].set_ylim([0, 1])

    ### reaction times ###
    ax[2, 0].hist(react_times, bins=20, color="dodgerblue", alpha=0.8)
    ax[2, 0].set_xlabel("reaction times")
    ax[2, 0].set_ylabel("count")

    ### trial counts ###
    ax[2, 1].bar(
        xvalues, data["n_trials"][::-1], color="grey", label="total", alpha=0.8
    )
    ax[2, 1].bar(
        xvalues,
        data["n_with_choice"][::-1],
        color="dodgerblue",
        label="with choice",
        alpha=0.8,
    )
    ax[2, 1].bar(
        xvalues, data["n_correct"][::-1], color="deeppink", label="correct", alpha=0.8
    )
    ax[2, 1].set_ylabel("trials")
    ax[2, 1].set_xlabel("sessions back")
    ax[2, 1].set_xticks(xvalues[::2])
    ax[2, 1].legend()

    fig.suptitle(
        f"{sesdata.subject_name.values[0]}\n{sesdata.session_name.values[0]}",
        x=0.1,
        ha="left",
    )
    plt.tight_layout()
    plt.show()
    fig.savefig("/Users/gabriel/Downloads/perf_summary_labdata2.png")
    return


if __name__ == "__main__":
    app.run()
