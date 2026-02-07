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
def _(DecisionTask, pd):
    mice = "GRB045"
    ses_back = 10
    # Optional: Add session date filter here
    # Example: session_date_filter = 'session_datetime >= "2024-01-01"'
    session_date_filter = (
        None  # Set to None to include all sessions, or add a date filter string
    )

    # Fetch data from database with optional session date filter
    base_query = f"subject_name = '{mice}'"
    if session_date_filter:
        full_query = base_query + f" AND {session_date_filter}"
    else:
        full_query = base_query

    data = pd.DataFrame(DecisionTask.TrialSet() & full_query).tail(ses_back)
    sesdata = data[data.session_name == data.session_name.iloc[-1]]
    return data, ses_back, session_date_filter, sesdata


@app.cell
def _(data, np):
    ew_rate = []
    for ses in data.itertuples(index=False):
        ew_trials = ~(np.isin(np.array(ses.response_values), [-1, 1]))
        rate = (ew_trials).sum() / ew_trials.shape[0] if ew_trials.shape[0] > 0 else 0
        ew_rate.append(rate)
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
    # Use actual number of sessions in data, not ses_back (in case filter reduces count)
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
