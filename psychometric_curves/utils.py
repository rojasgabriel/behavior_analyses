import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
from djchurchland.chipmunk.task import Chipmunk
from djchurchland.chipmunk.psychometric import PsychometricFit
from djchurchland.chipmunk.fit_psychometric import (
    cumulative_gaussian,
    PsychometricRegression,
)


def plot_single_mouse_psychometric_fit(
    mouse_id, query=None, ax=None, plot_mode="both", sessions_list=None
):
    """
    Plot individual sessions and average psychometric fit for a single mouse.

    Args:
        mouse_id: The ID of the mouse to plot.
        query: Optional additional query as a string.
        ax: Optional matplotlib axis to plot on.
        plot_mode: Mode of plotting ('individual', 'average', 'both').
        sessions_list: List of session_datetime strings or datetime objects to include.
    """

    if ax is None:
        _, ax = plt.subplots(figsize=(5, 5))

    # Build the base query with the mouse_id
    combined_query = f'subject_name="{mouse_id}"'

    # Add any additional query
    if query:
        combined_query += f" AND ({query})"

    # Prepare session query if sessions_list is provided
    if sessions_list is not None:
        # Ensure all session_datetimes are strings formatted correctly
        sessions_list_str = [
            f'"{s.strftime("%Y-%m-%d %H:%M:%S")}"'
            if isinstance(s, datetime.datetime)
            else f'"{s}"'
            for s in sessions_list
        ]
        session_query = "session_datetime in (" + ", ".join(sessions_list_str) + ")"
        combined_query += f" AND ({session_query})"

    # Assign a color to the mouse
    color = "black"

    # Fetch fits for the current mouse with querys
    fits = (PsychometricFit & combined_query).fetch(as_dict=True)
    if fits:
        if plot_mode in ["individual", "both"]:
            # Plot individual fits
            for fit in fits:
                nx = np.linspace(np.min(fit["stims"]), np.max(fit["stims"]), 100)
                ax.plot(
                    nx,
                    cumulative_gaussian(*fit["fit_params"], nx),
                    linewidth=1,
                    alpha=0.1,
                    color=color,
                )
                ax.plot(
                    fit["stims"],
                    fit["p_side"],
                    "o",
                    markersize=4,
                    alpha=0.1,
                    color=color,
                )

        if plot_mode in ["average", "both"]:
            print(
                f"Calculating average fit for {len(fits)} sessions for mouse {mouse_id}..."
            )

            # Get trial data for all sessions of the current mouse
            trial_data = (
                Chipmunk.Trial() & combined_query & "response != 0"
            ) * Chipmunk.TrialParameters()
            response_values, stim_values = trial_data.fetch("response", "stim_rate")

            if len(response_values) > 0:
                # Convert responses: 1 (left) -> 0, -1 (right) -> 1
                all_responses = (response_values == -1).astype(int)
                all_stims = stim_values

                print(f"Combined data: {len(all_stims)} trials for mouse {mouse_id}")

                # Fit psychometric function to combined data
                ft = PsychometricRegression(
                    all_responses.astype(float),
                    exog=all_stims.astype(float)[:, np.newaxis],
                )
                res = ft.fit(min_required_stim_values=6)

                if res is not None:
                    print(f"Successfully fit average curve for mouse {mouse_id}")

                    # Plot average curve
                    nx = np.linspace(np.min(ft.stims), np.max(ft.stims), 100)
                    ax.plot(
                        nx,
                        cumulative_gaussian(*res.params, nx),
                        linewidth=2,
                        label=f"{mouse_id} Average",
                        color=color,
                    )

                    # Plot confidence intervals as vertical lines
                    for stim, p_side, ci in zip(ft.stims, ft.p_side, ft.ci_side):
                        ax.plot([stim, stim], ci, "-_", color=color)

                    # Plot average data points
                    ax.plot(
                        ft.stims,
                        ft.p_side,
                        "o",
                        markerfacecolor="lightgray",
                        markersize=6,
                        color=color,
                    )
                else:
                    print(f"Failed to fit average curve for mouse {mouse_id}")
            else:
                print(
                    f"No trial data available for mouse {mouse_id} with the given querys."
                )
    else:
        print(f"No data available for mouse {mouse_id} with the given querys.")

    # Format plot
    ax.set_ylabel("P(right choice)", fontsize=14)
    ax.set_xlabel("Stimulus rate (Hz)", fontsize=14)
    ax.tick_params(axis="both", which="major", labelsize=10)
    ax.set_ylim([0, 1])

    return ax


def plot_multi_mouse_psychometric_fit(
    mouse_sessions_dict, mice_list=None, query=None, ax=None
):
    """
    Plot average psychometric fits for multiple mice, each with their own session querys.

    Args:
        mouse_sessions_dict: Dictionary where keys are mouse IDs and values are lists of session_datetime
                             strings or datetime objects to include for that mouse.
        query: Optional additional query as a string.
        ax: Optional matplotlib axis to plot on.
    """

    if ax is None:
        _, ax = plt.subplots(figsize=(5, 5))

    if not mice_list:
        mice_list = list(mouse_sessions_dict.keys())

    # Assign colors to mice
    # cmap = plt.get_cmap('viridis')
    colors = sns.color_palette("Set1")
    # colors = cmap(np.linspace(0,1,len(mice_list)))

    for mouse_id, color in zip(mice_list, colors):
        # Get sessions for this mouse
        sessions_list = mouse_sessions_dict[mouse_id]

        # Build the base query for each mouse
        combined_query = f'subject_name="{mouse_id}"'

        # Add any additional query
        if query:
            combined_query += f" AND ({query})"

        # Prepare session query if sessions_list is provided
        if sessions_list is not None:
            # Ensure all session_datetimes are strings formatted correctly
            sessions_list_str = [
                f'"{s.strftime("%Y-%m-%d %H:%M:%S")}"'
                if isinstance(s, datetime.datetime)
                else f'"{s}"'
                for s in sessions_list
            ]
            session_query = "session_datetime in (" + ", ".join(sessions_list_str) + ")"
            combined_query += f" AND ({session_query})"

        # Fetch fits for the current mouse with querys
        fits = (PsychometricFit & combined_query).fetch(as_dict=True)
        if fits:
            print(
                f"Calculating average fit for mouse {mouse_id} with {len(fits)} sessions..."
            )

            # Get trial data for all sessions of the current mouse
            trial_data = (
                Chipmunk.Trial() & combined_query & "response != 0"
            ) * Chipmunk.TrialParameters()
            response_values, stim_values = trial_data.fetch("response", "stim_rate")

            if len(response_values) > 0:
                # Convert responses: 1 (left) -> 0, -1 (right) -> 1
                all_responses = (response_values == -1).astype(int)
                all_stims = stim_values

                print(f"Combined data: {len(all_stims)} trials for mouse {mouse_id}")

                # Fit psychometric function to combined data
                ft = PsychometricRegression(
                    all_responses.astype(float),
                    exog=all_stims.astype(float)[:, np.newaxis],
                )
                res = ft.fit(min_required_stim_values=6)

                if res is not None:
                    print(f"Successfully fit average curve for mouse {mouse_id}")

                    # Plot average curve
                    nx = np.linspace(np.min(ft.stims), np.max(ft.stims), 100)
                    ax.plot(
                        nx,
                        cumulative_gaussian(*res.params, nx),
                        linewidth=2,
                        label=f"{mouse_id}",
                        color=color,
                    )

                    # # Plot confidence intervals as vertical lines
                    # for stim, p_side, ci in zip(ft.stims, ft.p_side, ft.ci_side):
                    #     ax.plot([stim, stim], ci, '-_', color=color)

                    # Plot average data points
                    # ax.plot(
                    #     ft.stims, ft.p_side, 'o',
                    #     markerfacecolor='lightgray', markersize=6, color=color
                    # )
                else:
                    print(f"Failed to fit average curve for mouse {mouse_id}")
            else:
                print(
                    f"No trial data available for mouse {mouse_id} with the given querys."
                )
        else:
            print(f"No data available for mouse {mouse_id} with the given querys.")

    # Format plot
    ax.set_ylabel("P(right choice)", fontsize=14)
    ax.set_xlabel("Stimulus rate (Hz)", fontsize=14)
    ax.tick_params(axis="both", which="major", labelsize=10)
    ax.set_ylim([0, 1])

    return ax
