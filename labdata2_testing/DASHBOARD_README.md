# Performance Dashboard

An interactive web-based dashboard for visualizing behavioral performance summaries, replacing the CLI-based `perf_summaries_script.py`.

## Features

- **Interactive Plots**: All visualizations are interactive with hover information, zoom, and pan capabilities
- **Real-time Updates**: Select different mice and adjust session ranges dynamically
- **No File Saving**: All visualizations are displayed in the browser, eliminating the need to save static files
- **Single Command Launch**: Start the dashboard with one simple command

## Installation

Install the required dependencies:

```bash
cd labdata2_testing
pip install -r dashboard_requirements.txt
```

## Usage

Launch the dashboard with a single command:

```bash
python launch_dashboard.py
```

Or alternatively:

```bash
python dashboard.py
```

The dashboard will start and be accessible at: **http://127.0.0.1:8050**

Open this URL in your web browser to interact with the dashboard.

## Dashboard Features

The dashboard displays 6 interactive plots:

1. **Fraction Correct per Stimulus**: Performance accuracy across different stimulus intensities
2. **Performance Over Sessions**: Track performance on easy trials and all trials over recent sessions
3. **P(Right) per Stimulus**: Probability of right responses for each stimulus intensity
4. **Early Withdrawal Rate**: Monitor early withdrawal behavior across sessions
5. **Reaction Times**: Distribution of reaction times for the most recent session
6. **Trial Counts**: Breakdown of total, choice, and correct trials over sessions

## Controls

- **Mouse Selector**: Choose which mouse's data to display from a dropdown menu
- **Sessions Back Slider**: Adjust how many recent sessions to include in the analysis (1-30)

## Comparison with Original Script

### Original CLI Script (`perf_summaries_script.py`)
```bash
python perf_summaries_script.py GRB050 GRB051 --out-dir ~/Downloads --sessions-back 10
```
- Required command-line arguments
- Generated static PNG files
- No interactivity
- Manual file management

### New Dashboard
```bash
python launch_dashboard.py
```
- Single command to launch
- Interactive web interface
- No file saving required
- Real-time visualization updates
- Easy switching between mice and session ranges

## Technical Details

- **Framework**: Plotly Dash with Bootstrap styling
- **Backend**: Connects to the same `labdata.schema.DecisionTask` database as the original script
- **Port**: Runs on port 8050 by default
- **Host**: Accessible at `0.0.0.0` (all network interfaces)

## Stopping the Dashboard

Press `Ctrl+C` in the terminal where the dashboard is running to stop the server.
