# Behavior Analyses

Repository for behavioral data analysis and visualization.

## Interactive Performance Dashboard 🎯

A new interactive web-based dashboard for visualizing behavioral performance summaries has been added to this repository. This dashboard replaces the CLI-based plotting script with a user-friendly web interface.

### Quick Start

1. **Install dependencies:**
   ```bash
   ./setup_dashboard.sh
   ```

   Or manually:
   ```bash
   pip install -r labdata2_testing/dashboard_requirements.txt
   ```

2. **Launch the dashboard:**
   ```bash
   cd labdata2_testing
   python launch_dashboard.py
   ```

3. **Open your browser:**
   Navigate to [http://127.0.0.1:8050](http://127.0.0.1:8050)

### Dashboard Features

The interactive dashboard provides:

- **Real-time Data Visualization**: Select different mice and adjust session ranges dynamically
- **Interactive Plots**: Zoom, pan, and hover over plots for detailed information
- **No File Management**: All visualizations are displayed in the browser - no more saving PNG files!
- **Single Command Launch**: Start the entire dashboard with one simple command

### Documentation

For detailed documentation about the dashboard, see [labdata2_testing/DASHBOARD_README.md](labdata2_testing/DASHBOARD_README.md)

### Visualization Types

The dashboard displays 6 interactive plots:

1. Fraction correct per stimulus intensity
2. Performance over recent sessions (easy vs. all trials)
3. Probability of right responses per stimulus
4. Early withdrawal rate across sessions
5. Reaction time distribution
6. Trial count breakdown (total, with choice, correct)

### Migration from CLI Script

**Old approach** (`perf_summaries_script.py`):
```bash
python perf_summaries_script.py GRB050 --out-dir ~/Downloads --sessions-back 10
```

**New approach** (Dashboard):
```bash
python launch_dashboard.py
```

The dashboard provides all the same visualizations but with added interactivity and a much better user experience.

## Repository Structure

```
behavior_analyses/
├── labdata2_testing/
│   ├── dashboard.py              # Main dashboard application
│   ├── launch_dashboard.py       # Simple launcher script
│   ├── dashboard_requirements.txt # Dashboard dependencies
│   ├── DASHBOARD_README.md       # Detailed dashboard documentation
│   ├── perf_summaries_script.py  # Original CLI script (for reference)
│   └── ...
├── behavioral_metrics/
├── oft/
├── psychometric_curves/
├── psychophysical_kernels/
└── setup_dashboard.sh            # Quick setup script
```

## Requirements

- Python 3.8+
- Access to the `labdata` database (for data fetching)
- Dashboard dependencies (automatically installed via setup script)

## Development

The dashboard is built with:
- **Plotly Dash**: Web application framework
- **Plotly**: Interactive plotting library
- **Dash Bootstrap Components**: UI components
- **Pandas/NumPy**: Data processing

## License

See repository license for details.
