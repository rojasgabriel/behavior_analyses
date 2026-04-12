# behavior_analyses

Behavior analysis notebooks and psychometric fitting utilities.

## Repository layout

- `behavioral_metrics/`: notebooks for session-level behavioral metrics and learning/performance analyses.
- `psychometric_curves/`: psychometric fitting code and notebooks (`fit_psychometric.py`, plotting and parameter exploration notebooks).
- `psychophysical_kernels/`: notebook for psychophysical kernel analysis.
- `oft/`: open-field test notebooks.
- `labdata2_testing/`: notebook for labdata2 workflow testing.
- `utils.py`: shared plotting/helper utilities.

## Main Python utilities

- `psychometric_curves/fit_psychometric.py`
  - Implements cumulative Gaussian / Weibull-style psychometric functions.
  - Provides trial proportion/confidence interval computation.
  - Includes `PsychometricRegression` and `fit_psychometric` helpers.
- `psychometric_curves/utils.py`
  - Plot helpers to fit and visualize psychometric curves for single or multiple mice from DataJoint tables.

## Requirements

Based on imports in this repository:

- `numpy`
- `pandas`
- `scipy`
- `statsmodels`
- `matplotlib`
- `seaborn`
- `djchurchland` (for DataJoint pipeline access in plotting utilities)

## Usage

1. Open notebooks in the analysis subfolders to reproduce figures/analyses.
2. Use `psychometric_curves/fit_psychometric.py` in scripts for programmatic fitting.
3. Configure DataJoint access before using `psychometric_curves/utils.py` plotting functions.
