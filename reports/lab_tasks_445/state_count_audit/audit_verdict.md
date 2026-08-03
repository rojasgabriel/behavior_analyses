# LAB-TASKS-505 GLM-HMM state-count audit

## Result

Recommend **K = 2** for the next behavioral-validity audit. Do not yet call either state engaged and do not run the neural comparison.

Five-fold held-out whole-session likelihood improved from -0.517736 nats per trial for K = 1 to -0.507464 for K = 2, -0.497671 for K = 3, and -0.488417 for K = 4. The automated one-standard-error rule therefore selected K = 4, but the required stability audit rejected it: the fourth state had only 10 hard-assigned trials, no assignments above posterior 0.70, five two-trial dwells, and an unstable negative sensory coefficient. K = 3 also failed, with 175 hard assignments, 24 above posterior 0.90, four-trial median dwells, and an unstable negative sensory coefficient.

K = 2 cleared the rarity and persistence checks. The selected full-data seed separated a high-sensory state (evidence coefficient 0.435; 16,740 hard assignments; median dwell 22 trials) from a low-sensory/history-driven state (evidence coefficient 0.157; 21,011 assignments; median dwell 16 trials). At posterior 0.90, 5,240 and 11,134 trials remained, respectively. The best full-data seed also had the best held-out score; its full-data log likelihood was -18,907.48 versus -19,079.93 and -19,281.54 for the other seeds. The high-sensory state's reward-history coefficient and the inferior-seed parameter spread remain external-validity targets, so this is a state-count recommendation rather than an engagement label or neural-decoding claim.

## Artifacts

- `../grb006_state_count/`: complete K = 1-4 fold-by-seed contrast tables and automated model-selection figure.
- `../grb006_state_count_k2_cap/`: selected K = 2 full-data seed tables, posterior table, transition matrix, dwell/occupancy/entropy diagnostics, and figures.
- `../synthetic_recovery/`: synthetic persistence and state-weight recovery check.

## Regeneration

```bash
CHIPMUNK_PLUGIN_PATH="$CHIPMUNK_PLUGIN_PATH" MPLCONFIGDIR=/tmp/mpl UV_CACHE_DIR=/tmp/uv-cache \
uv run python scripts/analyses/fit_engagement_states.py \
  --subject GRB006 \
  --output-dir reports/lab_tasks_445/grb006_state_count \
  --states 1 2 3 4 --seeds 0 1 2 --folds 5 \
  --em-iters 15 --m-step-iters 20

CHIPMUNK_PLUGIN_PATH="$CHIPMUNK_PLUGIN_PATH" MPLCONFIGDIR=/tmp/mpl UV_CACHE_DIR=/tmp/uv-cache \
uv run python scripts/analyses/fit_engagement_states.py \
  --subject GRB006 \
  --output-dir reports/lab_tasks_445/grb006_state_count_k2_cap \
  --states 1 2 --seeds 0 1 2 --folds 5 \
  --em-iters 15 --m-step-iters 20
```
