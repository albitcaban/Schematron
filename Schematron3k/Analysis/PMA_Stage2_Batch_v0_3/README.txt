PMA Stage 2 — Batch v0.2
========================
This package copies the previously validated Stage 2 v0.4 analysis and event viewer unchanged.
New file pma_batch.py processes all session folders, creates a batch log, and combines
small summary/event/bin tables. Freezing is deferred and not required.

Dependencies:
 python -m pip install numpy pandas tables matplotlib opencv-python

From this folder (with mkultra activated):
 1. Preview all sessions:
 python pma_batch.py --root "/Volumes/Extreme SSD/schema" --dry-run
 2. Run batch:
 python pma_batch.py --root "/Volumes/Extreme SSD/schema"
 3. To re-run previously completed sessions (overwrites their PMA_ANALYSIS outputs):
 python pma_batch.py --root "/Volumes/Extreme SSD/schema" --force

Folders: adults/0f_a/0f_a_tp1 and adolescents/0f_t/0f_t_lp1 etc.
WT and WT2 are Arduino-only. They use PMA_QC/events_corrected_timing.csv if present, otherwise behavioral_events.csv directly (using arduino_ms). No video, ROI, or DLC required. Their session_summary contains poke/reward totals, unrewarded-poke estimate, durations, and rates.
TP and LP require PMA_QC frame_time_mapping.csv, corrected events, zone_rois.json,
and a filtered DLC CSV/H5. Original sources are never modified.

Outputs:
 - Per session: existing PMA_ANALYSIS outputs from Stage 2 v0.4
 - Root: schema/PMA_BATCH/batch_run_status.csv and batch_manifest.json
 - Root: schema/PMA_BATCH/combined session_summary.csv, trial_summary.csv,
   baseline_summary.csv, behavior_events.csv, reward_visits.csv,
   trial_probability_bins.csv and session_probabilities.csv
 - Large frame_behaviors.csv files stay in their respective session folders.

Default behavior: skip completed sessions, include them in combined outputs.
A missing or failing session is reported and does not stop other sessions.

Dry run should label WT/WT2 as ready_baseline. Existing baselines previously marked completed may be skipped; use --force to regenerate their summaries.

SMALL FRAME DISCREPANCIES (batch v0.3)
Default --small-mismatch-max 20 permits 1-19 frame differences via an approximate monotone endpoint-preserving timestamp assignment. This DOES NOT recover which camera frames were lost. For affected sessions a small_mismatch_alignment.json warning is saved. Do not interpret sub-second transition/shock timings from these sessions as verified. Differences >=20 frames still fail safely.
To retry failed sessions without reprocessing successful ones: python pma_batch.py --root '/Volumes/Extreme SSD/schema'
