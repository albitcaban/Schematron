PMA Stage 2 v0.4 — reward-zone visit detection and validation

Install into your existing mkultra environment:
  python -m pip install numpy pandas tables matplotlib opencv-python

Analyze an individual session (replace the folder with your own):
  python pma_stage2.py --session "/Volumes/Extreme SSD/schema/adults/0f_a/0f_a_lp1"

Inspect reward entry events:
  python validate_events.py --session "/Volumes/Extreme SSD/schema/adults/0f_a/0f_a_lp1" --event reward_entry

Inspect reward exit events:
  python validate_events.py --session "/Volumes/Extreme SSD/schema/adults/0f_a/0f_a_lp1" --event reward_exit

Inspect all event types, including platform entries, exits, head dips and reward entries/exits:
  python validate_events.py --session "/Volumes/Extreme SSD/schema/adults/0f_a/0f_a_lp1"

Viewer: A/D previous/next event, comma/period previous/next frame, Q quit.

New outputs (within PMA_ANALYSIS):
  reward_visits.csv — paired visit start/end, duration, censored status, pokes and rewards within visit
  baseline_summary.csv — pre-first-cue baseline zone occupancy and event counts
  behavior_events.csv — now also reward_entry and reward_exit
  trial_summary.csv — cue/ITI visit counts, time, rate, visits associated with pokes/rewards,
                       entry/exit counts, first cue reward-approach latency
  frame_behaviors.csv — independent reward head state plus mutually exclusive priority zone
  trial_probability_bins.csv — reward_zone_binary at 2-second bin midpoints
  session_probabilities.csv — reward_zone_probability and valid trial counts
  poke_platform_probability.png — includes reward-zone probability as a third curve

Reward ROI rule: any reliable nose, left ear or right ear inside = inside; ALL 3 reliable
and outside = outside; otherwise unknown. Entry/exit requires 200 ms persistence;
unknown stretches and >150 ms gaps prevent unsupported transitions. A visit crossing
cue/ITI contributes time to both, while the event count belongs to its start period.
Visits starting before first stable detection inside are left-censored and have no
reward_entry event; visits still open at the session end are right-censored.

The 'reward_occupancy_pct' column retains priority/exclusive zones (platform wins);
'reward_head_occupancy_pct' measures independent head-based reward occupancy even
when platform and reward regions overlap. Do not interpret visit-to-poke association
as causality; it means an Arduino event fell inside the detected visit interval.

No source data, Stage 1 QC files, or zone_rois.json are changed. Existing
PMA_ANALYSIS outputs WILL be overwritten; save a copy if you want old results.
Freezing remains unavailable without calibrated cm-per-pixel input.

Scientific validation remains essential: check ROI and several reward transitions
in the viewer before batch analysis. Test fixtures used during development employed
artificial ROIs, NOT the user's saved session polygons.
