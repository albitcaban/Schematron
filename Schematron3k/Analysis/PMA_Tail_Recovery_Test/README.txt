PMA tail-truncation vs distributed-loss timing test

Run one session at a time:
python pma_tail_recovery_test.py --session "/Volumes/Extreme SSD/schema/adults/2f_a/2f_a_tp4"
python pma_tail_recovery_test.py --session "/Volumes/Extreme SSD/schema/teens/5m_t/5m_t_tp1"
python pma_tail_recovery_test.py --session "/Volumes/Extreme SSD/schema/adults/0m_a/0m_a_tp3"

Dependencies: pip install numpy pandas matplotlib tables

Reads: PMA_QC/frame_time_mapping.csv, PMA_QC/events_corrected_timing.csv (or raw behavioral_events.csv), filtered DLC H5/CSV, zone_rois.json.
Creates: PMA_RECOVERY_TEST/{recovery_test_report.json,nosepoke_alignment_by_5min.csv,trial_window_coverage.csv,nosepoke_alignment_comparison.png}

SAFE DIAGNOSTIC ONLY. Does not change source data, PMA_QC or PMA_ANALYSIS. The diagnostic compares two timing hypotheses, not a verified correspondence. Saved reward ROI may be broad and can have a high coincidence rate by chance. Examine the score's excess over shifted controls, timing consistency across epochs and trial window coverage. For exact shock/transition timing, frame-write IDs or stronger independent synchronization are needed.
