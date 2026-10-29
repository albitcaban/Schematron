PMA GROUP ANALYSIS v1.3 — SPYDER
=================================

Open pma_group_analysis.py in Spyder (mkultra environment). Run the file or run
# %% 0 and # %% 1 and # %% 1B and the plotting helper first, then any Q1–Q6 cell.

Expected input folder: /Volumes/Extreme SSD/schema/PMA_BATCH
Change ROOT at top if needed.
Output: /Volumes/Extreme SSD/schema/PMA_BATCH/GROUP_FIGURES_v1_3
All PNGs 300 DPI (SAVE_PDF_ALSO=True for vector PDFs).

Fixed from v1.2:
- Blank cue_video_covered is not interpreted as failed DLC video.
- Platform, zone and reward-head cue outcomes are filtered separately by actual
  valid tracking duration; default threshold 15 seconds of a 30-second cue.
- ITI outcomes use their own valid duration columns.
- Arduino poke and reward counts are never masked by cue DLC validity.
- 3f_a_tp3 remains excluded from video-derived analyses, but independently
  recorded valid shock/poke/reward outcomes can remain.
- Per-bin platform probabilities are not discarded when other bins of the
  same trial have insufficient 30-second coverage.
- Coverage audit CSV and console table added.

Statistical caution: These are descriptive day-wise Mann–Whitney tests with
FDR correction, comparing mouse means. They do not model age × day repeated
measures. Avoidance from Stage2 shock_received may depend on video platform
classification, and missing shock outcomes stay missing.

Assumed-tail alignment sessions (0m_a_tp3, 2f_a_tp4, 5m_t_tp1) remain
included by default and are not proven frame-accurate. Set
INCLUDE_ASSUMED_TAIL=False for sensitivity analyses.
