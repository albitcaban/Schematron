PMA Stage 1 Quality Control + Zone Editor v1.2
==============================================
Replace pma_qc_zones.py and pma_roi_editor.py in your PMA_Stage1_QC_Zones folder with the files in this archive.

Requirements: pandas numpy opencv-python and ffprobe available in your mkultra PATH.

Run QC (preserves saved ROI JSON):
python pma_qc_zones.py --session '/Volumes/Extreme SSD/schema/adults/0f_a/0f_a_tp1' --qc

Run QC then open ROI editor:
python pma_qc_zones.py --session '/Volumes/Extreme SSD/schema/adults/0f_a/0f_a_tp1' --roi

Run across all sessions without opening videos:
python pma_qc_zones.py --root '/Volumes/Extreme SSD/schema' --all

SAVES INSIDE each session/PMA_QC:
- events_corrected_timing.csv (Arduino host timeline)
- ttl_corrected_timing.csv (when TTL log supplied)
- frame_time_mapping.csv (DLC-row-to-acquisition-time table)
- qc_report.json; qc_summary.txt
Does not overwrite session/zone_rois.json when run with --qc.

NEW IN v1.2
- ffprobe video container frame-count and duration checks, with warnings on mismatches
- compare camera embedded clock with host monotonic clock in legacy recordings
- select embedded acquisition clock if monotonic and its host offset drift is low; otherwise use host timestamps and flag review
- estimate camera->host offset, with documented uncertainty due to frame-arrival latency
- retain host timestamps and selected acquisition times in output mapping

LIMITATIONS
- MP4 container frame count is metadata, NOT a full decoded frame audit.
- Assumes video and DLC rows map by their original order; does not repair missing/reordered video frames.
- Camera-to-host synchronization is estimated from software logging; precise absolute offset cannot be guaranteed without hardware synchronization.
- Camera frame gaps may be acquisition gaps or logging artifacts. Examine qc_report.json warnings.
- Keep originals; outputs are new derived QC tables, not changes to source videos.
- Rectangles are saved by pressing S in ROI editor; ESC quits. Q rotates (not quits).
