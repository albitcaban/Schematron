#!/usr/bin/env python3
"""PMA Stage 1 v1.2: non-destructive timing QC, synchronized exports, and per-session ROIs.

Usage:
 python pma_qc_zones.py --root '/Volumes/Extreme SSD/schema' --all
 python pma_qc_zones.py --session '/Volumes/Extreme SSD/schema/adults/0f_a/0f_a_lp1' --roi
 python pma_qc_zones.py --session '/path/to/session' --qc

Original Arduino/DLC/video files are NEVER modified. QC outputs go in PMA_QC/.
Requires: numpy, pandas, opencv-python. Optional: ffprobe on PATH.
"""
import argparse
import csv
import json
import re
import shutil
import subprocess
from pathlib import Path
import numpy as np
import pandas as pd

DAY_RE=re.compile(r'^(?P<num>\d+)(?P<sex>[fm])_(?P<age>[at])_(?P<day>wt2?|tp[1-9]\d*|lp[1-9]\d*)$',re.I)

def identify(path):
    m=DAY_RE.fullmatch(path.name.lower())
    if not m: return None
    g=m.groupdict()
    return dict(animal=f"{g['num']}{g['sex']}_{g['age']}",sex='female' if g['sex']=='f' else 'male',age='adult' if g['age']=='a' else 'adolescent',training_day=g['day'].upper(),baseline=g['day'].startswith('wt'))

def locate(p):
    find=lambda pat:sorted(p.glob(pat),key=lambda x:(len(x.name),x.name.lower()))
    ev=find('behavioral_events.csv'); cam=find('camera_frames*.csv'); meta=find('session_metadata.json'); ttl=find('ttl_events.csv')
    vids=find('*.mp4') or find('*.avi')
    dlc=[x for x in find('*filtered.h5')+find('*filtered.csv') if not x.name.startswith('PMA_')]
    if not dlc: dlc=find('*DLC*.h5') or find('*DLC*.csv')
    # Match tracking by video basename where possible; otherwise use first found.
    if vids and dlc:
        matching=[d for d in dlc if any(d.name.lower().startswith(v.stem.lower()) for v in vids)]
        if matching: dlc=matching
    return {k:(v[0] if v else None) for k,v in [('events',ev),('camera',cam),('metadata',meta),('ttl',ttl),('video',vids),('dlc',dlc)]}

def dlc_info(path):
    if not path:return {'rows':None,'bodyparts':[]}
    try:
        if path.suffix.lower()=='.h5':
            with pd.HDFStore(path,'r') as h:
                keys=h.keys()
                if not keys:return {'rows':None,'bodyparts':[],'error':'empty HDF5'}
                frame=h.select(keys[0],start=0,stop=1)
                st=h.get_storer(keys[0]); rows=getattr(st,'nrows',None)
                if rows is None: rows=len(h.select(keys[0]))
        else:
            frame=pd.read_csv(path,header=[0,1,2],nrows=1,index_col=0)
            # Count rows without loading full tracking into memory.
            with path.open('rb') as f: rows=max(sum(block.count(b'\n') for block in iter(lambda:f.read(1024*1024),b''))-3,0)
        cols=frame.columns
        parts=list(dict.fromkeys(str(x) for x in cols.get_level_values(1))) if isinstance(cols,pd.MultiIndex) else []
        return {'rows':int(rows),'bodyparts':parts}
    except Exception as exc:return {'rows':None,'bodyparts':[],'error':str(exc)}

def video_info(path):
    if not path: return {}
    if not shutil.which('ffprobe'):
        return {'warning':'ffprobe not on PATH; install FFmpeg/ffprobe in the active environment'}
    cmd=['ffprobe','-v','error','-select_streams','v:0','-show_entries',
         'stream=nb_frames,avg_frame_rate,r_frame_rate,width,height,duration,time_base:format=duration',
         '-of','json',str(path)]
    try:
        raw=json.loads(subprocess.check_output(cmd,stderr=subprocess.STDOUT,timeout=120).decode())
        streams=raw.get('streams',[])
        if not streams:return {'warning':'ffprobe found no video stream'}
        stream=streams[0]
        result={'video_metadata':stream,'format_duration_s':raw.get('format',{}).get('duration'),
                'decoded_frame_count_verified':False,
                'note':'nb_frames comes from container metadata; no full decoded-frame audit'}
        # Some codecs/containers omit nb_frames. This packet-based query is not a decoded audit.
        if not stream.get('nb_frames'):
            cmd2=['ffprobe','-v','error','-select_streams','v:0','-count_packets',
                  '-show_entries','stream=nb_read_packets','-of','json',str(path)]
            try:
                extra=json.loads(subprocess.check_output(cmd2,stderr=subprocess.STDOUT,timeout=300).decode())
                result['video_packet_count']=extra['streams'][0].get('nb_read_packets')
            except Exception as exc:result['packet_count_warning']=str(exc)
        return result
    except Exception as exc:return {'warning':f'ffprobe failed: {exc}'}


def camera_clock(cam):
    """Choose frame timing while respecting host/camera clock differences.

    Camera embedded (camera_seconds+microseconds) generally measures acquisition
    more closely; it is aligned to host clock by a robust offset, not by assuming
    it shares an epoch. Large clock drift => fall back to host and flag for review.
    """
    preferred=('host_capture_monotonic_s','host_monotonic_s')
    chosen=next((c for c in preferred if c in cam.columns),None)
    if chosen is None:
        raise ValueError('No compatible host monotonic camera timestamp. Columns: '+', '.join(cam.columns))
    host=pd.to_numeric(cam[chosen],errors='coerce')
    info={'host_column':chosen,'legacy_format':chosen=='host_monotonic_s',
          'selected_column':chosen,'clock_type':'host monotonic',
          'camera_embedded_timestamp_used_for_alignment':False}
    if not {'camera_seconds','camera_microseconds'}.issubset(cam.columns):
        return host,host,info
    sensor=pd.to_numeric(cam.camera_seconds,errors='coerce')+pd.to_numeric(cam.camera_microseconds,errors='coerce')/1e6
    sd=sensor.diff();hd=host.diff()
    valid=(sd>0)&(hd>0)&np.isfinite(sd)&np.isfinite(hd)
    info['sensor_nonincreasing_intervals']=int(((sd<=0)|~np.isfinite(sd)).iloc[1:].sum())
    if valid.sum()<10:
        info['sensor_rejected_reason']='Too few positive valid camera/host timestamp pairs'
        return host,host,info
    info['sensor_median_interframe_ms']=float(sd[valid].median()*1000)
    info['host_median_interframe_ms']=float(hd[valid].median()*1000)
    info['sensor_vs_host_interval_median_abs_difference_ms']=float((sd[valid]-hd[valid]).abs().median()*1000)
    good=np.isfinite(sensor)&np.isfinite(host)
    if int(good.sum())<10 or info['sensor_nonincreasing_intervals']:
        info['sensor_rejected_reason']='Invalid or nonincreasing embedded camera time'
        return host,host,info
    # Evaluate stability of sensor -> host offset in first and last 5% of frames.
    offsets=(host-sensor).to_numpy(dtype=float)
    idx=np.flatnonzero(good.to_numpy()); take=max(10,min(1000,len(idx)//20))
    first=float(np.median(offsets[idx[:take]])); last=float(np.median(offsets[idx[-take:]]))
    drift=last-first
    info['host_minus_sensor_offset_start_s']=first
    info['host_minus_sensor_offset_end_s']=last
    info['host_sensor_offset_drift_s']=float(drift)
    info['host_sensor_offset_median_s']=float(np.median(offsets[idx]))
    # Host timestamps can be delayed by processing; use robust overall offset to
    # avoid forcing late-arriving video frames to match delayed host logging.
    # Reject potentially unstable clocks (large drift over session).
    elapsed=float(sensor.iloc[-1]-sensor.iloc[0])
    allowable_drift=max(.15,.0002*max(elapsed,0))
    if abs(drift)>allowable_drift:
        info['sensor_rejected_reason']=f'Clock offset drifts {drift:.3f}s; threshold {allowable_drift:.3f}s'
        return host,host,info
    corrected=sensor+float(np.median(offsets[idx]))
    info['selected_column']='camera_seconds + camera_microseconds / 1e6'
    info['clock_type']='camera embedded acquisition time + robust host offset'
    info['camera_embedded_timestamp_used_for_alignment']=True
    info['alignment_uncertainty_note']='Absolute offset estimated from frame-arrival host time; variable readout latency cannot be fully corrected without hardware sync.'
    return corrected,host,info

def qc_one(session,run_roi=False):
    session=Path(session).expanduser().resolve()
    who=identify(session)
    if who is None:raise ValueError(f'Not a recognized session folder: {session.name}')
    paths=locate(session)
    if not paths['events']:raise FileNotFoundError(f'No behavioral_events.csv in {session}')
    out=session/'PMA_QC';out.mkdir(exist_ok=True)
    ev=pd.read_csv(paths['events'])
    if 'host_monotonic_s' not in ev or 'event' not in ev:raise ValueError('Unexpected behavioral_events.csv columns')
    ev['host_monotonic_s']=pd.to_numeric(ev['host_monotonic_s'],errors='coerce')
    starts=ev.loc[ev['event'].astype(str).str.startswith('EXPERIMENT_START'),'host_monotonic_s'].dropna()
    camera_starts=ev.loc[ev.event.eq('CAMERA_START'),'host_monotonic_s'].dropna()
    t0=float(starts.iloc[0]) if len(starts) else (float(camera_starts.iloc[0]) if len(camera_starts) else float(ev.host_monotonic_s.dropna().iloc[0]))
    ev['corrected_session_s']=ev.host_monotonic_s-t0
    ev['timing_reference']='host_monotonic_s - EXPERIMENT_START host_monotonic_s'
    ev.to_csv(out/'events_corrected_timing.csv',index=False)
    if paths['ttl']:
        ttl=pd.read_csv(paths['ttl']);
        if 'host_monotonic_s' in ttl:
            ttl['corrected_session_s']=pd.to_numeric(ttl.host_monotonic_s,errors='coerce')-t0
            ttl.to_csv(out/'ttl_corrected_timing.csv',index=False)
    result={'session':str(session),'identity':who,'source_files':{k:str(v) if v else None for k,v in paths.items()},'reference_monotonic_s':t0,'event_count':len(ev),'event_types':ev.event.value_counts().to_dict(),'warnings':[],'status':'ok'}
    if who['baseline']:
        result['camera_needed']=False
        result['note']='WT/WT2: Arduino-only; no camera, DLC or ROIs required'
    else:
        result['camera_needed']=True
        if paths['camera']:
            cam=pd.read_csv(paths['camera'])
            clock,host_clock,clock_info=camera_clock(cam)
            cam['corrected_session_s']=clock-t0
            cam['corrected_from_first_frame_s']=clock-clock.iloc[0]
            cam['dlc_row_zero_based']=np.arange(len(cam),dtype=int)
            delta=clock.diff()
            good=delta[delta.gt(0)&np.isfinite(delta)]
            median=float(good.median()) if len(good) else None
            gap=delta.gt(1.8*median) if median else pd.Series(False,index=cam.index)
            backwards=delta.le(0).sum()
            cam['interframe_gap_flag']=gap.fillna(False)
            cam['timestamp_nonincreasing_flag']=delta.le(0).fillna(False)
            cam['selected_host_monotonic_s']=host_clock
            cam['selected_acquisition_monotonic_s']=clock
            cam['host_minus_selected_acquisition_s']=host_clock-clock
            cam['timing_source_column']=clock_info['selected_column']
            export_cols=['frame_index','dlc_row_zero_based','corrected_session_s',
                'corrected_from_first_frame_s','selected_host_monotonic_s',
                'selected_acquisition_monotonic_s','host_minus_selected_acquisition_s',
                'timing_source_column','interframe_gap_flag','timestamp_nonincreasing_flag']
            cam[[c for c in export_cols if c in cam]].to_csv(out/'frame_time_mapping.csv',index=False)
            result['camera_timing_source']=clock_info
            if clock_info.get('legacy_format') and not clock_info.get('camera_embedded_timestamp_used_for_alignment'):
                result['warnings'].append('Legacy host frame-arrival timestamps in use; embedded camera clock not validated: '+clock_info.get('sensor_rejected_reason','unavailable'))
            elif clock_info.get('legacy_format'):
                result['warnings'].append('Legacy recording: camera acquisition clock aligned to host with estimated offset; absolute offset may have frame-arrival latency')
            result['camera']={'rows':len(cam),'duration_s':float(cam.corrected_from_first_frame_s.iloc[-1]),'median_interframe_ms':median*1000 if median else None,'estimated_fps':1/median if median else None,'long_interframe_gaps':int(gap.sum()),'nonincreasing_intervals':int(backwards),'first_frame_session_s':float(cam.corrected_session_s.iloc[0])}
            if backwards:result['warnings'].append(f'{backwards} nonincreasing acquisition timestamps')
            if gap.any():result['warnings'].append(f'{int(gap.sum())} long interframe gaps >1.8x median')
            if 'frame_index' in cam and not np.array_equal(pd.to_numeric(cam.frame_index,errors='coerce').to_numpy(),np.arange(1,len(cam)+1)):
                result['warnings'].append('frame_index is not contiguous 1..N')
        else: result['warnings'].append('camera_frames.csv missing')
        dlc=dlc_info(paths['dlc']);result['dlc']=dlc
        if not paths['dlc']:result['warnings'].append('DLC tracking missing')
        if paths['camera'] and dlc['rows'] is not None and dlc['rows']!=result['camera']['rows']:
            result['warnings'].append(f"DLC rows ({dlc['rows']}) != camera rows ({result['camera']['rows']}); mapping not validated")
        result['video']=video_info(paths['video'])
        if not paths['video']:result['warnings'].append('Video missing; ROIs cannot be drawn')
        else:
            stream=result['video'].get('video_metadata',{})
            n=stream.get('nb_frames')
            if n and paths['camera'] and int(n)!=result['camera']['rows']:
                result['warnings'].append(f"Video reported frames ({n}) != camera rows ({result['camera']['rows']}); investigate before using row mapping")
            if n and result.get('dlc',{}).get('rows') is not None and int(n)!=result['dlc']['rows']:
                result['warnings'].append(f"Video reported frames ({n}) != DLC rows ({result['dlc']['rows']})")
            vd=stream.get('duration') or result['video'].get('format_duration_s')
            if vd and paths['camera']:
                try:
                    vd=float(vd); cd=result['camera']['duration_s']; diff=vd-cd
                    result['video']['duration_vs_acquisition_s']=float(diff)
                    result['video']['duration_ratio_to_acquisition']=float(vd/cd) if cd else None
                    if abs(diff)>max(2.0,cd*.005):
                        result['warnings'].append(f'MP4 duration differs from camera acquisition by {diff:+.2f}s (use corrected frame mapping, not video playback timestamps)')
                except (TypeError,ValueError):pass
    if result['warnings']:result['status']='review_required'
    (out/'qc_report.json').write_text(json.dumps(result,indent=2,default=str))
    (out/'qc_summary.txt').write_text('\n'.join([f"Session: {session.name}",f"Status: {result['status']}",f"Age/sex/day: {who['age']}, {who['sex']}, {who['training_day']}",f"Event count: {len(ev)}",f"Camera: {result.get('camera',{})}",f"DLC: {result.get('dlc',{})}",f"Video: {result.get('video',{})}",'Warnings:']+['  - '+w for w in result['warnings']]+['','NOTE: DLC row-to-frame mapping is ASSUMED by order until container counts/visual fidelity are verified.','Raw source files were not changed.']))
    print(f"[{result['status']}] {session.name}: {len(ev)} events; {result.get('camera',{}).get('rows','no')} frames; {result.get('dlc',{}).get('rows','no')} DLC rows")
    for w in result['warnings']:print('  WARNING:',w)
    if run_roi and not who['baseline']:
        if not paths['video']:print('No video available for ROI editor')
        else:
            # Import the reusable standalone editor with its CLI arguments.
            import sys
            import pma_roi_editor
            old=sys.argv
            sys.argv=['pma_roi_editor.py','--video',str(paths['video']),'--output',str(session/'zone_rois.json')]
            try:pma_roi_editor.main()
            finally:sys.argv=old
    return result

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    g=ap.add_mutually_exclusive_group(required=True)
    g.add_argument('--session',type=Path,help='Single session folder, e.g. 0f_a_lp1')
    g.add_argument('--root',type=Path,help='Root schema directory')
    ap.add_argument('--all',action='store_true',help='QC every nested session under --root')
    ap.add_argument('--qc',action='store_true',help='QC only, no ROI window')
    ap.add_argument('--roi',action='store_true',help='Open ROI editor after QC (one session)')
    args=ap.parse_args()
    if args.session:
        qc_one(args.session,run_roi=not args.qc)
    else:
        sessions=sorted(x for x in args.root.rglob('*') if x.is_dir() and identify(x) and (x/'behavioral_events.csv').exists())
        print(f'Found {len(sessions)} session folders')
        reports=[]
        for s in sessions:
            try:reports.append(qc_one(s,run_roi=False))
            except Exception as exc:print(f'[ERROR] {s}: {exc}')
        if reports:
            dest=args.root/'PMA_QC_all_sessions.csv'
            pd.DataFrame([{'folder':r['session'],'animal':r['identity']['animal'],'day':r['identity']['training_day'],'age':r['identity']['age'],'sex':r['identity']['sex'],'status':r['status'],'warnings':'; '.join(r['warnings']),'camera_rows':r.get('camera',{}).get('rows'),'dlc_rows':r.get('dlc',{}).get('rows')} for r in reports]).to_csv(dest,index=False)
            print('Wrote',dest)
        print('ROIs: choose one session using --session PATH --roi')

if __name__=='__main__':main()
