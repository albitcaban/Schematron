#!/usr/bin/env python3
"""PMA large-mismatch diagnosis: evidence only, no fabricated frame mapping.
Usage: python pma_large_mismatch_diagnostic.py --session '/Volumes/Extreme SSD/schema/adults/3f_a/3f_a_tp3'
"""
import argparse, json, re
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def run(session, camera_path=None, dlc_path=None, output=None):
    session=Path(session) if session else None
    cam_path=Path(camera_path) if camera_path else session/'camera_frames.csv'
    if dlc_path:
        dlc_file=Path(dlc_path)
    else:
        candidates=sorted(session.glob('*filtered.h5')) or sorted(session.glob('*filtered.csv'))
        if not candidates: raise FileNotFoundError('No filtered DLC file found')
        dlc_file=candidates[0]
    out=Path(output) if output else session/'PMA_RECOVERY_DIAGNOSTIC'
    out.mkdir(parents=True,exist_ok=True)
    c=pd.read_csv(cam_path)
    # Read DLC count while retaining no tracking arrays. HDF preferred, CSV fallback.
    if dlc_file.suffix.lower()=='.h5':
        d=pd.read_hdf(dlc_file)
        nd=len(d)
    else:
        with open(dlc_file,'rb') as f: nd=max(sum(1 for _ in f)-3,0)
    nc=len(c)
    if 'host_elapsed_from_first_frame_s' in c:
        t=pd.to_numeric(c.host_elapsed_from_first_frame_s,errors='coerce').to_numpy()
    elif 'host_capture_monotonic_s' in c:
        z=pd.to_numeric(c.host_capture_monotonic_s,errors='coerce').to_numpy();t=z-z[0]
    elif 'host_monotonic_s' in c:
        z=pd.to_numeric(c.host_monotonic_s,errors='coerce').to_numpy();t=z-z[0]
    else: raise ValueError('No recognized elapsed/host timestamp column')
    if not np.all(np.diff(t)>0): print('WARNING: camera host timestamps are not strictly increasing')
    has_q='video_queue_size_at_capture' in c
    report={'session':session.name if session else re.sub(r'(?i).*?([0-9]+[fm]_[at]_[a-z0-9]+).*',r'\1',dlc_file.name),'camera_records':nc,'dlc_rows':nd,'frame_difference':nc-nd,'difference_percent_of_camera':round(100*(nc-nd)/nc,3),'camera_duration_s':float(t[-1]-t[0]),'queue_column_present':has_q,'recovery_status':'No verified frame mapping: diagnostics only'}
    if has_q:
        q=pd.to_numeric(c.video_queue_size_at_capture,errors='coerce').to_numpy()
        cap=float(np.nanmax(q))
        report['maximum_observed_queue']=cap
        report['first_90pct_of_max_queue_s']=float(t[np.where(q>=.9*cap)[0][0]]) if np.any(q>=.9*cap) else None
        report['first_at_max_queue_s']=float(t[np.where(q==cap)[0][0]]) if np.any(q==cap) else None
        report['fraction_at_or_above_95pct_max']=float(np.nanmean(q>=.95*cap))
        report['fraction_at_max']=float(np.nanmean(q==cap))
        # 1-minute queue statistics. Full timestamp domain, no inferred dropped-frame positions.
        bins=np.floor(t/60).astype(int)
        gb=pd.DataFrame({'minute':bins,'queue':q}).groupby('minute').queue
        table=pd.DataFrame({'minute':gb.median().index,'queue_median':gb.median().values,'queue_p95':gb.quantile(.95).values,'queue_max':gb.max().values})
        table.to_csv(out/'queue_by_minute.csv',index=False)
        fig,ax=plt.subplots(figsize=(9,4))
        ax.plot(table.minute+0.5,table.queue_median,label='Median queue size')
        ax.plot(table.minute+0.5,table.queue_p95,label='95th percentile',alpha=.7)
        ax.set(xlabel='Session time (min)',ylabel='Frames waiting to be written',title=f'{report["session"]} — writer queue')
        ax.legend();fig.tight_layout();fig.savefig(out/'writer_queue.png',dpi=160);plt.close(fig)
    report['note']='Frame count mismatch plus queue saturation indicates writer overload, but does not identify which camera frame indices survived in MP4/DLC. Do not align by naive resampling for shock/transition analysis.'
    with open(out/'recovery_diagnostic.json','w') as f:json.dump(report,f,indent=2)
    print(json.dumps(report,indent=2));print('\nSaved diagnostics to',out)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--session');p.add_argument('--camera');p.add_argument('--dlc');p.add_argument('--output');a=p.parse_args()
    if not a.session and not (a.camera and a.dlc):p.error('Specify --session, or both --camera and --dlc')
    run(a.session,a.camera,a.dlc,a.output)
