#!/usr/bin/env python3
"""Diagnostic ONLY: test end-of-video truncation vs distributed frame loss.
Does NOT modify Stage 1 / Stage 2 data or certify a mapping.
"""
import argparse, json, re
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def find_file(folder, patterns):
    candidates=[]
    for pat in patterns: candidates.extend(p for p in folder.glob(pat) if p.is_file())
    if not candidates: raise FileNotFoundError(f'Missing {patterns} in {folder}')
    candidates=sorted(set(candidates),key=lambda p:(len(p.name),p.name))
    return candidates[0]

def load_dlc(path):
    d=pd.read_hdf(path) if path.suffix.lower()=='.h5' else pd.read_csv(path,header=[0,1,2],index_col=0)
    if not isinstance(d.columns,pd.MultiIndex): raise ValueError('DLC requires three-level headers')
    return d

def point(d,name):
    def grab(axis):
        matches=[c for c in d.columns if str(c[-2]).lower().strip()==name.lower() and str(c[-1]).lower().strip()==axis]
        return pd.to_numeric(d[matches[0]],errors='coerce').to_numpy(float) if matches else np.full(len(d),np.nan)
    return grab('x'),grab('y'),grab('likelihood')

def inside_poly(x,y,poly):
    poly=np.asarray(poly,float)
    good=np.isfinite(x)&np.isfinite(y)
    hit=np.zeros(len(x),bool)
    for j in range(len(poly)):
        a,b=poly[j],poly[(j+1)%len(poly)]
        cond=(a[1]>y)!=(b[1]>y)
        cut=(b[0]-a[0])*(y-a[1])/(b[1]-a[1]+1e-12)+a[0]
        hit ^= cond & (x<cut) & good
    return hit

def time_score(times,head_roi,valid,pokes,start,end,window=.2):
    p=pokes[(pokes>=start)&(pokes<end)]
    if len(p)==0:return dict(pokes=0,aligned=np.nan,control=np.nan,lift=np.nan,known=0)
    def evaluate(ts):
        ix=np.searchsorted(times,ts)
        matched=[];known=[]
        for k,t in zip(ix,ts):
            lo=max(0,np.searchsorted(times,t-window))
            hi=min(len(times),np.searchsorted(times,t+window,side='right'))
            known.append(bool(hi>lo and np.any(valid[lo:hi])))
            matched.append(bool(hi>lo and np.any(head_roi[lo:hi]&valid[lo:hi])))
        return np.asarray(matched),np.asarray(known)
    m,v=evaluate(p)
    controls=[]
    for shift in (2.5,5.0,7.5):
        q=p+shift
        q=q[(q>=start)&(q<end)]
        mc,vc=evaluate(q)
        if np.any(vc):controls.append(float(mc[vc].mean()))
    aligned=float(m[v].mean()) if np.any(v) else np.nan
    control=float(np.mean(controls)) if controls else np.nan
    return dict(pokes=int(len(p)),aligned=aligned,control=control,lift=aligned-control if np.isfinite(aligned) and np.isfinite(control) else np.nan,known=int(v.sum()))

def run(session,confidence=.8,window=.2):
    folder=Path(session).expanduser().resolve()
    cam=pd.read_csv(find_file(folder,['PMA_QC/frame_time_mapping.csv']))
    times=pd.to_numeric(cam['corrected_session_s'],errors='coerce').to_numpy(float)
    if not np.all(np.isfinite(times)) or not np.all(np.diff(times)>0):raise ValueError('Camera time map not strictly increasing')
    roi=json.loads((folder/'zone_rois.json').read_text())
    poly=roi['zones']['reward']['polygon_px']
    dlcfile=find_file(folder,['*filtered.h5','*filtered.csv'])
    d=load_dlc(dlcfile);n=len(d);m=len(times)
    if n>m:raise ValueError('Not a tail-truncation candidate: DLC has extra rows')
    x,y,lik=point(d,'Nose'); lx,ly,ll=point(d,'Left Ear'); rx,ry,rl=point(d,'Right Ear')
    valid=(np.isfinite(x)&np.isfinite(y)&(lik>=confidence))
    # Strict nose ROI: candidate must have reliable nose in saved reward rectangle.
    head=inside_poly(x,y,poly)
    # Also score the independent union-of-head-landmarks criterion as a diagnostic.
    hv=(valid & head) | ((ll>=confidence)&inside_poly(lx,ly,poly)) | ((rl>=confidence)&inside_poly(rx,ry,poly))
    evpath=find_file(folder,['PMA_QC/events_corrected_timing.csv','behavioral_events.csv'])
    ev=pd.read_csv(evpath)
    if 'corrected_session_s' in ev.columns:
        evt=pd.to_numeric(ev.corrected_session_s,errors='coerce').to_numpy(float)
    elif 'arduino_ms' in ev.columns:
        evt=pd.to_numeric(ev.arduino_ms,errors='coerce').to_numpy(float)/1000
    else:raise ValueError('Cannot find Arduino time column')
    typ=ev.event.astype(str).str.upper().to_numpy()
    pokes=evt[(typ=='BEAM_BREAK') & np.isfinite(evt)]
    on=evt[np.isin(typ,['TONE_ON','LIGHT_ON']) & np.isfinite(evt)]
    off=evt[np.isin(typ,['TONE_OFF','LIGHT_OFF']) & np.isfinite(evt)]
    # Candidate 1: no missing frames before surviving video ends.
    tail=times[:n]
    # Candidate 2: distributed loss: count-linear remapping (null hypothesis, not actual recovery).
    uniform=np.interp(np.linspace(0,m-1,n),np.arange(m),times)
    rows=[]
    for label,tt in [('tail_only',tail),('distributed_uniform',uniform)]:
        for start in np.arange(0,max(times[-1],1800),300):
            end=float(start+300)
            s=time_score(tt,head,valid,pokes,start,end,window)
            rows.append(dict(mapping=label,window_start_s=float(start),window_end_s=end,**s))
    scores=pd.DataFrame(rows)
    output=folder/'PMA_RECOVERY_TEST';output.mkdir(exist_ok=True)
    scores.to_csv(output/'nosepoke_alignment_by_5min.csv',index=False)
    trial=[]
    for j,cue in enumerate(on):
        offset=off[off>cue][0] if np.any(off>cue) else cue+30
        required=min(float(times[-1]),float(offset+120))
        trial.append(dict(trial=j+1,cue_on_s=float(cue),cue_off_s=float(offset),needed_through_s=float(required),tail_coverage=bool(tail[-1]>=required),uniform_coverage=bool(uniform[-1]>=required)))
    trials=pd.DataFrame(trial);trials.to_csv(output/'trial_window_coverage.csv',index=False)
    fig,ax=plt.subplots(figsize=(10,5))
    for label,g in scores.groupby('mapping'):
        ax.plot((g.window_start_s+g.window_end_s)/120,g.lift,marker='o',label=label)
    ax.axhline(0,color='black',lw=.6)
    ax.set(xlabel='Session minute',ylabel='Poke-alignment lift over time-shifted control',title=f'{folder.name}: alignment diagnostic (not validation)')
    ax.legend();fig.tight_layout();fig.savefig(output/'nosepoke_alignment_comparison.png',dpi=160);plt.close(fig)
    summary={'session':folder.name,'camera_frames':int(m),'dlc_rows':int(n),'unmatched_camera_frames':int(m-n),'camera_end_s':float(times[-1]),'tail_candidate_video_end_s':float(tail[-1]),'uniform_candidate_end_s':float(uniform[-1]),'reward_roi':poly,'DLC_file':dlcfile.name,'score_note':'Poke alignment inside ROI compared with shifted non-poke times. Broad ROI can inflate both; compare lift and trends and inspect manual frames. NOT sufficient to certify exact frame mapping.','tail_coverage_trial_windows':int(trials.tail_coverage.sum()),'total_trials':int(len(trials)),'status':'diagnostic_only_not_recovered'}
    (output/'recovery_test_report.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2))
    print('\nScores:');print(scores.to_string(index=False,float_format=lambda z:f'{z:.3f}'))
    print('\nResults:',output)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--session',required=True)
    p.add_argument('--confidence',type=float,default=.8)
    p.add_argument('--window',type=float,default=.2,help='Time tolerance in seconds for nose / Arduino poke coincidence')
    a=p.parse_args();run(a.session,a.confidence,a.window)
