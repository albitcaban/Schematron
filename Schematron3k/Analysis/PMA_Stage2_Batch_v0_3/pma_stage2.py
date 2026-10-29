#!/usr/bin/env python3
"""PMA Stage 2, version 0.4. Reads Stage1 corrected timing and ROI JSON.
Does not overwrite source data. Scientific classifications are provisional pending manual validation.
"""
import argparse, json, re, sys
from pathlib import Path
import numpy as np
import pandas as pd

PARTS={'nose':'Nose','left_ear':'Left Ear','right_ear':'Right Ear','implant':'Implant',
       'mid_back':'Mid-Back','left_hip':'Left Leg','right_hip':'Right Leg','tail_base':'Tailbase'}
HEAD=['nose','left_ear','right_ear','implant']
BODY=['mid_back','left_hip','right_hip','tail_base']

def identity(folder):
    m=re.fullmatch(r'(\d+)([fm])_([at])_((?:wt2?|tp\d+|lp\d+))',folder.name,re.I)
    if not m: raise ValueError(f'Unexpected folder name: {folder.name}; expected 0f_a_tp1')
    n,s,a,d=m.groups()
    return dict(animal=f'{n}{s.lower()}_{a.lower()}',mouse_number=n,sex={'f':'female','m':'male'}[s.lower()],age={'a':'adult','t':'adolescent'}[a.lower()],day=d.upper(),baseline=d.lower() in ('wt','wt2'))

def locate(folder,patterns,required=True):
    paths=[p for pat in patterns for p in folder.glob(pat) if p.is_file()]
    if not paths:
        if required: raise FileNotFoundError(f'{folder}: expected one of {patterns}')
        return None
    return sorted(set(paths), key=lambda p: (len(p.name),str(p)))[0]

def load_dlc(path):
    if path.suffix.lower()=='.h5':
        df=pd.read_hdf(path)
    else:
        df=pd.read_csv(path,header=[0,1,2],index_col=0)
    if not isinstance(df.columns,pd.MultiIndex) or df.columns.nlevels<2: raise ValueError('DLC tracking must have multi-index columns')
    return df

def coords(df,part):
    name=PARTS[part]
    cols=[c for c in df.columns if str(c[-2]).strip().lower()==name.lower()]
    if not cols: raise ValueError(f'DLC missing {name}')
    def get(axis):
        match=[c for c in cols if str(c[-1]).strip().lower()==axis]
        if not match: raise ValueError(f'DLC missing {name} {axis}')
        return pd.to_numeric(df[match[0]],errors='coerce').to_numpy(dtype=float)
    return get('x'), get('y'), get('likelihood')

def inside_poly(x,y,vertices):
    # Vectorized ray-casting, boundary within 1e-6 pixels is inside
    poly=np.asarray(vertices,dtype=float)
    if poly.shape!=(4,2): raise ValueError('Expected four ROI corners')
    good=np.isfinite(x)&np.isfinite(y)
    hit=np.zeros(len(x),dtype=bool)
    xx=np.where(good,x,0.); yy=np.where(good,y,0.)
    for j in range(len(poly)):
        a=poly[j];b=poly[(j+1)%len(poly)]
        cross=(a[1]>yy)!=(b[1]>yy)
        xcut=(b[0]-a[0])*(yy-a[1])/(b[1]-a[1]+1e-15)+a[0]
        hit ^= cross&(xx<xcut)
        dist=np.abs((xx-a[0])*(b[1]-a[1])-(yy-a[1])*(b[0]-a[0]))
        proj=(xx-a[0])*(b[0]-a[0])+(yy-a[1])*(b[1]-a[1])
        side=(dist<1e-6)&(proj>=0)&(proj<=np.sum((b-a)**2))
        hit |= side
    return hit&good

def bouts(mask,times,min_s=0.,max_gap=.15):
    """Return [start-index, end-exclusive-index] bouts; long time gaps split bouts."""
    mask=np.asarray(mask,dtype=bool); idx=np.flatnonzero(mask)
    if not len(idx):return []
    cuts=np.flatnonzero((np.diff(idx)>1)|(np.diff(times[idx])>max_gap))
    a=np.r_[0,cuts+1];b=np.r_[cuts,len(idx)-1]
    out=[]
    for aa,bb in zip(a,b):
        st=int(idx[aa]);end=int(idx[bb])+1
        dur=float(times[end-1]-times[st]) if end-st>1 else 0
        if dur>=min_s:out.append((st,end))
    return out

def durations(t):
    delta=np.diff(t,append=np.nan)
    med=np.nanmedian(np.diff(t)) if len(t)>1 else 0
    return np.where(np.isfinite(delta)&(delta>0)&(delta<=max(.15,3*med)),delta,0.)

def time_at_event_frame(t,when,tolerance=.12):
    i=np.searchsorted(t,when)
    candidates=[j for j in [i-1,i] if 0<=j<len(t)]
    if not candidates:return None
    k=min(candidates,key=lambda j:abs(t[j]-when))
    return k if abs(t[k]-when)<=tolerance else None

def intervals_for_session(ev,t):
    on=ev[ev.event.astype(str).str.upper().isin(['TONE_ON','LIGHT_ON'])].sort_values('corrected_session_s')
    off=ev[ev.event.astype(str).str.upper().isin(['TONE_OFF','LIGHT_OFF'])].sort_values('corrected_session_s')
    offs=off.corrected_session_s.to_numpy(float)
    results=[]
    for k,(_,v) in enumerate(on.iterrows()):
        start=float(v.corrected_session_s)
        future=offs[offs>start]
        end=float(future[0]) if len(future) else start+30.
        next_start=float(on.iloc[k+1].corrected_session_s) if k+1<len(on) else min(float(t[-1]) if len(t) else end+120,end+120)
        results.append(dict(trial=k+1,cue_on_s=start,cue_off_s=end,next_cue_s=next_start,cue_type=str(v.event)))
    return pd.DataFrame(results)

def stable_midback_events(t, x, y, valid, polygon, persistence_s=.2, boundary_px=3., max_gap_s=.15):
    """Hysteretic stable mid-back states and STRICTLY alternating transitions.

    State is retained through brief tracking loss, but events are never inferred
    across a long unobserved gap. Event times denote first observed crossing.
    A run counts only if its confirmed persistence is measured in actual seconds.
    """
    inside=inside_poly(x,y,polygon)
    poly=np.asarray(polygon,float)
    distance=np.full(len(t),np.inf)
    for a,b in zip(poly,np.roll(poly,-1,axis=0)):
        ab=b-a
        u=np.clip(((x-a[0])*ab[0]+(y-a[1])*ab[1]) / max(float(np.dot(ab,ab)),1e-12),0,1)
        distance=np.minimum(distance,np.hypot(x-(a[0]+u*ab[0]),y-(a[1]+u*ab[1])))
    observed=np.where(valid & (distance>=boundary_px),inside.astype(float),np.nan)
    # Retain confirmed state across small ambiguous boundary regions. After
    # a tracking gap, a change may be possible but must NOT generate an event.
    confirmed=np.full(len(t),np.nan)
    events=[];state=None;candidate=None;candidate_start=None;prev_observed=None
    for i in range(len(t)):
        gap=bool(i and (t[i]-t[i-1]>max_gap_s))
        if gap:
            candidate=None; candidate_start=None
            prev_observed=None
            # Do not claim an observed transition at a recording gap.
            state=None
        side=observed[i]
        if not np.isfinite(side):
            candidate=None;candidate_start=None;prev_observed=None
            if state is not None:confirmed[i]=state
            continue
        side=int(side)
        if state is None:
            if candidate!=side:
                candidate=side;candidate_start=i
            if (t[i]-t[candidate_start])>=persistence_s:
                state=side;candidate=None;candidate_start=None
        elif side==state:
            candidate=None;candidate_start=None
        else:
            if candidate!=side:
                candidate=side;candidate_start=i
            if (t[i]-t[candidate_start])>=persistence_s:
                # Timestamp is first observed crossing, not the confirmation frame.
                events.append((int(candidate_start),'platform_entry' if side else 'platform_exit',float(t[i]-t[candidate_start])))
                state=side;candidate=None;candidate_start=None
        if state is not None:confirmed[i]=state
        prev_observed=side
    # A 200-ms run can overlap a cue-offset. For time-local state queries,
    # mark first-crossing-to-confirmation frames UNKNOWN to avoid look-ahead.
    for row,_,duration in events:
        ending=np.searchsorted(t,t[row]+duration,side='left')
        confirmed[row:min(ending+1,len(t))]=np.nan
    return confirmed,events

def detect_head_dips(t,plat,valid,mid_state,body_events,max_transition=.5,return_window=3.,max_gap=.15):
    """Implant and >=1 additional head marker move outside and return while rear remains on.
    Reject any candidate containing a confirmed mid-back exit.
    """
    impl=valid['implant']
    other=['nose','left_ear','right_ear']
    other_valid=np.stack([valid[k] for k in other])
    other_out=np.stack([valid[k]&~plat[k] for k in other])
    head_out=impl & (~plat['implant']) & np.any(other_out,axis=0)
    head_in=impl & plat['implant'] & np.any(np.stack([valid[k]&plat[k] for k in other]),axis=0)
    rear_on=valid['tail_base']&plat['tail_base']&np.isfinite(mid_state)&(mid_state==1)
    exit_indices=np.asarray([r[0] for r in body_events if r[1]=='platform_exit'],dtype=int)
    out=[];active=False;start=None; prev=False
    for i in range(len(t)):
        if i and (t[i]-t[i-1]>max_gap):active=False
        if not active:
            if i and head_out[i] and not prev and rear_on[i]:
                active=True;start=i
        elif t[i]-t[start]>return_window or not rear_on[i]:
            active=False
        elif head_in[i]:
            if t[i]-t[start]<=return_window and not np.any((exit_indices>=start)&(exit_indices<=i)):
                # Rear must remain on in the initial confirmation period.
                end_confirm=np.searchsorted(t,t[start]+max_transition,side='right')
                relevant=rear_on[start:min(i+1,end_confirm)]
                if len(relevant) and np.all(relevant):out.append((start,'head_dip',float(t[i]-t[start])))
            active=False
        prev=bool(head_out[i])
    return out

def draw_events_from_states(frame,t,trial_table,plat,valid,xy,polygon,persistence=.2,boundary_px=3.,max_transition=.5):
    mid_state, transitions=stable_midback_events(t,*xy['mid_back'],valid['mid_back'],polygon,persistence,boundary_px)
    dips=detect_head_dips(t,plat,valid,mid_state,transitions,max_transition=max_transition)
    raw=sorted(transitions+dips,key=lambda z:(z[0],z[1]))
    out=pd.DataFrame(raw,columns=['frame_row','event','duration_s'])
    if out.empty:return pd.DataFrame(columns=['frame_row','time_s','event','duration_s','trial','period']),mid_state
    out['time_s']=t[out.frame_row.to_numpy(int)]
    def period(ts):
        for r in trial_table.itertuples():
            if r.cue_on_s<=ts<r.cue_off_s:return (r.trial,'cue')
        return (np.nan,'outside_cue')
    periods=[period(z) for z in out.time_s]
    out['trial']=[z[0] for z in periods];out['period']=[z[1] for z in periods]
    return out,mid_state

def detect_reward_visits(t, reward_state, persistence_s=.2, max_gap_s=.15):
    """Head-defined zone visits; any valid nose/ear inside=on, all valid outside=off.
    NaN is unknown. Require 200 ms of consecutive observed state before confirming.
    Do not invent crossings across missing observations or camera gaps. Returns
    events and paired visits; the final open visit is right-censored.
    """
    state=None; candidate=None; first=None; last_good=None
    events=[]; visits=[]; opened=None
    for i, value in enumerate(reward_state):
        if i and t[i]-t[i-1]>max_gap_s:
            if opened is not None:
                visits.append(dict(entry_frame=opened, exit_frame=np.nan, start_s=float(t[opened]),
                                   end_s=float(t[i-1]), duration_s=float(t[i-1]-t[opened]),
                                   exit_observed=False))
            opened=None;state=None;candidate=None;first=None;last_good=None
        if not np.isfinite(value):
            candidate=None; first=None;last_good=None
            continue
        side=int(value)
        if last_good is not None and t[i]-t[last_good]>max_gap_s:
            if opened is not None:
                visits.append(dict(entry_frame=opened, exit_frame=np.nan, start_s=float(t[opened]),
                                   end_s=float(t[last_good]), duration_s=float(t[last_good]-t[opened]),
                                   exit_observed=False))
            opened=None;state=None;candidate=None;first=None
        last_good=i
        if state==side:
            candidate=None;first=None
            continue
        if candidate!=side:
            candidate=side;first=i
        if t[i]-t[first]<persistence_s:
            continue
        if state is None:
            state=side
            # If observation starts inside, record a left-censored visit, not an entry.
            if side==1:opened=first
        else:
            events.append((int(first),'reward_entry' if side==1 else 'reward_exit',float(t[i]-t[first])))
            if side==1:opened=first
            elif opened is not None:
                visits.append(dict(entry_frame=opened,exit_frame=first,start_s=float(t[opened]),
                                   end_s=float(t[first]),duration_s=float(t[first]-t[opened]),exit_observed=True))
                opened=None
            state=side
        candidate=None;first=None
    if opened is not None:
        visits.append(dict(entry_frame=opened,exit_frame=np.nan,start_s=float(t[opened]),
                           end_s=float(t[-1]),duration_s=float(t[-1]-t[opened]),exit_observed=False))
    return events,pd.DataFrame(visits,columns=['entry_frame','exit_frame','start_s','end_s','duration_s','exit_observed'])

def analyze_one(folder,conf=.9,points=4,bin_s=2.,win_pre=-10.,win_post=40.,min_freeze=.9,back_vel=.59,angular_vel=15.,max_transition=.5,cm_per_pixel=None,transition_persistence=.2,boundary_px=3.,small_mismatch_max=20):
    folder=Path(folder).expanduser(); ident=identity(folder)
    outdir=folder/'PMA_ANALYSIS';outdir.mkdir(exist_ok=True)
    events=locate(folder/'PMA_QC',['events_corrected_timing.csv'],False)
    if events is None and ident['baseline']:
        events=locate(folder,['behavioral_events.csv'],False)
    if events is None: raise FileNotFoundError(f'Missing event data: {folder}/PMA_QC/events_corrected_timing.csv' + (' or behavioral_events.csv' if ident['baseline'] else ''))
    ev=pd.read_csv(events); ev['event']=ev.event.astype(str).str.upper().str.strip()
    if 'corrected_session_s' in ev:
        ev['corrected_session_s']=pd.to_numeric(ev.corrected_session_s,errors='coerce')
    elif ident['baseline'] and 'arduino_ms' in ev:
        # Only Arduino-relative timings are necessary for water training. First event is time zero.
        ms=pd.to_numeric(ev.arduino_ms,errors='coerce')
        ev['corrected_session_s']=(ms-ms.dropna().iloc[0])/1000. if ms.notna().any() else np.nan
    elif ident['baseline'] and 'host_monotonic_s' in ev:
        host=pd.to_numeric(ev.host_monotonic_s,errors='coerce')
        ev['corrected_session_s']=host-host.dropna().iloc[0] if host.notna().any() else np.nan
    else:
        raise ValueError(f'No usable timing column in {events}')
    result=dict(**ident,session=folder.name)
    # count events directly: BEAM_BREAK_TIMEOUT is a tag, not an additional poke
    result.update(total_pokes=int((ev.event=='BEAM_BREAK').sum()),total_rewards=int((ev.event=='REWARD').sum()),timeout_tag_count=int((ev.event=='BEAM_BREAK_TIMEOUT').sum()))
    if ident['baseline']:
        tt=ev['corrected_session_s'].dropna()
        duration=float(tt.max()-tt.min()) if len(tt)>1 else np.nan
        result.update(session_duration_s=duration,
                      unrewarded_pokes=max(0,result['total_pokes']-result['total_rewards']),
                      poke_rate_per_min=(result['total_pokes']/duration*60 if duration>0 else np.nan),
                      reward_rate_per_min=(result['total_rewards']/duration*60 if duration>0 else np.nan),
                      timing_source='Stage1 corrected events' if events.name=='events_corrected_timing.csv' else ('raw Arduino milliseconds' if 'arduino_ms' in ev else 'host monotonic'))
        pd.DataFrame([result]).to_csv(outdir/'session_summary.csv',index=False)
        ev.to_csv(outdir/'events_arduino_timed.csv',index=False)
        print(f'[baseline] {folder.name}: {result["total_pokes"]} pokes; {result["total_rewards"]} rewards')
        return
    mapping=locate(folder/'PMA_QC',['frame_time_mapping.csv'])
    roi_file=locate(folder,['zone_rois.json'])
    roi=json.loads(roi_file.read_text())
    if not all(k in roi.get('zones',{}) for k in ('platform','reward')):raise ValueError('Two zones required')
    dlcfile=locate(folder,['*filtered.h5','*filtered.csv'])
    dlc=load_dlc(dlcfile)
    mapping=pd.read_csv(mapping)
    if len(mapping)!=len(dlc):
        diff=abs(len(mapping)-len(dlc))
        if diff>=small_mismatch_max:
            raise ValueError(f'Mismatch: {len(mapping)} camera timestamps, {len(dlc)} DLC rows; discrepancy {diff} >= {small_mismatch_max}; recovery needed')
        # Permissive mode for tiny discrepancies: monotone endpoint-preserving nearest-index
        # assignment. NEVER present this as verified original frame identity.
        sample_positions=np.rint(np.linspace(0,len(mapping)-1,len(dlc))).astype(int)
        mapping=mapping.iloc[sample_positions].copy().reset_index(drop=True)
        mapping['dlc_row_zero_based']=np.arange(len(dlc))
        (outdir/'small_mismatch_alignment.json').write_text(json.dumps({
            'status':'approximate_timing_unverified',
            'method':'endpoint-preserving monotonically rounded camera timestamp assignment',
            'camera_rows_original':int(len(pd.read_csv(folder/'PMA_QC'/'frame_time_mapping.csv',usecols=['frame_index']))),
            'dlc_rows':int(len(dlc)), 'absolute_discrepancy':int(diff),
            'maximum_frame_index_ambiguity':int(diff),
            'caution':'Exact missing/extra frame positions unknown; time-critical transition and shock outcomes require review.'
        },indent=2))
        print(f'[WARNING] {folder.name}: {diff}-frame discrepancy accepted approximately (<{small_mismatch_max}); exact mapping UNKNOWN; see small_mismatch_alignment.json')
    t=mapping.corrected_session_s.to_numpy(float)
    if np.any(np.diff(t)<=0):raise ValueError('Nonmonotonic camera timestamps')
    if np.any(np.diff(mapping.dlc_row_zero_based.to_numpy(int))!=1) or mapping.dlc_row_zero_based.iloc[0]!=0:raise ValueError('Camera/DLC alignment not sequential')
    poly_plat=roi['zones']['platform']['polygon_px'];poly_reward=roi['zones']['reward']['polygon_px']
    col={};valid={};plat={};reward={};xy={}
    for part in PARTS:
        x,y,p=coords(dlc,part);v=(p>=conf)&np.isfinite(x)&np.isfinite(y)
        xy[part]=(x,y);valid[part]=v
        plat[part]=inside_poly(x,y,poly_plat)&v
        reward[part]=inside_poly(x,y,poly_reward)&v
    nvalid=np.sum(np.stack(list(valid.values())),axis=0)
    nplat=np.sum(np.stack(list(plat.values())),axis=0)
    # Unknown if number of missing points makes threshold unattainable/uncertain
    on=np.where((nplat>=points),1.,np.where(nplat+(len(PARTS)-nvalid)<points,0.,np.nan))
    rvalid=np.stack([valid[k] for k in ['nose','left_ear','right_ear']])
    rin=np.stack([reward[k] for k in ['nose','left_ear','right_ear']])
    reward_bool=np.any(rin,axis=0)
    reward_state=np.where(reward_bool,1.,np.where(np.all(rvalid,axis=0),0.,np.nan))
    # priority platform > reward > grid, unknown retained
    zone=np.full(len(t),'unknown',dtype=object)
    zone[(on==0)&(reward_state==0)]='grid'
    zone[(on==0)&(reward_state==1)]='reward'
    zone[on==1]='platform'
    head_in=np.any(np.stack([plat[k] for k in HEAD]),axis=0)
    head_valid=np.any(np.stack([valid[k] for k in HEAD]),axis=0)
    rear_valid=valid['mid_back']&valid['tail_base']
    rear_in=plat['mid_back']&plat['tail_base']
    # Movement in cm/s requires user-provided chamber corner calibration.
    # Use pixel velocity as descriptive data; freezing remains unscored without spatial calibration.
    midx,midy=xy['mid_back']; dt=np.diff(t,prepend=np.nan)
    dx=np.diff(midx,prepend=np.nan);dy=np.diff(midy,prepend=np.nan)
    velocity_px=np.hypot(dx,dy)/dt
    velocity_px[~(valid['mid_back']&np.r_[False,valid['mid_back'][:-1]])]=np.nan
    velocity_px[(dt<=0)|(dt>.15)]=np.nan
    # Head direction: mid-back -> implant, angular velocity (degrees/s)
    ix,iy=xy['implant']; ang=np.unwrap(np.arctan2(iy-midy,ix-midx))
    head_angular=np.abs(np.diff(ang,prepend=np.nan))/dt*180/np.pi
    head_angular[~(valid['implant']&valid['mid_back'])]=np.nan
    head_angular[(dt<=0)|(dt>.15)]=np.nan
    freeze=np.full(len(t),np.nan)
    if cm_per_pixel is not None:
        if cm_per_pixel<=0:raise ValueError('cm-per-pixel must be positive')
        cmvel=velocity_px*cm_per_pixel
        candidate=(cmvel<back_vel)&(head_angular<angular_vel)&np.isfinite(cmvel)&np.isfinite(head_angular)
        freeze[:]=0.
        freeze[~(np.isfinite(cmvel)&np.isfinite(head_angular))]=np.nan
        for b0,b1 in bouts(candidate,t,min_s=min_freeze,max_gap=.15):
            freeze[b0:b1]=1.
    else:
        cmvel=np.full(len(t),np.nan)
    # Nominal 7.4 x 8.1 inch chamber dimensions alone do not determine pixel scale.

    frame=pd.DataFrame(dict(frame_index=mapping.frame_index,time_s=t,platform=on,reward=reward_state,zone=zone,valid_points=nvalid,
                            head_any_inside=head_in,head_valid=head_valid,rear_both_inside=rear_in,rear_valid=rear_valid,
                            midback_velocity_px_s=velocity_px,midback_velocity_cm_s=cmvel,head_angular_velocity_deg_s=head_angular,freezing=freeze))
    trials=intervals_for_session(ev,t)
    behavioral,midstate=draw_events_from_states(frame,t,trials,plat,valid,xy,poly_plat,
        persistence=transition_persistence,boundary_px=boundary_px,max_transition=max_transition)
    frame['midback_transition_state']=midstate
    # Reward-zone state is independent of platform occupancy and includes
    # nose/ear investigation even when the animal is also on the platform.
    reward_events, visits=detect_reward_visits(t,reward_state,persistence_s=transition_persistence)
    if reward_events:
        rw=pd.DataFrame(reward_events,columns=['frame_row','event','duration_s'])
        rw['time_s']=t[rw.frame_row.to_numpy(int)]
        rw['trial']=np.nan;rw['period']='outside_cue'
        for tr in trials.itertuples():
            mask=(rw.time_s>=tr.cue_on_s)&(rw.time_s<tr.cue_off_s)
            rw.loc[mask,'trial']=int(tr.trial)
            rw.loc[mask,'period']='cue'
        behavioral=pd.concat([behavioral,rw],ignore_index=True).sort_values('time_s',kind='stable').reset_index(drop=True)
    # Export visits separately, with optional Arduino event associations.
    poke_times=ev.loc[ev.event=='BEAM_BREAK','corrected_session_s'].dropna().to_numpy(float)
    reward_times=ev.loc[ev.event=='REWARD','corrected_session_s'].dropna().to_numpy(float)
    if len(visits):
        visits['poke_count']= [int(np.sum((poke_times>=a)&(poke_times<b))) for a,b in zip(visits.start_s,visits.end_s)]
        visits['reward_count']=[int(np.sum((reward_times>=a)&(reward_times<b))) for a,b in zip(visits.start_s,visits.end_s)]
        visits['has_poke']=visits.poke_count.gt(0)
        visits['has_reward']=visits.reward_count.gt(0)
    visits.to_csv(outdir/'reward_visits.csv',index=False)
    if len(behavioral):behavioral.to_csv(outdir/'behavior_events.csv',index=False)
    else:pd.DataFrame(columns=['frame_row','time_s','event','duration_s','trial','period']).to_csv(outdir/'behavior_events.csv',index=False)
    shock=ev[ev.event=='SHOCK_ON'].corrected_session_s.to_numpy(float)
    pokes=ev[ev.event=='BEAM_BREAK'].corrected_session_s.to_numpy(float)
    rewards=ev[ev.event=='REWARD'].corrected_session_s.to_numpy(float)
    weights=durations(t)
    def stats(a,b):
        take=(t>=a)&(t<b)
        w=weights[take]; z=zone[take]; p=on[take]
        valid_dur=np.sum(w[z!='unknown'])
        out={f'{k}_occupancy_pct':float(100*np.sum(w[z==k])/valid_dur) if valid_dur else np.nan for k in ('platform','reward','grid')}
        fr=freeze[take]; f_valid=np.isfinite(fr)
        out.update(freezing_pct=float(100*np.sum(w[(fr==1)&f_valid])/np.sum(w[f_valid])) if np.sum(w[f_valid])>0 else np.nan)
        rr=reward_state[take]
        rw=w[np.isfinite(rr)]
        out['reward_head_occupancy_pct']=float(100*np.sum(w[rr==1])/np.sum(rw)) if np.sum(rw)>0 else np.nan
        out.update(platform_valid_duration_s=float(np.sum(w[np.isfinite(p)])),zone_valid_duration_s=float(valid_dur),reward_valid_duration_s=float(np.sum(rw)))
        return out
    edges=np.arange(win_pre,win_post+bin_s/2,bin_s)
    if len(edges)<2:raise ValueError('Bin window too small')
    bins=[];rows=[]
    # Baseline is session start to first cue; ITI starts at each cue offset.
    first_cue=float(trials.cue_on_s.min()) if len(trials) else float(t[-1])
    baseline_stats=stats(float(t[0]),first_cue)
    baseline_counts={f'{event}_count':int(((behavioral.event==event)&(behavioral.time_s>=float(t[0]))&(behavioral.time_s<first_cue)).sum()) for event in ('reward_entry','reward_exit','platform_entry','platform_exit','head_dip')}
    pd.DataFrame([dict(**ident,session=folder.name,start_s=float(t[0]),end_s=first_cue,
                       **baseline_stats,**baseline_counts)]).to_csv(outdir/'baseline_summary.csv',index=False)
    def visit_metrics(a,b):
        if not len(visits):return dict(reward_visit_count=0,reward_visit_total_duration_s=0.,reward_visit_mean_duration_s=np.nan,reward_visits_with_poke=0,reward_visits_with_reward=0)
        # A visit spanning cue/ITI contributes time to both periods, but its
        # event count is assigned only to the period containing entry.
        entry=visits[(visits.start_s>=a)&(visits.start_s<b)]
        overlap=np.maximum(0.,np.minimum(visits.end_s.to_numpy(float),b)-np.maximum(visits.start_s.to_numpy(float),a))
        n=int(len(entry))
        return dict(reward_visit_count=n,reward_visit_total_duration_s=float(np.sum(overlap)),
                    reward_visit_mean_duration_s=float(entry.duration_s.mean()) if n else np.nan,
                    reward_visits_with_poke=int(entry.has_poke.sum()) if n else 0,
                    reward_visits_with_reward=int(entry.has_reward.sum()) if n else 0,
                    reward_visit_rate_per_min=float(60*n/(b-a)) if b>a else np.nan)
    for r in trials.itertuples():
        start=float(r.cue_on_s);end=float(r.cue_off_s)
        following=min(float(r.next_cue_s),float(t[-1]))
        offs=behavioral[(behavioral.event=='platform_exit') & (behavioral.time_s>=end)&(behavioral.time_s<following)] if len(behavioral) else behavioral
        at=time_at_event_frame(t,end,tolerance=.15)
        body_at_offset=midstate[at] if at is not None else np.nan
        if not np.isfinite(body_at_offset):
            lat=np.nan; observed=np.nan; exit_status='unknown_at_cue_offset'
        elif body_at_offset==0:
            lat=0.;observed=False;exit_status='already_off_platform'
        elif len(offs):
            lat=float(offs.time_s.iloc[0]-end);observed=True;exit_status='observed_exit'
        else:
            lat=max(0.,following-end);observed=False;exit_status='censored_no_exit'
        # Do not exceed 120s even when last-session recording continues longer.
        lat=min(lat,120.) if np.isfinite(lat) else lat
        in_shock=shock[(shock>=start)&(shock<end+0.5)]
        shockstates=[]
        for tm in in_shock:
            if tm<end-3.5:continue
            ii=time_at_event_frame(t,tm)
            shockstates.append(np.nan if ii is None else (1.-on[ii] if np.isfinite(on[ii]) else np.nan))
        shock_received=float(np.nanmax(shockstates)) if len(shockstates) and np.any(np.isfinite(shockstates)) else (0. if not len(in_shock) else np.nan)
        row=dict(**ident,session=folder.name,trial=int(r.trial),cue_on_s=start,cue_off_s=end,shock_received=shock_received,
                 exit_latency_s=lat,exit_observed=observed,exit_status=exit_status,midback_at_cue_offset=body_at_offset,
                 poke_count_cue=int(np.sum((pokes>=start)&(pokes<end))),reward_count_cue=int(np.sum((rewards>=start)&(rewards<end))))
        row.update({f'cue_{k}':v for k,v in stats(start,end).items()})
        row.update({f'iti_{k}':v for k,v in stats(end,following).items()})
        row.update({f'cue_{k}':v for k,v in visit_metrics(start,end).items()})
        row.update({f'iti_{k}':v for k,v in visit_metrics(end,following).items()})
        next_reward=behavioral[(behavioral.event=='reward_entry') & (behavioral.time_s>=start) & (behavioral.time_s<end)]
        at_cue=time_at_event_frame(t,start,tolerance=.15)
        if at_cue is not None and reward_state[at_cue]==1:
            row['reward_approach_latency_s']=0.;row['reward_approach_status']='already_in_zone'
        elif len(next_reward):
            row['reward_approach_latency_s']=float(next_reward.time_s.iloc[0]-start);row['reward_approach_status']='observed_entry'
        else:
            row['reward_approach_latency_s']=np.nan;row['reward_approach_status']='no_observed_entry' 
        for event in ('head_dip','platform_entry','platform_exit','reward_entry','reward_exit'):
            sub=behavioral[behavioral.event==event] if len(behavioral) else behavioral
            row[f'{event}_cue_count']=int(np.sum((sub.time_s>=start)&(sub.time_s<end))) if len(sub) else 0
            row[f'{event}_iti_count']=int(np.sum((sub.time_s>=end)&(sub.time_s<following))) if len(sub) else 0
        rows.append(row)
        for a,b in zip(edges[:-1],edges[1:]):
            mid=(a+b)/2; pos=start+mid; ix=time_at_event_frame(t,pos,tolerance=.12)
            platform_val=on[ix] if ix is not None else np.nan
            reward_val=reward_state[ix] if ix is not None else np.nan
            poke_val=float(np.any((pokes>=start+a)&(pokes<start+b)))
            bins.append(dict(**ident,session=folder.name,trial=int(r.trial),bin_start_s=float(a),bin_end_s=float(b),bin_midpoint_s=float(mid),platform_binary=platform_val,reward_zone_binary=reward_val,poke_binary=poke_val))
    frame.to_csv(outdir/'frame_behaviors.csv',index=False)
    pd.DataFrame(rows).to_csv(outdir/'trial_summary.csv',index=False)
    bin_df=pd.DataFrame(bins)
    bin_df.to_csv(outdir/'trial_probability_bins.csv',index=False)
    if len(bin_df):
        probs=bin_df.groupby('bin_midpoint_s',as_index=False).agg(platform_probability=('platform_binary','mean'),reward_zone_probability=('reward_zone_binary','mean'),poke_probability=('poke_binary','mean'),valid_platform_trials=('platform_binary','count'),valid_reward_zone_trials=('reward_zone_binary','count'),n_trials=('trial','count'))
        probs.to_csv(outdir/'session_probabilities.csv',index=False)
    result.update(total_reward_visits=len(visits),total_reward_zone_entries=int((behavioral.event=='reward_entry').sum()) if len(behavioral) else 0,total_reward_zone_exits=int((behavioral.event=='reward_exit').sum()) if len(behavioral) else 0,total_trials=len(trials),shock_received_count=int(pd.Series([z['shock_received'] for z in rows]).eq(1).sum()),
                  platform_threshold=points,tracking_confidence=conf,transition_persistence_ms=transition_persistence*1000,boundary_tolerance_px=boundary_px,freezing_status=('scored with user-provided pixel scale' if cm_per_pixel else 'requires calibrated cm_per_pixel; not scored'),
                  stage1_timing='corrected_session_s',roi_file=str(roi_file))
    pd.DataFrame([result]).to_csv(outdir/'session_summary.csv',index=False)
    # Save convenient preliminary plot with matching time axis for the two probabilities.
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,ax=plt.subplots(figsize=(8,4.5))
        pp=pd.read_csv(outdir/'session_probabilities.csv')
        ax.plot(pp.bin_midpoint_s,pp.platform_probability,label='Platform state at bin midpoint')
        ax.plot(pp.bin_midpoint_s,pp.poke_probability,label='Any beam break in 2-s bin')
        ax.plot(pp.bin_midpoint_s,pp.reward_zone_probability,label='Reward-zone head at bin midpoint')
        ax.axvspan(0,30,alpha=.12,color='gray',label='Cue')
        ax.set(xlabel='Seconds relative to cue onset',ylabel='Fraction of valid trials',ylim=(0,1),title=f'{folder.name}: preliminary behavioral probabilities')
        ax.legend(fontsize=8);fig.tight_layout();fig.savefig(outdir/'poke_platform_probability.png',dpi=160);plt.close(fig)
    except ImportError:pass
    print(f'[ok] {folder.name}: {len(trials)} trials; {len(t)} frames; results in {outdir}; freezing: {result["freezing_status"]}')

def main():
    p=argparse.ArgumentParser(description=__doc__)
    g=p.add_mutually_exclusive_group(required=True);g.add_argument('--session',type=Path);g.add_argument('--root',type=Path)
    p.add_argument('--confidence',type=float,default=.9);p.add_argument('--platform-points',type=int,default=4)
    p.add_argument('--transition-ms',type=float,default=500.,help='Head dip rear confirmation window')
    p.add_argument('--midback-persistence-ms',type=float,default=200.,help='Minimum stable duration after midback crossing')
    p.add_argument('--boundary-px',type=float,default=3.,help='Exclude edge-adjacent midback tracking from transitions')
    p.add_argument('--small-mismatch-max',type=int,default=20,help='Allow approximate alignment only for discrepancies strictly below this number of frames')
    p.add_argument('--cm-per-pixel',type=float,default=None,help='Optional calibrated scale enabling freezing scoring')
    args=p.parse_args()
    folders=[args.session] if args.session else [f for f in args.root.rglob('*') if f.is_dir() and re.fullmatch(r'\d+[fm]_[at]_(wt2?|tp\d+|lp\d+)',f.name,re.I)]
    if not folders:raise SystemExit('No session folders found.')
    failed=[]
    for f in sorted(folders):
        try:analyze_one(f,conf=args.confidence,points=args.platform_points,max_transition=args.transition_ms/1000,cm_per_pixel=args.cm_per_pixel,transition_persistence=args.midback_persistence_ms/1000,boundary_px=args.boundary_px,small_mismatch_max=args.small_mismatch_max)
        except Exception as e:
            print(f'[ERROR] {f}: {e}',file=sys.stderr);failed.append((str(f),str(e)))
    if failed:
        print(f'{len(failed)} sessions failed validation',file=sys.stderr);sys.exit(1)
if __name__=='__main__':main()
