#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 28 16:47:48 2026

@author: albitcaban
"""

# -*- coding: utf-8 -*-
"""PMA behavioral group analysis v1.3, adult vs adolescent. Open in Spyder; run cells (# %%).
Inputs: PMA_BATCH/*.csv produced by PMA Stage2 v0.5. No DLC reprocessing.
Statistical unit is the MOUSE (animal); days are not independent animals.
Per-day Mann-Whitney tests are exploratory; BH-FDR corrects across days PER METRIC.
Do not interpret missing/unknown shock states as successful avoidance.
"""
# %% 0. LIBRARIES & USER SETTINGS — EDIT THIS CELL FIRST
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy import stats

ROOT = Path('/Volumes/Extreme SSD/schema')
BATCH = ROOT / 'PMA_BATCH'
OUT = BATCH / 'GROUP_FIGURES_v1_3'
OUT.mkdir(parents=True, exist_ok=True)

# All visual settings in one place (RGB/hex colors may be changed freely).
COLORS = {'adult':'#0a2463', 'adolescent':'#70d6ff',
          'platform':'#fb3640', 'poke':'#007e5d', 'reward':'b0db43',
          'grid':'939597', 'points':'#2B313A'}
DAYS = ['TP1','TP2','TP3','TP4','LP1','LP2']
GROUPS = ['adult','adolescent']
DPI = 300
FIG_EXT = 'png'  # png or pdf; DPI applies to raster export
FIGSIZE = (7.1,4.7)
FONT_SIZE = 11
LINE_WIDTH = 2.8
SEM_ALPHA = 0.17
POINT_SIZE = 24
JITTER = 0.065
BOX_WIDTH = 0.47
MIN_MICE_FOR_TEST = 2 # don't report group p-values below this
SHOW_SAMPLE_SIZE_LABELS = True
SHOW_FIGURES = True
SAVE_PDF_ALSO = False
SEED = 21


AXIS_WIDTH = 2.0        # Thickness of X and Y axes
TICK_WIDTH = 1.5        # Thickness of tick marks
TICK_LENGTH = 5         # Length of tick marks

plt.rcParams.update({
    'axes.linewidth': AXIS_WIDTH,
    'xtick.major.width': TICK_WIDTH,
    'ytick.major.width': TICK_WIDTH,
    'xtick.major.size': TICK_LENGTH,
    'ytick.major.size': TICK_LENGTH,
    'axes.grid': False,
})

plt.rcParams["axes.grid"] = False
# Session policy (video-derived variables from 3f_a_tp3 must never be included).
EXCLUDE_VIDEO = {'3f_a_tp3'}
ASSUMED_TAIL = {'2f_a_tp4','5m_t_tp1','0m_a_tp3'}
# Set False for conservative sensitivity analysis; independently saved output directory.
INCLUDE_ASSUMED_TAIL = True
if not INCLUDE_ASSUMED_TAIL:
    OUT = OUT / 'EXCLUDING_ASSUMED_TAIL'
    OUT.mkdir(parents=True, exist_ok=True)

# Observation threshold for cue percentage outcomes (typically 30 s of cue).
MIN_VALID_CUE_S = 15.0  # configurable sensitivity cutoff, seconds / 30-second cue
# Classify 1=shock, 0=no shock only when shock_received is known.
# This is DLC-assisted shock avoidance, NOT independent Arduino-only outcome.

mpl.rcParams.update({
    'font.family':'DejaVu Sans', 'font.size':FONT_SIZE, 'axes.titlesize':13,
    'axes.titleweight':'semibold', 'axes.labelsize':11,
    'axes.spines.top':False, 'axes.spines.right':False,
    'axes.linewidth':1, 'xtick.direction':'out','ytick.direction':'out',
    'savefig.dpi':DPI, 'figure.dpi':110, 'figure.facecolor':'white',
    'axes.facecolor':'white', 'axes.edgecolor':'#444444',
    'text.color':'#232323', 'axes.labelcolor':'#232323',
    'xtick.color':'#444444', 'ytick.color':'#444444'})
rng = np.random.default_rng(SEED)

def export(fig, name):
    fig.savefig(OUT / (name+'.'+FIG_EXT), dpi=DPI, bbox_inches='tight', facecolor='white')
    if SAVE_PDF_ALSO:
        fig.savefig(OUT / (name+'.pdf'), bbox_inches='tight', facecolor='white')
    if SHOW_FIGURES: plt.show()
    else: plt.close(fig)

def clean_axis(ax, ylabel=None, title=None):
    ax.grid(axis='y',alpha=.13,zorder=0)
    if ylabel: ax.set_ylabel(ylabel)
    if title: ax.set_title(title,loc='left',pad=14)

def sem(x):
    a = pd.to_numeric(pd.Series(x),errors='coerce').dropna().to_numpy()
    return float(np.std(a,ddof=1)/np.sqrt(len(a))) if len(a)>1 else np.nan

def ensure(df, columns, label):
    missing=[c for c in columns if c not in df.columns]
    if missing: raise ValueError(f'{label}: missing columns {missing}; check Stage 2 version')

def normalize(df, exclude_video=True):
    if df.empty:return df
    if 'source_session' in df.columns:
        df['session'] = df['source_session'].astype(str)
    elif 'session' not in df.columns: raise ValueError('Missing session/source_session column')
    df['session'] = df['session'].astype(str)
    if 'animal' not in df.columns: df['animal']=df['session'].str.rsplit('_',n=1).str[0]
    if 'age' not in df.columns: df['age']=np.where(df['animal'].str.endswith('_a'),'adult','adolescent')
    df['age']=df['age'].astype(str).str.lower().replace({'teen':'adolescent','teenage':'adolescent'})
    if 'day' not in df.columns: df['day']=df['session'].str.split('_').str[-1].str.upper()
    df['day']=df['day'].astype(str).str.upper()
    df=df[df['day'].isin(DAYS)&df['age'].isin(GROUPS)].copy()
    if exclude_video: df=df[~df['session'].isin(EXCLUDE_VIDEO)]
    if not INCLUDE_ASSUMED_TAIL:df=df[~df['session'].isin(ASSUMED_TAIL)]
    return df

def read(name, exclude_video=True):
    p=BATCH/name
    if not p.is_file():
        warnings.warn(f'Not found: {p}; corresponding plots will be skipped')
        return pd.DataFrame()
    return normalize(pd.read_csv(p,low_memory=False), exclude_video=exclude_video)

def bh_fdr(pvalues):
    """Benjamini-Hochberg across the available day-comparisons for a metric."""
    x=np.asarray(pvalues,float); out=np.full(len(x),np.nan)
    ix=np.flatnonzero(np.isfinite(x)); n=len(ix)
    if n:
        order=ix[np.argsort(x[ix])]
        adjusted=x[order]*n/np.arange(1,n+1)
        out[order]=np.minimum.accumulate(adjusted[::-1])[::-1].clip(0,1)
    return out

def p_label(p):
    if pd.isna(p):return 'n/a'
    if p<.001:return 'q < 0.001'
    return f'q = {p:.3f}'

def mouse_means(df, value, by=('day','age','animal')):
    return (df.groupby(list(by),observed=True,as_index=False)[value]
            .mean().dropna(subset=[value]))

# %% 1. LOAD, VALIDATE, DOCUMENT EXCLUSIONS
trial=read('trial_summary.csv', exclude_video=False)  # Preserve independently valid avoidance outcomes
bins=read('trial_probability_bins.csv')
events=read('behavior_events.csv')
visits=read('reward_visits.csv')
if trial.empty:raise FileNotFoundError('Run PMA Stage 2 batch first: missing PMA_BATCH/trial_summary.csv')
ensure(trial,['animal','age','day','trial','cue_on_s','cue_off_s'],'trial_summary')
trial['trial']=pd.to_numeric(trial['trial'],errors='coerce')
for col in trial.columns:
    if col.endswith('_pct') or col.endswith('_count') or col.endswith('_duration_s') or col in ('shock_received','exit_latency_s','cue_on_s','cue_off_s'):
        trial[col]=pd.to_numeric(trial[col],errors='coerce')
# Coverage metadata may be entirely blank in older batch outputs. It is NOT
# evidence that DLC tracking is missing. Derive per-metric validity from observed
# tracking durations and finite measurements, and remove ONLY the appropriate
# video-derived values from the visibly distorted 3f_a_tp3 session.
trial['video_session_eligible'] = ~trial['session'].isin(EXCLUDE_VIDEO)
platform_secs = pd.to_numeric(trial.get('cue_platform_valid_duration_s', pd.Series(np.nan,index=trial.index)),errors='coerce')
zone_secs = pd.to_numeric(trial.get('cue_zone_valid_duration_s', pd.Series(np.nan,index=trial.index)),errors='coerce')
reward_secs = pd.to_numeric(trial.get('cue_reward_valid_duration_s', pd.Series(np.nan,index=trial.index)),errors='coerce')
trial['valid_cue_platform'] = trial['video_session_eligible'] & platform_secs.ge(MIN_VALID_CUE_S)
trial['valid_cue_zone'] = trial['video_session_eligible'] & zone_secs.ge(MIN_VALID_CUE_S)
trial['valid_cue_reward_head'] = trial['video_session_eligible'] & reward_secs.ge(MIN_VALID_CUE_S)
# This legacy name is retained for code that displays diagnostic summaries,
# but NEVER used as a universal gate for all video metrics.
trial['valid_cue_video'] = trial['valid_cue_zone']

# Mask ONLY the variables whose measurement requires the corresponding tracking.
# Do not mask Arduino poke_count_cue or reward_count_cue.
platform_cols = ['cue_platform_occupancy_pct','head_dip_cue_count',
                 'platform_entry_cue_count','platform_exit_cue_count']
zone_cols = ['cue_reward_occupancy_pct','cue_grid_occupancy_pct',
             'reward_entry_cue_count','reward_exit_cue_count',
             'cue_reward_visit_count','cue_reward_visit_total_duration_s',
             'cue_reward_visit_mean_duration_s','cue_reward_visit_rate_per_min']
reward_head_cols = ['cue_reward_head_occupancy_pct']
for col in platform_cols:
    if col in trial: trial.loc[~trial['valid_cue_platform'],col] = np.nan
for col in zone_cols:
    if col in trial: trial.loc[~trial['valid_cue_zone'],col] = np.nan
for col in reward_head_cols:
    if col in trial: trial.loc[~trial['valid_cue_reward_head'],col] = np.nan
# ITI outcomes have their own valid-duration columns; these are NOT 30-second cues.
MIN_VALID_ITI_S = 15.0
iti_platform_secs = pd.to_numeric(trial.get('iti_platform_valid_duration_s',pd.Series(np.nan,index=trial.index)),errors='coerce')
iti_zone_secs = pd.to_numeric(trial.get('iti_zone_valid_duration_s',pd.Series(np.nan,index=trial.index)),errors='coerce')
trial['valid_iti_platform'] = trial['video_session_eligible'] & iti_platform_secs.ge(MIN_VALID_ITI_S)
trial['valid_iti_zone'] = trial['video_session_eligible'] & iti_zone_secs.ge(MIN_VALID_ITI_S)
for col in ['iti_platform_occupancy_pct','platform_entry_iti_count','platform_exit_iti_count','head_dip_iti_count']:
    if col in trial: trial.loc[~trial['valid_iti_platform'],col] = np.nan
for col in ['iti_reward_occupancy_pct','iti_grid_occupancy_pct','reward_entry_iti_count','reward_exit_iti_count',
            'iti_reward_visit_count','iti_reward_visit_total_duration_s','iti_reward_visit_mean_duration_s']:
    if col in trial: trial.loc[~trial['valid_iti_zone'],col] = np.nan

# Avoidance must NOT be gated on continuous DLC coverage throughout the cue.
# Stage 2 shock_received can be 0 when there was no SHOCK_ON, or 0/1
# when a shock event was classified using the platform state at that instant.
# If tracking is unknown exactly at shock time, retain NaN; never fabricate success.
# The >=25 s valid tracking rule applies ONLY to time-occupancy metrics.
if 'shock_received' in trial.columns:
    trial['avoidance_success']=np.where(trial['shock_received'].isin([0,1]),1-trial['shock_received'],np.nan)
else:
    trial['avoidance_success']=np.nan
if 'exit_status' in trial.columns:
    trial['exit_observed_bool']=trial['exit_status'].astype(str).eq('observed_exit')
else:
    trial['exit_observed_bool']=trial.get('exit_observed',pd.Series(False,index=trial.index)).astype(str).str.lower().eq('true')

qc = trial.groupby(['day','age'],observed=True).agg(n_mice=('animal','nunique'),n_trials=('trial','size'),
                video_valid=('valid_cue_zone','sum'), platform_valid=('valid_cue_platform','sum'),
                reward_zone_valid=('valid_cue_zone','sum')).reset_index()
qc.to_csv(OUT/'sample_sizes_by_day.csv',index=False)
(OUT/'analysis_settings.txt').write_text(
    'Mouse is statistical unit. Per-day adult vs adolescent two-sided Mann-Whitney; '
    'BH-FDR across six days per metric. Exploratory, not a longitudinal mixed model.\n'
    f'Excluded video sessions: {sorted(EXCLUDE_VIDEO)}\n'
    f'Assumed-tail sessions: {sorted(ASSUMED_TAIL)}\n'
    f'Included assumed-tail: {INCLUDE_ASSUMED_TAIL}\n'
    f'Minimum cue tracking used for occupancy: {MIN_VALID_CUE_S:.1f} s / 30 s; independently for platform vs zone.\n'
    'Blank cue_video_covered metadata are ignored; actual tracking duration is used.\n'
    'Video exclusion is applied only to video-derived metrics, not independently valid avoidance/pokes/rewards.\n'
    'Avoidance from Stage2 shock_received, not filtered by whole-cue valid DLC duration. Unknown shock outcomes remain NaN.\n'
    'Censored exit latencies are not treated as observed exits.\n')
print('\nSample sizes (mouse IDs represent biological replicates):\n',qc.to_string(index=False))
print('Figures and statistics:',OUT)
coverage_audit = trial.groupby(['day','age'],observed=True).agg(
    n_trials=('trial','size'), n_platform_valid=('valid_cue_platform','sum'),
    n_zone_valid=('valid_cue_zone','sum'), n_reward_head_valid=('valid_cue_reward_head','sum'),
    n_avoidance_known=('avoidance_success','count')).reset_index()
coverage_audit.to_csv(OUT/'coverage_audit_by_day_age.csv',index=False)
print('\nValid data by day/age (coverage-based, not metadata-based):\n', coverage_audit.to_string(index=False))

# %% 1B. DATA AUDIT — CHECK ACTUAL VALID MICE BEFORE DRAWING FIGURES
# A visually empty figure usually means missing input values, not a plotting bug.
# This table makes the reason inspectable, without inventing avoidance values.
if 'shock_received' in trial.columns:
    audit=trial.assign(shock_is_01=trial['shock_received'].isin([0,1]),
                       final_avoidance_valid=trial['avoidance_success'].notna(),
                       rejected_by_old_rule=trial['shock_received'].isin([0,1]) & ~trial['valid_cue_video'])
    audit=(audit.groupby(['day','age','animal','session'],dropna=False,observed=True)
           .agg(n_trials=('trial','size'),valid_shock_01=('shock_is_01','sum'),
                valid_cue_video=('valid_cue_zone','sum'),
                valid_avoidance=('final_avoidance_valid','sum'),
                previously_rejected_by_video_filter=('rejected_by_old_rule','sum'))
           .reset_index())
    audit.to_csv(OUT/'Q1_avoidance_data_audit.csv',index=False)
    print('\nQ1 mouse counts with valid avoidance outcomes by day / age:')
    aud_mice=(audit.assign(has_data=audit.valid_avoidance.gt(0))
               .groupby(['day','age'],observed=True)['has_data'].sum()
               .unstack(fill_value=0).reindex(DAYS,fill_value=0))
    print(aud_mice.to_string())
    print('Trial-level avoidance outcomes:', int(trial.avoidance_success.notna().sum()), '/', len(trial))
    if (aud_mice.values<2).any():
        print('WARNING: Some age/day groups have fewer than 2 mice with avoidance data.')
        print('Inspect Q1_avoidance_data_audit.csv. If still sparse, inspect Stage2 shock_received generation and event logs.')

# %% HELPER: AESTHETIC BOX PLOTS + ADULT/ADOLESCENT STATISTICS
all_tests=[]
def box_by_day(df, value, ylabel, filename, title=None, ylim=None):
    if value not in df or df[value].notna().sum()==0:
        print('Skipping missing metric:',value);return
    mouse=mouse_means(df,value)
    if mouse.empty:return
    # Plot EACH mouse, not each trial. Always show n and missing combinations.
    fig,ax=plt.subplots(figsize=(max(10,1.65*len(DAYS)),5.7))
    records=[]
    daily_values={}
    for day_index,day in enumerate(DAYS):
        vals={group:mouse.loc[(mouse.day==day)&(mouse.age==group),value].dropna().to_numpy(dtype=float)
              for group in GROUPS}
        daily_values[day]=vals
        a,b=vals['adult'],vals['adolescent']
        p=float(stats.mannwhitneyu(a,b,alternative='two-sided').pvalue) if len(a)>=MIN_MICE_FOR_TEST and len(b)>=MIN_MICE_FOR_TEST else np.nan
        records.append({'metric':value,'day':day,'n_adult':len(a),'n_adolescent':len(b),
                        'adult_mean':np.mean(a) if len(a) else np.nan,
                        'adolescent_mean':np.mean(b) if len(b) else np.nan,'p_raw':p})
        for group,offset in [('adult',-.23),('adolescent',+.23)]:
            arr=vals[group];x=day_index+offset
            if not len(arr):continue
            if len(arr)>=2:
                artist=ax.boxplot(arr,positions=[x],widths=[BOX_WIDTH*.72],patch_artist=True,
                                  showfliers=False,whis=(0,100),zorder=3,
                                  medianprops={'color':'#21252B','linewidth':2.2},
                                  boxprops={'edgecolor':COLORS[group],'linewidth':1.55},
                                  whiskerprops={'color':COLORS[group],'linewidth':1.45},
                                  capprops={'color':COLORS[group],'linewidth':1.45})
                artist['boxes'][0].set_facecolor(mpl.colors.to_rgba(COLORS[group],.25))
            # Single observations are drawn as LARGE points; no fake box.
            ax.scatter(np.full(len(arr),x)+rng.uniform(-JITTER,JITTER,len(arr)),arr,
                       s=max(40,POINT_SIZE*1.7),facecolors=COLORS[group],
                       edgecolors='white',linewidths=.8,alpha=.90,zorder=5)
    result=pd.DataFrame(records)
    result['q_fdr']=bh_fdr(result['p_raw'].values)
    result.to_csv(OUT/(filename+'_statistics.csv'),index=False)
    all_values=mouse[value].dropna().to_numpy(float)
    ymin,ymax=float(np.min(all_values)),float(np.max(all_values))
    fraction_metric=value in ('avoidance_success','platform_binary','poke_binary')
    if fraction_metric:
        ax.set_ylim(-.07,1.24)
        ax.set_yticks(np.linspace(0,1,6))
        data_ymin,data_ymax=0.,1.
    else:
        span=max(ymax-ymin,abs(ymax)*.18,1.)
        ax.set_ylim(min(0,ymin-.13*span),ymax+.39*span)
        data_ymin,data_ymax=ymin,ymax
    yrange=max(data_ymax-data_ymin,1.)
    for i,r in result.iterrows():
        v=daily_values[r.day]
        a,b=v['adult'],v['adolescent']
        if len(a)>=MIN_MICE_FOR_TEST and len(b)>=MIN_MICE_FOR_TEST and np.isfinite(r.q_fdr):
            top=max(np.max(a),np.max(b))+.07*yrange
            ax.plot([i-.23,i-.23,i+.23,i+.23],[top-.015*yrange,top,top,top-.015*yrange],
                    color='#555555',lw=1.05,clip_on=False)
            ax.text(i,top+.012*yrange,p_label(r.q_fdr),ha='center',va='bottom',fontsize=8)
        if SHOW_SAMPLE_SIZE_LABELS:
            ax.text(i, .925, f"n={len(a)} / {len(b)}",transform=ax.get_xaxis_transform(),
                    ha='center',va='top',fontsize=8,color='#666666',
                    bbox={'facecolor':'white','edgecolor':'none','alpha':.7,'pad':1.5})
    if ylim is not None:ax.set_ylim(ylim)
    ax.set_xticks(np.arange(len(DAYS)),DAYS)
    ax.set_xlim(-.62,len(DAYS)-.38)
    clean_axis(ax,ylabel,title or ylabel)
    ax.grid(False) 
    ax.scatter([],[],color=COLORS['adult'],s=45,label='Adults')
    ax.scatter([],[],color=COLORS['adolescent'],s=45,label='Adolescents')
    ax.legend(frameon=False,loc='upper right',ncol=2,bbox_to_anchor=(1,1.01))
    fig.tight_layout();export(fig,filename)
    if np.sum(result[['n_adult','n_adolescent']].to_numpy())<12:
        warnings.warn(f'{filename}: unusually sparse mouse-level data. Inspect Q1_avoidance_data_audit.csv and source trial_summary.csv!')
    all_tests.append(result)
    return result

# %% QUESTION 1 — DO ADULTS AND ADOLESCENTS LEARN AT DIFFERENT RATES?
# Daily performance: pooled trials within each mouse first; the plot shows mouse averages.
box_by_day(trial,'avoidance_success','Avoidance success (fraction of trials)',
           'Q1_avoidance_success','Q1 | Shock avoidance across training')
for col,lab,slug in [
    ('cue_platform_occupancy_pct','Time on platform during cue (%)','cue_platform_time'),
    ('cue_reward_occupancy_pct','Time in reward zone during cue (%)','cue_reward_time'),
    ('cue_grid_occupancy_pct','Time on grid during cue (%)','cue_grid_time'),
    ('poke_count_cue','Nose pokes per cue','poke_count'),
    ('reward_count_cue','Rewards per cue','reward_count')]:
    box_by_day(trial,col,lab,'Q1_'+slug)

# %% QUESTION 2 — WHAT HAPPENS BEFORE, DURING, AND AFTER CUE ONSET?
# Bin midpoint occupancy = binary at a timepoint. Nose-poke probability = any poke in bin.
# FIRST calculate per-mouse mean across trials; THEN mean ± mouse-level SEM.
if not bins.empty:
    ensure(bins,['animal','age','day','trial','bin_midpoint_s','platform_binary','poke_binary'],
           'trial_probability_bins')
    for key in ('platform_binary','poke_binary'):
        bins[key]=pd.to_numeric(bins[key],errors='coerce')
    bins['bin_midpoint_s']=pd.to_numeric(bins['bin_midpoint_s'],errors='coerce')
    # Hide video-only bins if cue video for a trial is not available.
    # Probability bins are individually observed frames. Do not invalidate all
    # 2-second bins merely because a whole 30-second cue is <15 s valid.
    # EXCLUDE_VIDEO was already applied when bins were read.
    bins=bins.drop(columns=['valid_cue_video'], errors='ignore')
    mouse_bins=bins.groupby(['day','age','animal','bin_midpoint_s'],as_index=False,observed=True)[
        ['platform_binary','poke_binary']].mean()
    def probability_trace(day,metric,ax,group):
        subset=mouse_bins[(mouse_bins.day==day)&(mouse_bins.age==group)]
        g=subset.groupby('bin_midpoint_s')[metric].agg(['mean','count',sem]).reset_index()
        if g.empty:return
        t=g.bin_midpoint_s.to_numpy(float);mean=g['mean'].to_numpy(float)
        error=g['sem'].fillna(0).to_numpy(float)
        ax.plot(t,mean,lw=LINE_WIDTH,color=COLORS[group],label=f'{group.title()} (n={subset.animal.nunique()})',zorder=5)
        ax.fill_between(t,np.clip(mean-error,0,1),np.clip(mean+error,0,1),
                        color=COLORS[group],alpha=SEM_ALPHA,lw=0,zorder=3)
    for metric,name in [('platform_binary','Platform occupancy probability'),
                        ('poke_binary','Nose-poke probability')]:
        fig,axes=plt.subplots(2,3,figsize=(13.5,7.5),sharex=True,sharey=True)
        for ax,day in zip(axes.flat,DAYS):
            ax.axvspan(0,30,color='#69717B',alpha=.07,zorder=0)
            ax.axvline(0,color='#777777',ls='--',lw=1)
            ax.axvline(30,color='#777777',ls=':',lw=1)
            for group in GROUPS:probability_trace(day,metric,ax,group)
            ax.set_title(day,loc='left',fontweight='bold');ax.set_ylim(-.02,1.03)
            clean_axis(ax)
            ax.grid(False) 
        for ax in axes[-1,:]:ax.set_xlabel('Time relative to cue onset (s)')
        for ax in axes[:,0]:ax.set_ylabel('Trial probability')
        handles,labels=axes[0,0].get_legend_handles_labels()
        if handles:fig.legend(handles,labels,frameon=False,loc='upper center',ncol=2,bbox_to_anchor=(.5,1.015))
        fig.suptitle('Q2 | '+name,fontsize=15,y=1.07)
        fig.tight_layout();export(fig,'Q2_'+metric+'_traces')
    # Export clean mouse-level traces for advanced statistics elsewhere.
    mouse_bins.to_csv(OUT/'Q2_mouse_level_probability_traces.csv',index=False)

# %% QUESTION 3 — DO PLATFORM STRATEGIES DIFFER BETWEEN AGES?
for col,label,slug in [
    ('head_dip_cue_count','Head dips during cue (count/trial)','head_dips'),
    ('platform_entry_cue_count','Platform entries during cue (count/trial)','entries'),
    ('platform_exit_cue_count','Platform exits during cue (count/trial)','exits'),
    ('iti_platform_occupancy_pct','Platform occupancy during ITI (%)','ITI_platform'),
    ('iti_reward_occupancy_pct','Reward-zone occupancy during ITI (%)','ITI_reward'),
    ('reward_entry_cue_count','Reward-zone entries during cue (count/trial)','reward_entries')]:
    box_by_day(trial,col,label,'Q3_'+slug)

# %% QUESTION 4 — LATENCIES AND THEIR DISTRIBUTIONS
# Exit latency is defined from CUE OFFSET; an animal already off platform at offset
# is not a genuine observed platform exit. Censored trials MUST NOT be treated as exits.
if 'exit_latency_s' in trial:
    valid_exit=trial.loc[trial['video_session_eligible']].copy()
    valid_exit['exit_latency_s']=pd.to_numeric(valid_exit['exit_latency_s'],errors='coerce')
    valid_exit.loc[~valid_exit['exit_observed_bool'],'exit_latency_s']=np.nan
    box_by_day(valid_exit,'exit_latency_s','Observed post-cue exit latency (s)',
               'Q4_postcue_observed_exit_latency','Q4 | Post-cue platform exit latency')
    fig,axes=plt.subplots(2,3,figsize=(13.5,7.5),sharex=True,sharey=True)
    bins_for_hist=np.arange(0,125,10)
    for day,ax in zip(DAYS,axes.flat):
        for group in GROUPS:
            part=valid_exit[(valid_exit.day==day)&(valid_exit.age==group)]
            # Distribution weights mice equally rather than trials equally:
            # each mouse contributes 1 / number of its observed exit latencies.
            part=part.dropna(subset=['exit_latency_s'])
            if part.empty:continue
            weights=1/part.groupby('animal')['exit_latency_s'].transform('count').to_numpy(float)
            weights/=max(1,part.animal.nunique())
            ax.hist(part.exit_latency_s,bins=bins_for_hist,weights=weights,histtype='step',
                    color=COLORS[group],linewidth=2.3,label=group.title())
        ax.set_title(day,loc='left',fontweight='bold');clean_axis(ax)
    for ax in axes[-1,:]:ax.set_xlabel('Observed exit latency after cue offset (s)')
    for ax in axes[:,0]:ax.set_ylabel('Mouse-balanced fraction')
    handles,labels=axes[0,0].get_legend_handles_labels()
    if handles:fig.legend(handles,labels,loc='upper center',ncol=2,frameon=False,bbox_to_anchor=(.5,1.015))
    fig.suptitle('Q4 | Distribution of observed exit latencies (censored excluded)',fontsize=14,y=1.075)
    fig.tight_layout();export(fig,'Q4_exit_latency_distributions')
    # Censoring report makes clear why these distributions are incomplete.
    censor=trial.groupby(['day','age'],observed=True).agg(
        n_trials=('trial','size'),n_observed_exits=('exit_observed_bool','sum')).reset_index()
    censor.to_csv(OUT/'Q4_exit_latency_censoring.csv',index=False)

# Entry latency requires recorded platform_entry behavioral events and cue onset.
# When already on-platform at cue onset, entry latency is NOT zero: it is not a new entry.
if not events.empty and 'event' in events.columns and 'time_s' in events.columns:
    event_times=events.loc[events['event'].astype(str).str.lower().eq('platform_entry')].copy()
    event_times['time_s']=pd.to_numeric(event_times.time_s,errors='coerce')
    lookup={sess:np.sort(g.time_s.dropna().to_numpy()) for sess,g in event_times.groupby('session')}
    entry_rows=[]
    for r in trial.itertuples(index=False):
        if not r.valid_cue_platform:continue
        candidate=lookup.get(r.session,np.array([],dtype=float))
        first=candidate[(candidate>=r.cue_on_s)&(candidate<r.cue_off_s)]
        entry_rows.append({'session':r.session,'animal':r.animal,'age':r.age,'day':r.day,
                           'trial':r.trial,'entry_latency_s':float(first[0]-r.cue_on_s) if len(first) else np.nan,
                           'entry_observed':bool(len(first))})
    entry=pd.DataFrame(entry_rows)
    if not entry.empty:
        entry.to_csv(OUT/'Q4_first_platform_entry_latency_trials.csv',index=False)
        box_by_day(entry,'entry_latency_s','First recorded platform entry after cue onset (s)',
                   'Q4_entry_latency','Q4 | First platform entry latency (observed only)')
        fig,axes=plt.subplots(2,3,figsize=(13.5,7.5),sharex=True,sharey=True)
        for day,ax in zip(DAYS,axes.flat):
            for group in GROUPS:
                subset=entry[(entry.day==day)&(entry.age==group)].dropna(subset=['entry_latency_s'])
                if subset.empty:continue
                weights=1/subset.groupby('animal')['entry_latency_s'].transform('count').to_numpy(float)
                weights/=max(1,subset.animal.nunique())
                ax.hist(subset.entry_latency_s,bins=np.arange(0,33,3),weights=weights,
                        histtype='step',color=COLORS[group],lw=2.3,label=group.title())
            ax.set_title(day,loc='left',fontweight='bold');clean_axis(ax)
        for ax in axes[-1,:]:ax.set_xlabel('First entry latency (s)')
        for ax in axes[:,0]:ax.set_ylabel('Mouse-balanced fraction')
        handles,labels=axes[0,0].get_legend_handles_labels()
        if handles:fig.legend(handles,labels,frameon=False,ncol=2,loc='upper center',bbox_to_anchor=(.5,1.015))
        fig.suptitle('Q4 | Platform entry latency (observed entries only)',fontsize=15,y=1.07)
        fig.tight_layout();export(fig,'Q4_entry_latency_distributions')
        entry.groupby(['day','age'],as_index=False).agg(n_trials=('trial','size'),
            n_observed_entries=('entry_observed','sum')).to_csv(OUT/'Q4_entry_latency_observation_counts.csv',index=False)
else:print('No timestamped platform-entry events; entry latency plots skipped.')

# %% QUESTION 5 — HOW DOES PERFORMANCE CHANGE ACROSS THE TEN TRIALS?
# Separate mouse-level means for each trial. Adult/adolescent bold line ± SEM.
for metric,label,filename in [('avoidance_success','Avoidance success','Q5_success_by_trial'),
                              ('cue_platform_occupancy_pct','Platform occupancy (%)','Q5_platform_by_trial'),
                              ('poke_count_cue','Pokes per cue','Q5_pokes_by_trial')]:
    if metric not in trial:continue
    tmp=trial.groupby(['day','age','animal','trial'],as_index=False,observed=True)[metric].mean()
    fig,axes=plt.subplots(2,3,figsize=(13.5,7.5),sharex=True,sharey=True)
    ax.grid(False)
    for day,ax in zip(DAYS,axes.flat):
        for group in GROUPS:
            sub=tmp[(tmp.day==day)&(tmp.age==group)]
            g=sub.groupby('trial')[metric].agg(['mean',sem]).reset_index()
            if g.empty:continue
            x=g.trial.to_numpy(float);y=g['mean'].to_numpy(float)
            e=g['sem'].fillna(0).to_numpy(float)
            ax.plot(x,y,'-o',lw=LINE_WIDTH,ms=3.8,color=COLORS[group],label=group.title())
            ax.fill_between(x,y-e,y+e,color=COLORS[group],alpha=SEM_ALPHA,lw=0)
        ax.set_title(day,loc='left',fontweight='bold');ax.set_xticks(range(1,11,2));clean_axis(ax)
        if metric=='avoidance_success':ax.set_ylim(-.05,1.05)
    for ax in axes[-1,:]:ax.set_xlabel('Trial within day')
    for ax in axes[:,0]:ax.set_ylabel(label)
    handles,labels=axes[0,0].get_legend_handles_labels()
    if handles:fig.legend(handles,labels,frameon=False,ncol=2,loc='upper center',bbox_to_anchor=(.5,1.015))
    fig.suptitle('Q5 | '+label+' across individual trials',fontsize=15,y=1.07)
    fig.tight_layout();export(fig,filename)
    ax.grid(False)
# Early vs late trial differences, mouse-level per day.
for metric,label,filename in [('avoidance_success','Change in avoidance (late - early)','Q5_early_late_avoidance'),
                              ('cue_platform_occupancy_pct','Change in platform occupancy (%)','Q5_early_late_platform')]:
    temp=trial[trial.trial.isin([1,2,3,8,9,10])].copy()
    temp['phase']=np.where(temp.trial<=3,'early','late')
    a=temp.groupby(['day','age','animal','phase'],observed=True)[metric].mean().unstack('phase')
    if not {'early','late'}.issubset(a.columns):continue
    dif=(a['late']-a['early']).dropna().rename('late_minus_early').reset_index()
    box_by_day(dif,'late_minus_early',label,filename)

# %% QUESTION 6 — PLATFORM OCCUPANCY HEATMAPS (TRIALS × TIME)
# Organized by day, mouse, then trial. Only recorded video states enter the plot.
if not bins.empty:
    for day in DAYS:
        for age in GROUPS:
            b=bins[(bins.day==day)&(bins.age==age)].copy()
            if b.empty:continue
            b['row_label']=b.animal.astype(str)+'  T'+b.trial.astype('Int64').astype(str)
            b=b.sort_values(['animal','trial','bin_midpoint_s'])
            mat=b.pivot_table(index=['animal','trial'],columns='bin_midpoint_s',
                              values='platform_binary',aggfunc='mean',dropna=False)
            if mat.empty:continue
            mat=mat.sort_index();t=mat.columns.to_numpy(float)
            fig,ax=plt.subplots(figsize=(8.1,max(3.7,min(15.,.22*len(mat)+1.5))))
            cmap=mpl.colors.ListedColormap(['#F2E9DF',COLORS['platform']]);cmap.set_bad('#DBDBDB')
            ax.imshow(mat.to_numpy(dtype=float),aspect='auto',interpolation='nearest',
                      cmap=cmap,vmin=0,vmax=1,extent=[t.min()-1,t.max()+1,len(mat)-.5,-.5])
            ax.axvline(0,color='#222222',ls='--',lw=1.3)
            ax.axvline(30,color='#222222',ls=':',lw=1.3)
            ax.set_xlabel('Time relative to cue onset (s)');ax.set_ylabel('Mouse / trial')
            if len(mat)<=45:ax.set_yticks(np.arange(len(mat)),[f'{animal} / {int(tr)}' for animal,tr in mat.index],fontsize=7)
            else:ax.set_yticks([])
            ax.set_title(f'Q6 | Platform state · {day} · {age.title()}',loc='left')
            fig.tight_layout();export(fig,f'Q6_heatmap_{day}_{age}')
            
            

# %% QUESTION 6 — COMBINED ZONE HEATMAPS (TRIALS × TIME)

from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch

# ============================================================
# AESTHETICS — CHANGE COLORS HERE
# ============================================================

ZONE_COLORS = {

    'missing': '#FFFFFF', 'platform':'#FF7E66', 'poke':'#007e5d', 'reward':'#6C9E7F',
    'grid':'#FFFFFF'
}

HEATMAP_WIDTH = 9
HEATMAP_ROW_HEIGHT = 0.22
HEATMAP_MAX_HEIGHT = 16

# 0 = Grid, 1 = Platform, 2 = Reward
cmap = ListedColormap([
    ZONE_COLORS['grid'],
    ZONE_COLORS['platform'],
    ZONE_COLORS['reward']
])
cmap.set_bad(ZONE_COLORS['missing'])
norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], cmap.N)

if not bins.empty:

    required = ['platform_binary', 'reward_zone_binary']

    if not all(col in bins.columns for col in required):
        print("Available bin columns:", bins.columns.tolist())
        raise ValueError(
            "Expected platform_binary and reward_binary in bins."
        )

    for day in DAYS:
        for age in GROUPS:

            b = bins[
                (bins.day == day) &
                (bins.age == age)
            ].copy()

            if b.empty:
                continue

            # Convert both zone indicators to numeric
            b['platform_binary'] = pd.to_numeric(
                b['platform_binary'], errors='coerce'
            )
            b['reward_binary'] = pd.to_numeric(
                b['reward_zone_binary'], errors='coerce'
            )

            # Preserve unknown tracking as NaN
            b['zone_state'] = np.nan

            valid = (
                b['platform_binary'].notna()
                & b['reward_zone_binary'].notna()
            )

            b.loc[valid, 'zone_state'] = 0  # Grid
            b.loc[
                valid & (b['platform_binary'] >= 0.5),
                'zone_state'
            ] = 1  # Platform

            b.loc[
                valid & (b['reward_zone_binary'] >= 0.5)
                & (b['platform_binary'] < 0.5),
                'zone_state'
            ] = 2  # Reward

            # Platform takes priority if both are marked occupied.
            # Check overlapping flags rather than silently reclassifying.
            overlaps = (
                valid
                & (b['platform_binary'] >= 0.5)
                & (b['reward_zone_binary'] >= 0.5)
            ).sum()

            if overlaps:
                print(
                    f'{day} {age}: {overlaps} bins marked '
                    'as both platform and reward; '
                    'displayed as platform.'
                )

            b = b.sort_values([
                'animal', 'trial', 'bin_midpoint_s'
            ])

            mat = b.pivot_table(
                index=['animal', 'trial'],
                columns='bin_midpoint_s',
                values='zone_state',
                aggfunc='first',
                dropna=False
            )

            if mat.empty:
                continue

            mat = mat.sort_index()
            t = mat.columns.to_numpy(float)

            fig_height = max(
                3.7,
                min(
                    HEATMAP_MAX_HEIGHT,
                    HEATMAP_ROW_HEIGHT * len(mat) + 1.8
                )
            )

            fig, ax = plt.subplots(
                figsize=(HEATMAP_WIDTH, fig_height)
            )

            # Each trial is a horizontal strip of zone states
            ax.imshow(
                mat.to_numpy(dtype=float),
                aspect='auto',
                interpolation='nearest',
                cmap=cmap,
                norm=norm,
                extent=[
                    t.min() - 1,
                    t.max() + 1,
                    len(mat) - 0.5,
                    -0.5
                ]
            )
                        # ============================================================
            # HORIZONTAL DIVIDERS BETWEEN MICE
            # ============================================================
            
            DIVIDER_COLOR = 'black'
            DIVIDER_WIDTH = 2.5
            
            # Detect where a new mouse begins
            mouse_ids = mat.index.get_level_values('animal')
            
            for i in range(1, len(mouse_ids)):
                if mouse_ids[i] != mouse_ids[i - 1]:
                    ax.axhline(
                        y=i - 0.5,
                        color=DIVIDER_COLOR,
                        linewidth=DIVIDER_WIDTH,
                        zorder=10
                    )

            # Cue onset and offset
            ax.axvline(0, color='black', ls='--', lw=1.2)
            ax.axvline(30, color='black', ls=':', lw=1.2)

            ax.set_xlabel('Time from cue onset (s)')
            ax.set_ylabel('Mouse / trial')

            if len(mat) <= 60:
                ax.set_yticks(
                    np.arange(len(mat)),
                    [
                        f'{animal} / T{int(tr)}'
                        for animal, tr in mat.index
                    ],
                    fontsize=7
                )
            else:
                ax.set_yticks([])

            ax.set_title(
                f'Q6 | Behavioral zones · {day} · {age.title()}',
                loc='left'
            )

            # Custom categorical legend
            legend_handles = [
                Patch(facecolor=ZONE_COLORS['grid'], label='Grid'),
                Patch(facecolor=ZONE_COLORS['platform'], label='Platform'),
                Patch(facecolor=ZONE_COLORS['reward'], label='Reward zone'),
                Patch(
                    facecolor=ZONE_COLORS['missing'],
                    edgecolor='black',
                    label='Missing'
                )
            ]

            ax.legend(
                handles=legend_handles,
                frameon=False,
                loc='upper center',
                bbox_to_anchor=(0.5, -0.08),
                ncol=4
            )

            ax.grid(False)

            for spine in ax.spines.values():
                spine.set_visible(False)

            fig.tight_layout()

            export(
                fig,
                f'Q6_combined_zones_{day}_{age}'
            )


# %% 7. FINAL STATISTICS TABLE + NOTES
if all_tests:
    tests=pd.concat(all_tests,ignore_index=True)
    tests.to_csv(OUT/'ALL_adult_vs_adolescent_daywise_stats.csv',index=False)
    print('\nSaved group comparison statistics:',OUT/'ALL_adult_vs_adolescent_daywise_stats.csv')
print('\nDONE. 300-dpi figures:',OUT)
print('Important: mouse-level per-day tests do not replace longitudinal mixed-effects models.')
print('Exit and entry latency distributions include observed events only; see censoring/count CSVs.')
print('Video-derived 3f_a_tp3 data were excluded. Tail-assumed sessions are flagged in analysis_settings.txt.')


# %% QUESTION 7 — TP2 VS LP1 SHOCK CORRELATION (FIXED)

from scipy import stats

# ============================================================
# AESTHETICS / SETTINGS
# ============================================================

CORR_POINT_SIZE = 100
CORR_LINE_WIDTH = 2.5
CORR_JITTER = 0.12

MIN_VALID_TRIALS = 8
EXPECTED_TRIALS = 10

# ============================================================
# PREPARE DATA
# ============================================================

df = trial.copy()

df['shock_received'] = pd.to_numeric(
    df['shock_received'], errors='coerce'
)

df = df[df['day'].isin(['TP1', 'LP1'])].copy()

# One observation per mouse per training day
shocks = (
    df.groupby(['animal', 'age', 'day'], as_index=False)
    .agg(
        shocks_observed=('shock_received', 'sum'),
        valid_trials=('shock_received', 'count'),
        total_trials=('trial', 'nunique')
    )
)

# Only retain animals with sufficient valid trials
shocks.loc[
    shocks['valid_trials'] < MIN_VALID_TRIALS,
    'shocks_observed'
] = np.nan

# Normalize shock rate to 10 trials
shocks['shocks_per_10_trials'] = (
    shocks['shocks_observed']
    / shocks['valid_trials']
    * EXPECTED_TRIALS
)

wide = shocks.pivot(
    index=['animal', 'age'],
    columns='day',
    values='shocks_per_10_trials'
).reset_index()

wide = wide.dropna(subset=['TP1', 'LP1'])

print('\nMice included in TP2 vs LP1 correlation:')
print(wide.groupby('age').size())

# ============================================================
# CORRELATION PLOT
# ============================================================

fig, ax = plt.subplots(figsize=(6.7, 6.2))

results = []

for group in ['adult', 'adolescent']:

    sub = wide[wide['age'] == group]

    x = sub['TP1'].to_numpy(dtype=float)
    y = sub['LP1'].to_numpy(dtype=float)

    n = len(sub)

    if n == 0:
        continue

    color = COLORS[group]

    # Visual jitter only
    xj = x + rng.uniform(-CORR_JITTER, CORR_JITTER, n)
    yj = y + rng.uniform(-CORR_JITTER, CORR_JITTER, n)

    ax.scatter(
        xj, yj,
        s=CORR_POINT_SIZE,
        facecolor=color,
        edgecolor='white',
        linewidth=1.2,
        alpha=0.9,
        zorder=5,
        label=f'{group.title()} (n={n})'
    )

    rho, p = np.nan, np.nan

    if (
        n >= 3
        and np.unique(x).size > 1
        and np.unique(y).size > 1
    ):
        rho, p = stats.spearmanr(x, y)

        slope, intercept, *_ = stats.linregress(x, y)

        xx = np.linspace(x.min(), x.max(), 100)

        ax.plot(
            xx,
            intercept + slope * xx,
            color=color,
            linewidth=CORR_LINE_WIDTH,
            alpha=0.85,
            zorder=3
        )

    results.append({
        'age': group,
        'n_mice': n,
        'spearman_rho': rho,
        'p_value': p
    })

# ============================================================
# FIGURE AESTHETICS
# ============================================================

ax.set_xlabel('TP2 shocks per 10 valid trials', fontsize=12)
ax.set_ylabel('LP1 shocks per 10 valid trials', fontsize=12)

ax.set_title(
    'TP2 performance versus LP1 performance',
    fontsize=13,
    loc='left'
)

ax.set_xlim(-0.5, 10.5)
ax.set_ylim(-0.5, 10.5)

ax.set_xticks(np.arange(0, 11, 2))
ax.set_yticks(np.arange(0, 11, 2))

ax.grid(False)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

ax.legend(frameon=False)

fig.tight_layout()

export(fig, 'Q7_TP2_vs_LP1_shock_correlation')

# ============================================================
# SAVE DATA AND STATISTICS
# ============================================================

results_df = pd.DataFrame(results)

results_df.to_csv(
    OUT / 'Q7_shock_correlation_statistics.csv',
    index=False
)

wide.to_csv(
    OUT / 'Q7_shock_correlation_mouse_data.csv',
    index=False
)

shocks.to_csv(
    OUT / 'Q7_shock_trial_coverage.csv',
    index=False
)

print('\nCorrelations:')
print(results_df.to_string(index=False))