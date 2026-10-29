#!/usr/bin/env python3
"""Batch wrapper for PMA Stage 2 v0.4; keeps scientific classifiers unchanged."""
import argparse, csv, json, re, subprocess, sys, datetime
from pathlib import Path

PATTERN = re.compile(r'(?P<num>\d+)(?P<sex>[fm])_(?P<age>[at])_(?P<day>wt2?|tp\d+|lp\d+)$', re.I)
TABLES = ('session_summary.csv','trial_summary.csv','baseline_summary.csv','behavior_events.csv','reward_visits.csv','trial_probability_bins.csv','session_probabilities.csv')

def scan(root):
    sessions=[]
    for p in root.rglob('*'):
        if p.is_dir() and PATTERN.fullmatch(p.name) and PATTERN.fullmatch(p.parent.name) is None:
            sessions.append(p)
    return sorted(sessions, key=lambda p: str(p).lower())

def is_baseline(session):
    m=PATTERN.fullmatch(session.name)
    return m.group('day').lower() in ('wt','wt2')

def required(session):
    if is_baseline(session):
        if not ((session/'PMA_QC'/'events_corrected_timing.csv').is_file() or (session/'behavioral_events.csv').is_file()):
            return [str(session/'behavioral_events.csv') + ' OR PMA_QC/events_corrected_timing.csv']
        return []
    needed=[session/'PMA_QC'/'events_corrected_timing.csv']
    if not is_baseline(session):
        needed += [session/'PMA_QC'/'frame_time_mapping.csv',session/'zone_rois.json']
        if not (list(session.glob('*filtered.h5')) or list(session.glob('*filtered.csv'))):
            needed.append(session/'[missing filtered DLC h5 or csv]')
    return [str(p) for p in needed if not p.exists()]

def completed(session):
    out=session/'PMA_ANALYSIS'
    names=['session_summary.csv'] if is_baseline(session) else ['session_summary.csv','trial_summary.csv','trial_probability_bins.csv','behavior_events.csv','reward_visits.csv']
    return all((out/n).exists() for n in names)

def append_table(root, name, sessions, index_dir):
    # Merge only light summary/event tables, NOT the huge frame_behaviors.csv.
    paths=[(s,s/'PMA_ANALYSIS'/name) for s in sessions]
    paths=[(s,p) for s,p in paths if p.is_file()]
    if not paths:return 0
    colnames=['source_session','session_path']
    rows=[]
    for s,p in paths:
        with p.open(newline='',encoding='utf-8-sig') as fh:
            for row in csv.DictReader(fh):
                for k in row:
                    if k not in colnames:colnames.append(k)
                rows.append({'source_session':s.name,'session_path':str(s), **row})
    output=index_dir/name
    with output.open('w',newline='',encoding='utf-8') as fh:
        writer=csv.DictWriter(fh,fieldnames=colnames,extrasaction='ignore');writer.writeheader();writer.writerows(rows)
    return len(rows)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True, help='Root schema directory')
    p.add_argument('--dry-run',action='store_true',help='Show session readiness without running analysis')
    p.add_argument('--force',action='store_true',help='Reprocess existing sessions (overwrites PMA_ANALYSIS outputs)')
    p.add_argument('--confidence',type=float,default=.9)
    p.add_argument('--platform-points',type=int,default=4)
    p.add_argument('--midback-persistence-ms',type=float,default=200.)
    p.add_argument('--boundary-px',type=float,default=3.)
    p.add_argument('--small-mismatch-max',type=int,default=20)
    p.add_argument('--transition-ms',type=float,default=500.)
    args=p.parse_args()
    root=args.root.expanduser().resolve()
    if not root.is_dir():p.error(f'Not a directory: {root}')
    analyzer=Path(__file__).with_name('pma_stage2.py')
    if not analyzer.is_file():p.error(f'Missing {analyzer}')
    sessions=scan(root)
    if not sessions:p.error(f'No matching sessions found under {root}')
    index_dir=root/'PMA_BATCH'
    index_dir.mkdir(exist_ok=True)
    log=[]
    for i,s in enumerate(sessions,1):
        missing=required(s)
        if missing:status='missing_inputs';message='; '.join(missing)
        elif completed(s) and not args.force:status='skipped_existing';message='Existing result retained (use --force to rerun)'
        elif args.dry_run:status='ready_baseline' if is_baseline(s) else 'ready';message='Arduino-only baseline inputs found' if is_baseline(s) else 'Required inputs found'
        else:
            cmd=[sys.executable,str(analyzer),'--session',str(s),'--confidence',str(args.confidence),'--platform-points',str(args.platform_points), '--midback-persistence-ms',str(args.midback_persistence_ms),'--boundary-px',str(args.boundary_px),'--transition-ms',str(args.transition_ms),'--small-mismatch-max',str(args.small_mismatch_max)]
            proc=subprocess.run(cmd, text=True, capture_output=True)
            status='ok' if proc.returncode==0 and completed(s) else 'failed'
            message=(proc.stdout+'\n'+proc.stderr).strip()[-8000:]
        log.append({'session':s.name,'path':str(s),'day':PATTERN.fullmatch(s.name).group('day').upper(),'age':'adult' if '_a_' in s.name else 'adolescent','sex':'female' if PATTERN.fullmatch(s.name).group('sex').lower()=='f' else 'male','status':status,'message':message})
        print(f'[{i}/{len(sessions)}] {s.name}: {status}',flush=True)
    with (index_dir/'batch_run_status.csv').open('w',newline='',encoding='utf-8') as fh:
        writer=csv.DictWriter(fh,fieldnames=list(log[0]));writer.writeheader();writer.writerows(log)
    count={}
    if not args.dry_run:
        for table in TABLES:
            count[table]=append_table(root,table,[s for s in sessions if completed(s)],index_dir)
    manifest={'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'root':str(root),'script_version':'v0.4 classifiers + batch v0.3 (Arduino-only WT/WT2 fallback)','sessions_found':len(sessions),'statuses':{x:sum(1 for r in log if r['status']==x) for x in sorted(set(r['status'] for r in log))},'aggregated_rows':count,'note':'Existing completed sessions included; data should be interpreted only after ROI/QC validation. Freezing is not calculated.'}
    (index_dir/'batch_manifest.json').write_text(json.dumps(manifest,indent=2))
    print(f'Batch index: {index_dir}')
    if any(r['status'] in ('failed','missing_inputs') for r in log) and not args.dry_run:sys.exit(1)
if __name__=='__main__':main()
