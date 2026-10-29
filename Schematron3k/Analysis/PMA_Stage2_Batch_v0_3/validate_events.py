#!/usr/bin/env python3
"""Browse Stage2 events on video frames with DLC and ROI overlays.
Use left/right or A/D to navigate events, comma/period to step frames, Q/Esc to quit.
"""
import argparse,json
from pathlib import Path
import cv2,numpy as np,pandas as pd
from pma_stage2 import locate,load_dlc,coords,PARTS

def main():
 p=argparse.ArgumentParser();p.add_argument('--session',type=Path,required=True);p.add_argument('--event',default='all',choices=['all','platform_entry','platform_exit','head_dip','reward_entry','reward_exit']);a=p.parse_args()
 folder=a.session; out=folder/'PMA_ANALYSIS';ev=pd.read_csv(out/'behavior_events.csv')
 if a.event!='all':ev=ev[ev.event==a.event]
 ev=ev.sort_values('frame_row').reset_index(drop=True)
 if not len(ev):raise SystemExit('No events to inspect')
 video=locate(folder,['*.mp4']);cap=cv2.VideoCapture(str(video))
 if not cap.isOpened():raise RuntimeError('Cannot open video: '+str(video))
 dlc=load_dlc(locate(folder,['*filtered.h5','*filtered.csv']))
 mapping=pd.read_csv(out/'frame_behaviors.csv',usecols=['frame_index','time_s','platform','reward','midback_transition_state'])
 roi=json.loads((folder/'zone_rois.json').read_text())['zones']
 parts={k:coords(dlc,k) for k in PARTS}
 index=0;delta=0;window='PMA Event Validation (A/D events, comma/period frames, Q quit)'
 cv2.namedWindow(window,cv2.WINDOW_NORMAL)
 while True:
  e=ev.iloc[index]; f=max(0,min(len(mapping)-1,int(e.frame_row)+delta))
  cap.set(cv2.CAP_PROP_POS_FRAMES,f);ok,img=cap.read()
  if not ok:print('Could not decode frame',f);break
  for key,color in [('platform',(0,210,0)),('reward',(0,180,255))]:
   polygon=np.array(roi[key]['polygon_px'],dtype=np.int32);cv2.polylines(img,[polygon],True,color,2)
   x,y=polygon[0];cv2.putText(img,key,(int(x),int(y)-8),cv2.FONT_HERSHEY_SIMPLEX,.65,color,2)
  for j,(name,(xx,yy,pp)) in enumerate(parts.items()):
   if f>=len(xx) or not np.isfinite(xx[f]) or not np.isfinite(yy[f]) or pp[f]<.9:continue
   x,y=int(xx[f]),int(yy[f]); col=(255,50,50) if name=='mid_back' else (255,255,0)
   cv2.circle(img,(x,y),5,col,-1);cv2.putText(img,name,(x+7,y-4),cv2.FONT_HERSHEY_SIMPLEX,.35,col,1)
  h,w=img.shape[:2];img=cv2.copyMakeBorder(img,70,10,0,0,cv2.BORDER_CONSTANT,value=(30,30,30))
  t=mapping.iloc[f]; strings=[f'Event {index+1}/{len(ev)}: {e.event} @ {e.time_s:.3f}s   Current {t.time_s:.3f}s / frame {f}',
                       f'4-point platform={t.platform}  reward head state={t.reward}  mid-back={t.midback_transition_state}  delta={delta:+d} frames']
  for k,text in enumerate(strings):cv2.putText(img,text,(10,26+k*28),cv2.FONT_HERSHEY_SIMPLEX,.55,(255,255,255),1,cv2.LINE_AA)
  cv2.imshow(window,img);key=cv2.waitKey(0)&0xff
  if key in (ord('q'),27):break
  if key in (ord('d'),ord('n'),83):index=(index+1)%len(ev);delta=0
  if key in (ord('a'),ord('p'),81):index=(index-1)%len(ev);delta=0
  if key==ord('.'):delta+=1
  if key==ord(','):delta-=1
 cap.release();cv2.destroyAllWindows()
if __name__=='__main__':main()
