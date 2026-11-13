"""Reference-style glowing safety ellipses on actual RK3576 detections.

Effects are presentation overlays; region membership comes from board traces.
No invented people, generated video, or timeline-triggered alerts.
"""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
import cv2
import numpy as np
from PIL import Image,ImageDraw
from tools.render_style import font,text,GREEN,AMBER,RED,WHITE,MUTED,BG
from scripts.engine import _vehicle_ellipse


def halos(raw,row,summary):
    h,w=raw.shape[:2]
    yy,xx=np.mgrid[:h,:w].astype(np.float32)
    frame=cv2.cvtColor(raw,cv2.COLOR_BGR2RGB).astype(np.float32)
    visible=np.ones((h,w),np.float32)
    # Keep the glow behind the tracked objects. This is bbox masking, not segmentation.
    boxes=list(row['forklifts'].values())+[p['box'] for p in row['people'].values()]
    for box in boxes:
        x1,y1,x2,y2=[int(x) for x in box]
        visible[max(y1,0):min(y2,h),max(x1,0):min(x2,w)]=0
    extension=summary['image_zone']['horizontal_extension_box_widths']
    inner=summary['image_zone']['critical_ellipse_radius']
    warning=inner+summary['image_zone']['caution_radial_margin']
    for box in row['forklifts'].values():
        cx,cy,rx,ry=_vehicle_ellipse(box,extension)
        radius=np.sqrt(((xx-cx)/rx)**2+((yy-cy)/ry)**2)
        # Smooth color transitions mirror the three rings; colors are RGB.
        colors=np.empty((h,w,3),np.float32)
        for channel in range(3):
            colors[:,:,channel]=np.interp(radius,[0,inner,inner+.07,warning,1.35,1.5],
                [[255,72,45][channel],[255,85,30][channel],[255,157,36][channel],
                 [255,231,65][channel],[81,255,115][channel],[64,230,130][channel]])
        alpha=np.interp(radius,[0,inner,inner+.07,warning,1.25,1.42],[.27,.25,.20,.15,.09,0]).astype(np.float32)
        alpha*=visible
        frame=frame*(1-alpha[:,:,None])+colors*alpha[:,:,None]
        for bound,color in ((inner,RED),(warning,AMBER),(1.35,(104,255,149))):
            ring=(.63*np.exp(-((radius-bound)/.009)**2)+.26*np.exp(-((radius-bound)/.045)**2))*visible
            frame+=np.array(color,dtype=np.float32)[None,None,:]*ring[:,:,None]
        # Two subtle contour lines help sell the ground-plane projection.
        for bound in (.32,.82):
            ring=.15*np.exp(-((radius-bound)/.006)**2)*visible
            frame+=np.array([255,235,156],np.float32)[None,None,:]*ring[:,:,None]
    return np.clip(frame,0,255).astype(np.uint8)


def render(raw,row,summary):
    image=Image.fromarray(halos(raw,row,summary)).resize((1920,1080),Image.Resampling.LANCZOS)
    d=ImageDraw.Draw(image,'RGBA');scale=1.5
    pairs=row['pairs']; rank={'SAFE':0,'CAUTION':1,'CRITICAL':2}
    tier=max((p['tier'] for p in pairs),key=lambda x:rank[x],default='SAFE')
    tiercolor={'SAFE':GREEN,'CAUTION':AMBER,'CRITICAL':RED}[tier]
    # Soft top/bottom scrims keep footage dominant and text readable.
    for y in range(138):
        a=int(210*(1-y/138))
        d.line((0,y,1920,y),fill=(*BG,a))
    for y in range(1009,1080):
        d.line((0,y,1920,y),fill=(*BG,int(190*(y-1009)/71)))
    text(d,(30,15),'WAREHOUSE SAFETY',33,WHITE,True)
    text(d,(32,59),'Seeed Studio reComputer RK3576  |  YOLO11 + ByteTrack',22,(220,235,232))
    d.rounded_rectangle((1483,20,1890,91),radius=12,fill=(*BG,217))
    d.ellipse((1504,47,1518,61),fill=(*tiercolor,255))
    status={'SAFE':'MONITORING','CAUTION':'CAUTION','CRITICAL':'DANGER'}[tier]
    text(d,(1533,31),status,32,tiercolor,True)
    if tier=='CRITICAL':
        d.rectangle((3,3,1916,1076),outline=(*RED,230),width=6)
    for ident,box in row['forklifts'].items():
        x1,y1,x2,y2=[x*scale for x in box]
        d.rounded_rectangle((x1,y1,x2,y2),radius=3,outline=(*GREEN,240),width=3)
        d.rounded_rectangle((x1,y1-39,x1+246,y1),radius=4,fill=(*BG,230))
        text(d,(x1+9,y1-36),f'FORKLIFT  #{ident}',25,GREEN,True)
    for ident,p in row['people'].items():
        x1,y1,x2,y2=[x*scale for x in p['box']]
        operator=p['excluded_as_operator']; current=next((x for x in pairs if str(x['person_id'])==ident),None)
        pc=(174,194,202) if operator else {'SAFE':(90,255,135),'CAUTION':AMBER,'CRITICAL':RED}.get(current['tier'] if current else 'SAFE')
        d.rounded_rectangle((x1,y1,x2,y2),radius=3,outline=(*pc,245),width=2 if operator else 4)
        label='OPERATOR' if operator else f'PERSON  #{ident}'
        width=177 if operator else 190
        d.rounded_rectangle((x1,y1-39,x1+width,y1),radius=4,fill=(*pc,255))
        text(d,(x1+9,y1-36),label,25,BG,True)
        if current and current['tier']!='SAFE':
            forklift=row['forklifts'].get(str(current['forklift_id']))
            cx=_vehicle_ellipse(forklift,summary['image_zone']['horizontal_extension_box_widths'])[0]*scale if forklift else (x1+x2)/2
            px,py=min(max(cx,x1),x2),y2
            d.ellipse((px-8,py-8,px+8,py+8),fill=(*pc,255))
    if tier!='SAFE':
        d.rounded_rectangle((30,909,892,996),radius=11,fill=(*BG,224))
        d.rectangle((30,921,36,984),fill=(*tiercolor,255))
        title='PEDESTRIAN NEAR FORKLIFT' if tier=='CAUTION' else 'DANGER - PEDESTRIAN TOO CLOSE'
        text(d,(53,918),title,33,tiercolor,True)
        text(d,(54,960),'Alert triggered by tracked position within the projected zones',22,WHITE)
    text(d,(30,1035),'PROJECTED IMAGE ZONES  /  DEMO ONLY  /  NOT CALIBRATED DISTANCES',20,WHITE,True)
    text(d,(1488,1035),f'SOURCE  {row["time_s"]:05.2f} / 10.00 s',21,WHITE,True)
    return cv2.cvtColor(np.array(image),cv2.COLOR_RGB2BGR)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',default='outputs/rk3576-halo-int8')
    p.add_argument('--output',default='outputs/showcase/warehouse-safety-halo-master.mp4')
    a=p.parse_args();r=Path(a.run)
    rows=[json.loads(x) for x in (r/'frame-trace.jsonl').read_text().splitlines()]
    summary=json.loads((r/'summary.json').read_text())
    out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True)
    cap=cv2.VideoCapture('data/test_clip_nearmiss.mp4')
    writer=cv2.VideoWriter(str(out),cv2.VideoWriter_fourcc(*'mp4v'),24,(1920,1080))
    if not writer.isOpened():raise RuntimeError('Cannot open video writer')
    for i,row in enumerate(rows):
        ok,raw=cap.read()
        if not ok:raise RuntimeError('Trace/video length mismatch')
        frame=render(raw,row,summary)
        writer.write(frame)
        if i in (71,119,143,191):cv2.imwrite(str(out.parent/f'halo-preview-{i+1:04d}.jpg'),frame)
    cap.release();writer.release()
    print(f'Rendered {len(rows)} actual RK3576 frames')


if __name__=='__main__':main()
