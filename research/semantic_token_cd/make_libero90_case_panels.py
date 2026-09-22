#!/usr/bin/env python3
from __future__ import annotations
import argparse, csv, json
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import imageio.v2 as imageio
import numpy as np


def read_frames(path: Path):
    reader=imageio.get_reader(path)
    frames=[np.asarray(frame) for frame in reader]
    reader.close()
    return frames


def pick_indices(n, first_guided, first_major):
    if n<=0: return []
    picks=[0, int(0.25*(n-1)), int(0.50*(n-1)), int(0.75*(n-1)), n-1]
    for x in (first_guided, first_major):
        if x is not None and 0 <= int(x) < n: picks.insert(0,int(x))
    out=[]
    for x in picks:
        if x not in out: out.append(x)
    return sorted(out)[:6]


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--root',type=Path,required=True); ap.add_argument('case_ids',nargs='+'); a=ap.parse_args()
    root=a.root.resolve(); out=root/'diagnostic_panels'; out.mkdir(parents=True,exist_ok=True)
    rows={r['case_id']:r for r in csv.DictReader(open(root/'CASE_EVENT_TABLE.csv'))}
    font=ImageFont.load_default()
    for case_id in a.case_ids:
        if case_id not in rows:
            print('missing',case_id); continue
        r=rows[case_id]; task=r['task']; iid=int(r['init_state_id'])
        framesets={}
        for arm in ('vanilla','matched'):
            p=root/'videos'/task/arm/f'init_{iid:03d}.mp4'
            framesets[arm]=read_frames(p)
        n=min(len(framesets['vanilla']),len(framesets['matched']))
        idxs=pick_indices(n, int(r['first_guided_change_step']) if r['first_guided_change_step'] else None,
                           int(r['first_major_divergence_step']) if r['first_major_divergence_step'] else None)
        cols=[]
        for arm in ('vanilla','matched'):
            row=[]
            for idx in idxs:
                frame=framesets[arm][min(idx,len(framesets[arm])-1)]
                im=Image.fromarray(frame).resize((224,224)).convert('RGB')
                d=ImageDraw.Draw(im); d.rectangle((0,0,80,18),fill=(0,0,0)); d.text((3,3),f"{arm} s{idx}",fill=(255,255,255),font=font)
                row.append(im)
            cols.append(row)
        width=224*len(idxs); height=224*2+52
        canvas=Image.new('RGB',(width,height),'white'); d=ImageDraw.Draw(canvas)
        d.text((4,4),f"{case_id} | {r['group']} | V={r['vanilla_success']} M={r['matched_success']} | {r['instruction']}",fill=(0,0,0),font=font)
        d.text((4,20),f"V/M steps={r['vanilla_steps']}/{r['matched_steps']} first_div={r['first_major_divergence_step']} first_guided={r['first_guided_change_step']} | columns: {idxs}",fill=(0,0,0),font=font)
        for ri,row in enumerate(cols):
            for ci,im in enumerate(row): canvas.paste(im,(ci*224,52+ri*224))
        p=out/f'{case_id}.png'; canvas.save(p); print(p)

if __name__=='__main__': main()
