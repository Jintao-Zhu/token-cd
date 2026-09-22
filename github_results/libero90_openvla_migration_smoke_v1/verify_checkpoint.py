#!/usr/bin/env python3
from __future__ import annotations
import hashlib, json, os
from pathlib import Path

repo = 'VQ-VLA/openvla-7b-finetuned-libero-90'
revision = '794ef81b7be928ea9270e81ca1ef5b60ffa9420f'
ckpt = Path('/home/leju-suzhou/zjt_ws/checkpoints/libero/vq-vla-openvla-7b-libero90')
out = Path('/home/leju-suzhou/zjt_ws/token-cd/artifacts/libero90_openvla_migration_smoke_v1/CHECKPOINT_MANIFEST.json')
expected = {
    'model-00001-of-00004.safetensors': '96398076431553f13c8c78d91730bde8765b3da7243afebd03c13d0d257c3848',
    'model-00002-of-00004.safetensors': 'bff3948179307db8b7fdada21273b3d5baa7ab76dd3989d4b821a5b96e8ab086',
    'model-00003-of-00004.safetensors': 'd98b5de7638d3ec18cdb7de3bc706885a70d9a5379288dcc3051da04362794b9',
    'model-00004-of-00004.safetensors': 'bdc2d24168c1f616dbad4eb886727afda92415efbc29d7de4ae47a14684dbf07',
}
def sha256(path: Path) -> str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(16*1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()
idx=json.loads((ckpt/'model.safetensors.index.json').read_text())
indexed=sorted(set(idx['weight_map'].values()))
if indexed != sorted(expected):
    raise RuntimeError(f'indexed files mismatch: {indexed}')
files=[]
for name, want in expected.items():
    p=ckpt/name
    got=sha256(p)
    if got != want:
        raise RuntimeError(f'sha256 mismatch {name}: {got} != {want}')
    files.append({'name':name,'bytes':p.stat().st_size,'sha256':got,'sha256_source':'huggingface_lfs'})
config=json.loads((ckpt/'config.json').read_text())
dataset_stats=json.loads((ckpt/'dataset_statistics.json').read_text())
manifest={
    'repo_id':repo,'revision':revision,'local_dir':str(ckpt),
    'architecture':config.get('architectures'),'model_type':config.get('model_type'),
    'language_backbone':config.get('hf_llm_id'),'n_action_bins':config.get('n_action_bins'),
    'checkpoint_norm_stats_keys':sorted((config.get('norm_stats') or {}).keys()),
    'dataset_statistics_keys':sorted(dataset_stats.keys()),
    'selected_unnorm_key':'libero_90_no_noops',
    'indexed_weight_files':indexed,
    'files':files,
    'index_total_size':idx.get('metadata',{}).get('total_size'),
}
out.write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n')
print(json.dumps({'ok':True,'out':str(out),'files':len(files),'keys':manifest['dataset_statistics_keys']},indent=2))
