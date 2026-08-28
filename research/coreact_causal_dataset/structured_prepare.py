#!/usr/bin/env python3
"""Prepare structured object/interaction/background qualification from clean-only snapshots."""
from __future__ import annotations
import argparse, json, platform, shutil, socket
from datetime import datetime
from pathlib import Path
import torch, yaml
from research.coreact_causal_dataset.common import derive_seed
from research.coreact_closed_loop.runtime import CHECKPOINT_REVISION, checkpoint_path
from research.coreact_closed_loop.prepare_pilot import sha256_file

TASK_IDS=(4,7,9); BASE_SEED=823_000_000; K=5
EXCLUDED_SNAPSHOTS={"spatial_t09_i06_pre_grasp"}
def main():
 p=argparse.ArgumentParser(); p.add_argument('--workspace',type=Path,required=True); p.add_argument('--source-artifact',type=Path,required=True); a=p.parse_args()
 w=a.workspace.resolve(); src=a.source_artifact.resolve(); stamp=datetime.now().astimezone().strftime('%Y%m%d_%H%M%S')
 art=w/'artifacts'/f'coreact_structured_causal_unit_qualification_v1_{stamp}'; art.mkdir(parents=False,exist_ok=False)
 for d in ('snapshots','candidate_audits','episodes','logs'): (art/d).mkdir()
 source=[]
 for line in (src/'snapshot_plan.jsonl').read_text().splitlines():
  if line.strip(): source.append(json.loads(line))
 rows=[]
 for old in source:
  if old['snapshot_id'] in EXCLUDED_SNAPSHOTS: continue
  record=json.loads((src/'clean_extraction'/old['snapshot_id']/'trajectory.json').read_text())
  old={**old,**record}; state_src=src/old['state_path']; state_dst=art/'snapshots'/f"{old['snapshot_id']}.npy"; shutil.copy2(state_src,state_dst)
  for rep in range(K):
   seed=derive_seed(BASE_SEED,old['task_id'],old['init_state_id'],rep,10)
   rows.append({**old,'state_path':str(state_dst.relative_to(art)),'rollout_seed':seed,'replicate':rep,
                'snapshot_source':str(state_src.relative_to(src)),'proposal_seed':derive_seed(BASE_SEED,old['task_id'],old['init_state_id'],0,3)})
 protocol={'experiment_name':'structured_causal_unit_qualification_v1','stage':'development_not_confirmation',
  'source_snapshot_artifact':str(src),'tasks':list(TASK_IDS),'snapshots':44,'groups':['task_object','interaction','background_match'],
  'group_definition':'segmentation-grounded post-connector visual token sets; no attention selection',
  'intervention':'position-conditioned v8 visual mean; first 1 or 3 replans masked then vanilla',
  'durations':[1,3],'matched_noise_seeds':K,'expected_rollouts':1540,
  'paired_identity':['simulator state','language','preprocessing','solver','noise seed by replan','future conditions'],
  'object_group':'all eligible visual tokens whose 8x8 source footprint contains target object pixels',
  'interaction_group':'object group plus adjacent 8-neighbor visual tokens, excluding protected tokens',
  'background_group':'same count as each structured group, deterministic eligible background tokens',
  'replacement':'position_conditioned_modality_mean','attention_role':'not used for group selection',
  'no_classifier':True,'no_latent_intervention':True,'seed_base':BASE_SEED,
  'go_gate':'strong |tau| fraction >= 0.15, both signs, report phase/duration stratification',
  'source_snapshot_outcomes_not_used':True,
  'qualification_exclusion':{'snapshot_id':'spatial_t09_i06_pre_grasp','reason':'target object absent from both segmentation-mapped visual token footprints','intervention_outcomes_used':False}}
 ck=checkpoint_path(w); env={'timestamp':datetime.now().astimezone().isoformat(),'hostname':socket.gethostname(),
  'platform':platform.platform(),'python':platform.python_version(),'torch':torch.__version__,'cuda':torch.version.cuda,
  'gpu':torch.cuda.get_device_name(0),'checkpoint_revision':CHECKPOINT_REVISION,
  'checkpoint_config_sha256':sha256_file(ck/'config.json'),'checkpoint_weights_sha256':sha256_file(ck/'model.safetensors'),
  'calibration_mean_sha256':sha256_file(w/'artifacts/coreact_exploration_v8_20260807_002951/modality_means.pt')}
 (art/'protocol.lock.yaml').write_text(yaml.safe_dump(protocol,sort_keys=False)); (art/'environment.json').write_text(json.dumps(env,indent=2)+'\n')
 (art/'source_snapshot_manifest.jsonl').write_text(''.join(json.dumps(r,sort_keys=True)+'\n' for r in rows))
 print(art)
if __name__=='__main__': main()
