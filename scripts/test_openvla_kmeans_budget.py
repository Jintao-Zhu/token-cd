#!/usr/bin/env python3
"""CPU-only tests for isolated K sweep request validation and delegation."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research.semantic_token_cd import openvla_kmeans_budget_rollout as runner

class SweepTests(unittest.TestCase):
    def test_arm_names(self):
        for k in (8,16,32):
            self.assertEqual(runner.parse_arm(f'matched_k{k:02d}_l100'),('matched',1.0,k))
        for a in ('matched_k16_l050','fixed016_l100','matched_k64_l100','matched_k16_l100,matched_k32_l100'):
            with self.assertRaises(ValueError): runner.parse_arm(a)

    def test_strict_requests(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp); snap=p/'snapshots'
            lock={'protocol_id':'TEST','tasks':['task'],'arms':['matched_k16_l100'],
                  'seeds':[200,201], 'seed_blocks':[[200],[201]],'kmeans_K':{'matched_k16_l100':16},
                  'lambda':1.0,'kmeans_seed':0,'snapshot_artifact':str(snap)}
            (p/'EXPERIMENT_LOCK.json').write_text(json.dumps(lock))
            runner.validate_request(p,'task','matched_k16_l100',[200],1,'TEST',snap)
            for args in [('task',[200],0,'TEST'),('task',[200,201],1,'TEST'),('task',[200],1,'WRONG'),('wrong',[200],1,'TEST')]:
                with self.assertRaises(ValueError): runner.validate_request(p,args[0],'matched_k16_l100',args[1],args[2],args[3],snap)
            out=p/'episodes/task/matched_k16_l100'; out.mkdir(parents=True)
            (out/'episode_200_arrays.npz').touch()
            with self.assertRaises(RuntimeError): runner.validate_request(p,'task','matched_k16_l100',[200],1,'TEST',snap)

    def test_k_audit(self):
        row={'kmeans_K':16,'kmeans_seed':0,'coverage_mode':'own_state_matched_budget','m_t':2,
             'reference_shr_token_ids':[1,2],'selected_token_ids':[3,4],
             'selected_group_ids':[3],'matched_budget_group_ids':[3]}
        self.assertTrue(all(runner.k_checks([row],'matched_k16_l100').values()))
        self.assertFalse(all(runner.k_checks([row],'matched_k32_l100').values()))
        self.assertFalse(all(runner.k_checks([],'matched_k16_l100').values()))
        for field,value in [('m_t',3),('kmeans_seed',1),('coverage_mode','fixed_top_2'),('selected_token_ids',[3,3]),('selected_group_ids',[99])]:
            self.assertFalse(all(runner.k_checks([{**row,field:value}],'matched_k16_l100').values()))

    def test_real_reference_trace(self):
        p=ROOT/'artifacts/openvla_lambda1_extension_200_299_v1/episodes/google_robot_close_drawer/lambda_100/episode_200_summary.json'
        d=json.loads(p.read_text())
        self.assertTrue(all(runner.k_checks(d['selector_trace'],'matched_k08_l100').values()))

    def test_wrapper_changes_only_requested_k(self):
        # Test delegation without importing torch, SIMPLER, or allocating a GPU.
        package=__import__('research.semantic_token_cd',fromlist=['unused'])
        fake=types.ModuleType('research.semantic_token_cd.openvla_budget_rule_rollout')
        fake.parse_seeds=lambda s: [200] if s=='200' else list(range(200,300))
        def build(model,task,arms):
            return {arm:types.SimpleNamespace(kmeans_K=8,kmeans_seed=0,alpha=1.0,selection_count=None) for arm in arms}
        fake.build_policies=build
        fake.audit=lambda trace,arm,task:{'technical_pass':True,'old_audit':True}
        saved=[]
        fake.atomic_json=lambda path,payload:saved.append((path,payload))
        def run_base():
            arm='matched_k32_l100'
            self.assertEqual(fake.parse_arm(arm),('matched',1.0,32))
            p=fake.build_policies(None,'google_robot_close_drawer',(arm,))[arm]
            self.assertEqual((p.kmeans_K,p.kmeans_seed,p.alpha,p.selection_count),(32,0,1.0,None))
            row={'kmeans_K':32,'kmeans_seed':0,'coverage_mode':'own_state_matched_budget','m_t':1,
                 'reference_shr_token_ids':[5],'selected_token_ids':[6],
                 'selected_group_ids':[1],'matched_budget_group_ids':[1]}
            self.assertTrue(fake.audit([row],arm,'task')['technical_pass'])
            fake.atomic_json(Path('CONFIG_LOCK.json'),{'original':'retained'})
            fake.atomic_json(Path(tmp)/'episode_200_summary.json',{'technical_pass':True})
        fake.main=run_base
        with tempfile.TemporaryDirectory() as tmp:
            artifact=Path(tmp); snap=ROOT/'artifacts/vanilla_recon_shr_canonical_0_299_v2'
            lock={'protocol_id':'TEST','tasks':['google_robot_close_drawer'],'arms':['matched_k32_l100'],
                  'seeds':list(range(200,300)),'seed_blocks':[[200]],'kmeans_K':{'matched_k32_l100':32},
                  'lambda':1.0,'kmeans_seed':0,'snapshot_artifact':str(snap)}
            (artifact/'EXPERIMENT_LOCK.json').write_text(json.dumps(lock))
            argv=['runner','--artifact',tmp,'--task','google_robot_close_drawer','--arms','matched_k32_l100',
                  '--seeds','200','--gpu','1','--worker-id','test','--protocol-id','TEST','--snapshot-artifact',str(snap)]
            with patch.dict(sys.modules,{fake.__name__:fake}),patch.object(package,'openvla_budget_rule_rollout',fake,create=True),patch.object(sys,'argv',argv):
                runner.main()
            self.assertIn('invocations',str(saved[0][0]))
            self.assertEqual(saved[0][1]['kmeans_K'],{'matched_k32_l100':32})
            self.assertEqual(saved[1][1]['process_seed_block'],[200])
            self.assertFalse((artifact/'CONFIG_LOCK.json').exists())

if __name__=='__main__':
    unittest.main()
