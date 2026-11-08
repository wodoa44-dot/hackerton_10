"""새 폴더 구조와 이전 저장 모델의 모듈 경로 호환을 점검합니다."""
import importlib
import importlib.util
import pickle
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

class LayoutCompatibilityTests(unittest.TestCase):
    def test_old_and_new_preprocess_are_the_same_module(self):
        self.assertIs(importlib.import_module("erp_model.preprocess"),importlib.import_module("erp_model.encoders.preprocess"))

    def test_saved_pickle_global_resolves_after_moving_code(self):
        from erp_model.encoders.preprocess import Prepared
        self.assertIs(pickle.loads(b"cerp_model.preprocess\nPrepared\n."),Prepared)

    def test_every_legacy_alias_resolves_to_the_canonical_source(self):
        from erp_model._compat import ALIASES
        for old,new in ALIASES.items():
            with self.subTest(module=old):
                self.assertEqual(importlib.util.find_spec(old).origin,importlib.util.find_spec(new).origin)

    def test_legacy_numeric_module_still_runs_with_dash_m(self):
        subprocess.run([sys.executable,"-m","erp_model.numeric_inputs"],cwd=ROOT,check=True,capture_output=True,text=True)

    def test_root_entrypoint_help_works_from_another_directory(self):
        with tempfile.TemporaryDirectory(dir="/data/tools") as directory:
            result=subprocess.run([sys.executable,str(ROOT/"run.py"),"--help"],cwd=directory,check=True,capture_output=True,text=True)
        for command in ["web","train","evaluate","report"]:
            self.assertIn(command,result.stdout)

    def test_report_command_does_not_import_model_dependencies(self):
        result=subprocess.run([sys.executable,str(ROOT/"run.py"),"report","--help"],cwd=ROOT,check=True,capture_output=True,text=True)
        self.assertIn("--input",result.stdout)
        self.assertIn("--out",result.stdout)

    def test_model_namespace_remains_lazy_for_cli_tools(self):
        result=subprocess.run([sys.executable,"-c","import sys,erp_model.models; assert 'torch' not in sys.modules; assert 'numpy' not in sys.modules"],cwd=ROOT,check=True,capture_output=True,text=True)
        self.assertEqual(result.returncode,0)

    def test_grouped_layout_and_rule_bridge_contract(self):
        from erp_model.shared.diagnostics import check
        report=check()
        self.assertEqual(report["status"],"passed")
        self.assertTrue(report["legacy_pickle_global_verified"])
        self.assertTrue(report["node_rule_bridge_verified"])

if __name__=="__main__":
    unittest.main()
