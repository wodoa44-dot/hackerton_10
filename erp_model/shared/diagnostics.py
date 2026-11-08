"""외부 학습 라이브러리 없이 실행하는 구조·호환 경로 점검."""
import ast
import hashlib
import importlib.util
import json
import pickle
import shutil
import subprocess
from pathlib import Path
from erp_model._compat import ALIASES
from erp_model.shared.paths import PACKAGE_ROOT, PROJECT_ROOT

def check():
    python_files=list(PACKAGE_ROOT.rglob("*.py"))
    imports=0
    for file in python_files:
        tree=ast.parse(file.read_text(encoding="utf-8"),filename=str(file))
        for node in ast.walk(tree):
            if isinstance(node,ast.ImportFrom) and node.level==0 and node.module and node.module.startswith("erp_model"):
                name=ALIASES.get(node.module,node.module)
                target=PROJECT_ROOT.joinpath(*name.split("."))
                assert target.with_suffix(".py").is_file() or (target/"__init__.py").is_file(), (file,node.module)
                imports+=1
    for old,new in ALIASES.items():
        assert importlib.util.find_spec(old).origin==importlib.util.find_spec(new).origin,old
    import erp_model.preprocess as old_preprocess
    import erp_model.encoders.preprocess as new_preprocess
    assert old_preprocess is new_preprocess
    assert pickle.loads(b"cerp_model.preprocess\nPrepared\n.") is new_preprocess.Prepared
    from erp_model.encoders.rules.company_schema import CompanySchemas
    book=CompanySchemas()
    parsed=book.parse("DW1N4148-D3TV")
    assert parsed is not None
    audit=json.loads((PROJECT_ROOT/"artifacts/model_cleanup_v7.json").read_text())
    for item in audit["kept_required_files"]:
        relative=Path(item["path"]).relative_to("/data/main")
        filename=PROJECT_ROOT/relative
        assert hashlib.sha256(filename.read_bytes()).hexdigest()==item["sha256"],relative
    node=shutil.which("node")
    bridge_verified=False
    if node:
        bridge=PACKAGE_ROOT/"encoders/rules/company_bridge.cjs"
        response=subprocess.run([node,str(bridge)],input=json.dumps({"category":"D","text":"DIODE 1N4148","neighbors":[]})+"\n",text=True,capture_output=True,check=True,timeout=15)
        result=json.loads(response.stdout)
        assert "error" not in result,result
        assert result["proposals"] and result["proposals"][0]["id"]=="D",result
        bridge_verified=True
    return {"status":"passed","python_files_parsed":len(python_files),"internal_imports_checked":imports,"legacy_aliases_checked":len(ALIASES),"legacy_pickle_global_verified":True,"retained_model_files_hash_verified":len(audit["kept_required_files"]),"node_rule_bridge_verified":bridge_verified,"model_inference_rerun":False}
