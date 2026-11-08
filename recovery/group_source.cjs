const fs=require('fs'),path=require('path'),assert=require('assert/strict');
const ROOT='/data/main',PKG=ROOT+'/erp_model';
assert.equal(require('os').hostname(),'proxy-ssh-server-5d5b67c46b-jhc85');
const mapping={"accuracy_candidates":"encoders/candidates","accuracy_runtime":"prediction/accuracy_runtime","artifact_paths":"shared/artifact_paths","attribute_models":"models/attributes","batch_csv":"api/batch_csv","candidates_v2":"encoders/legacy_candidates","code_metrics":"prediction/code_metrics","code_rules":"encoders/rules/code_rules","company_schema":"encoders/rules/company_schema","compare_encoders":"evaluation/compare_encoders","encoder_data":"training/data/encoder_data","encoder_network":"encoders/cnn/network","encoder_runtime":"encoders/cnn/runtime","evaluate_encoder":"evaluation/evaluate_encoder","evaluate_generation":"evaluation/evaluate_generation","evaluate_spectrum":"evaluation/evaluate_spectrum","evaluation_audit":"evaluation/audit","existing_lookup":"models/lookup","exploratory":"prediction/exploratory","family_model":"models/family","features":"encoders/features","generation":"prediction/generation","inference":"prediction/inference","label_quality":"encoders/rules/label_quality","manual_audit":"training/data/manual_audit","material_runtime":"prediction/material_runtime","material_schema":"encoders/rules/material_schema","models":"models/tabular","numeric_inputs":"encoders/numeric_inputs","prepare_encoder_data":"training/data/prepare_encoder_data","preprocess":"encoders/preprocess","registry":"models/registry","schema":"shared/schema","service":"api/service","spectrum":"prediction/spectrum","train":"training/train","train_accuracy":"training/train_accuracy","train_attributes":"training/train_attributes","train_comparison_baseline":"training/train_comparison_baseline","train_encoder":"training/train_encoder","train_encoder_augmented":"training/train_encoder_augmented","train_generation":"training/train_generation","train_material":"training/train_material","train_verified":"training/train_verified","training_audit":"training/data/training_audit","training_io":"training/io","validate_accuracy":"evaluation/validate_accuracy","verified_predictor":"prediction/verified_predictor"};
const sourceNames=fs.readdirSync(PKG).filter(n=>n.endsWith('.py')&&!n.startsWith('__')).map(n=>n.slice(0,-3));
assert.deepEqual([...sourceNames].sort(),Object.keys(mapping).sort());
function rewrite(text){
  for(const [old,target] of Object.entries(mapping))text=text.replace(new RegExp('\\berp_model\\.'+old+'\\b','g'),'erp_model.'+target.replaceAll('/','.'));
  return text.replace(/(\bfrom\s+)\.([A-Za-z_]\w*)(\s+import\b)/g,(_m,a,name,z)=>{assert.ok(mapping[name],name);return a+'erp_model.'+mapping[name].replaceAll('/','.')+z;});
}
function initDirs(directory){
  if(directory===PKG)return;
  initDirs(path.dirname(directory));fs.mkdirSync(directory,{recursive:true});
  const init=directory+'/__init__.py';if(!fs.existsSync(init))fs.writeFileSync(init,'"""ERP 기능별 모듈."""\n');
}
const originalMain=fs.readFileSync(PKG+'/__main__.py','utf8');
for(const [old,target] of Object.entries(mapping)){
  const source=PKG+'/'+old+'.py',destination=PKG+'/'+target+'.py';
  assert.equal(fs.existsSync(destination),false,destination);initDirs(path.dirname(destination));
  let text=rewrite(fs.readFileSync(source,'utf8'));
  if(old==='company_schema'){text='from erp_model.shared.paths import PROJECT_ROOT\n'+text;text=text.replace('Path(__file__).resolve().parents[1] / "config" / "company_rules.json"','PROJECT_ROOT / "config" / "company_rules.json"');}
  if(old==='generation'){text='from erp_model.shared.paths import PACKAGE_ROOT\n'+text;text=text.replace('Path(__file__).with_name("company_bridge.cjs")','PACKAGE_ROOT / "encoders" / "rules" / "company_bridge.cjs"');}
  if(old==='service'){text='from erp_model.shared.paths import PACKAGE_ROOT\n'+text;text=text.replace('Path(__file__).with_name("web")','PACKAGE_ROOT / "web"');}
  fs.writeFileSync(destination,text);fs.unlinkSync(source);
}
fs.writeFileSync(PKG+'/shared/legacy_cli.py',rewrite(originalMain));
fs.writeFileSync(PKG+'/shared/paths.py','"""소스 이동과 무관하게 고정되는 프로젝트 경로."""\nfrom pathlib import Path\nPACKAGE_ROOT = Path(__file__).resolve().parents[1]\nPROJECT_ROOT = PACKAGE_ROOT.parent\n');
for(const name of ['company_bridge.cjs','company_engine.js']){
  let text=fs.readFileSync(PKG+'/'+name,'utf8');if(name.endsWith('.cjs'))text=text.replace("'../config/company_rules.json'","'../../../config/company_rules.json'");
  fs.writeFileSync(PKG+'/encoders/rules/'+name,text);fs.unlinkSync(PKG+'/'+name);
}
for(const dir of ['deploy']){
 for(const file of fs.readdirSync(ROOT+'/'+dir).filter(n=>n.endsWith('.py'))){
  const filePath=ROOT+'/'+dir+'/'+file;fs.writeFileSync(filePath,rewrite(fs.readFileSync(filePath,'utf8')));
 }
}
fs.renameSync(ROOT+'/reports',PKG+'/evaluation/report');
fs.writeFileSync(ROOT+'/recovery/source_layout_map.json',JSON.stringify(mapping,null,2)+'\n');
console.log(JSON.stringify({moved_python_modules:Object.keys(mapping).length,report:PKG+'/evaluation/report',models_untouched:true}));
