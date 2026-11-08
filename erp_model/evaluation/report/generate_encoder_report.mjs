#!/usr/bin/env node
/** 저장된 v6/v7 평가 결과만 사용합니다. 학습·새 예측·가중치 변경은 수행하지 않습니다. */
import fs from 'node:fs';
import path from 'node:path';
import assert from 'node:assert/strict';
import {SLOTS,LABELS,VARIANTS,MODELS,VIEWS,POPULATIONS,KS,DISPLAY_K,truthy,placeholder,sha256,json,saveCSV,hit,macroF1,loadExperiment,buildMetrics} from './report_data.mjs';
import {renderer,slotSpec,wholeSpec,tableSpec,saveFigure} from './report_plots.mjs';

const args=process.argv.slice(2);
if(args.includes('--help')){
  console.log('사용법: node generate_encoder_report.mjs --input artifacts/encoder_comparison --out artifacts/reports/encoder_topk');process.exit(0);
}
const option=(name,fallback)=>{const i=args.indexOf(name);if(i<0)return fallback;assert.ok(args[i+1]&&!args[i+1].startsWith('--'),name+' 값 누락');return args[i+1];};
for(let i=0;i<args.length;i+=2)assert.ok(['--input','--out'].includes(args[i]),'알 수 없는 인자 '+args[i]);
const input=path.resolve(option('--input','artifacts/encoder_comparison'));
const out=path.resolve(option('--out','artifacts/reports/encoder_topk'));
assert.ok(input!==out&&!input.startsWith(out+path.sep),'원본 평가 폴더에 덮어쓸 수 없습니다.');
const started=Date.now(),experiment=loadExperiment(input),metrics=buildMetrics(experiment);
fs.mkdirSync(out,{recursive:true});
const emit=(name,rows,columns)=>saveCSV(path.join(out,name),rows,columns);
const pct=x=>x==null?null:x*100;
const lookup=(view,population,slot,variant,k)=>metrics.find(r=>r.view===view&&r.population===population&&r.slot===slot&&r.variant===variant&&r.k===k);
const classMap=new Map(experiment.parts.no_encoder.filter(r=>r.slot==='classification').map(r=>[r.source_row,r.truth]));
const sourceIndex=Object.fromEntries(VARIANTS.map(v=>[v,new Map(experiment.whole[v].map(r=>[r.source_row,r]))]));

emit('topk_by_slot.csv',metrics.map(r=>({
  관점:VIEWS[r.view],관점ID:r.view,모집단:POPULATIONS[r.population],모집단ID:r.population,코드항목:LABELS[r.slot],항목ID:r.slot,모델:MODELS[r.variant],모델ID:r.variant,
  K:r.k,평가행수:r.n,원래채점행수:r.all_n,제외된미지정정답:r.excluded_n,정답수:r.correct,정답률:r.rate,정답률_percent:pct(r.rate),Top1_MacroF1:r.macro_f1,상태:r.status,근거:r.basis,
})));
const summaries=[];
for(const view of Object.keys(VIEWS))for(const population of Object.keys(POPULATIONS))for(const slot of SLOTS){
  const a=lookup(view,population,slot,'no_encoder',1),b=lookup(view,population,slot,'encoder',1);
  assert.equal(a.n,b.n);
  const row={관점:VIEWS[view],모집단:POPULATIONS[population],코드항목:LABELS[slot],항목ID:slot,N:a.n,원래채점행수:a.all_n,제외된미지정정답:a.excluded_n};
  for(const k of DISPLAY_K){
    const left=lookup(view,population,slot,'no_encoder',k),right=lookup(view,population,slot,'encoder',k);
    row['미사용_Top'+k+'_percent']=pct(left.rate);row['CNN_Top'+k+'_percent']=pct(right.rate);
    row['차이_Top'+k+'_%p']=left.rate!=null&&right.rate!=null?(right.rate-left.rate)*100:null;
  }
  row.미사용_Top1_MacroF1=a.macro_f1;row.CNN_Top1_MacroF1=b.macro_f1;summaries.push(row);
}
emit('topk_summary.csv',summaries);
const paired=[];
for(const population of Object.keys(POPULATIONS))for(const slot of SLOTS){
  const rows=experiment.parts.no_encoder.filter(r=>r.slot===slot&&truthy(r.eligible)&&(population==='all'||!placeholder(r.truth)));
  for(const k of KS){
    let both=0,leftOnly=0,rightOnly=0,neither=0;
    for(const a of rows){const b=experiment.partIndex.encoder.get(a.source_row+'|'+slot),l=hit(a.raw,a.truth,k),r=hit(b.raw,b.truth,k);if(l&&r)both++;else if(l)leftOnly++;else if(r)rightOnly++;else neither++;}
    assert.equal(both+leftOnly+rightOnly+neither,rows.length);
    paired.push({모집단:POPULATIONS[population],코드항목:LABELS[slot],항목ID:slot,K:k,N:rows.length,둘다정답:both,미사용만정답:leftOnly,CNN만정답:rightOnly,둘다오답:neither,정답수차이:rightOnly-leftOnly,정답률차이_percentpoint:rows.length?(rightOnly-leftOnly)/rows.length*100:null});
  }
}
emit('paired_topk_changes.csv',paired);
const componentRows=[];
for(const variant of VARIANTS)for(const population of Object.keys(POPULATIONS))for(const slot of SLOTS){
  const grouped=new Map();
  for(const row of experiment.parts[variant]){
    if(row.slot!==slot||!truthy(row.eligible)||population==='designated'&&placeholder(row.truth))continue;
    const category=classMap.get(row.source_row);if(!grouped.has(category))grouped.set(category,[]);grouped.get(category).push(row);
  }
  for(const [category,rows] of grouped)for(const k of KS){
    const correct=rows.filter(r=>hit(r.raw,r.truth,k)).length;
    componentRows.push({정답분류코드:category,모집단:POPULATIONS[population],코드항목:LABELS[slot],항목ID:slot,모델:MODELS[variant],모델ID:variant,K:k,N:rows.length,정답수:correct,정답률:correct/rows.length,정답률_percent:correct/rows.length*100,Top1_MacroF1:k===1?macroF1(rows,r=>r.raw_model_value):null});
  }
}
emit('topk_by_component.csv',componentRows);
const wholeRows=[];
for(const variant of VARIANTS)for(const [group,data] of Object.entries(experiment.comparison.variants[variant].candidate_k_metrics.groups))for(const p of data.points)wholeRows.push({variant,group,k:p.k,n:p.n,correct:p.correct,rate:p.rate});
emit('whole_code_topk.csv',wholeRows.map(r=>({모델:MODELS[r.variant],모델ID:r.variant,범위:r.group,K:r.k,N:r.n,정답수:r.correct,정답률:r.rate,정답률_percent:pct(r.rate)})));
const wholePredictions=[];
for(const variant of VARIANTS)for(const r of experiment.whole[variant]){
  const row={모델:MODELS[variant],모델ID:variant,원본행:r.source_row,품목명:r.품목명,규격명:r.규격명,정답코드:r.truth,표시예측코드:r.displayed_code,후보1순위코드:r.effective_top1_code,매뉴얼해석가능:truthy(r.manual_parseable),자동확정:truthy(r.auto_confirmed),후보수:r.candidates.length,후보코드_Top10:r.candidates};
  for(const s of SLOTS)row[LABELS[s]+'_최종값']=r[s];
  for(const k of KS)row['완성코드_적중_Top'+k]=hit(r.candidates,r.truth,k);
  wholePredictions.push(row);
}
emit('private/predictions/whole_codes.csv',wholePredictions);
for(const slot of SLOTS){
  const rows=[];
  for(const variant of VARIANTS)for(const r of experiment.parts[variant].filter(r=>r.slot===slot)){
    const full=sourceIndex[variant].get(r.source_row),eligible=truthy(r.eligible);
    const row={모델:MODELS[variant],모델ID:variant,원본행:r.source_row,품목명:r.품목명,규격명:r.규격명,정답코드:full.truth,정답분류코드:classMap.get(r.source_row),코드항목:LABELS[slot],항목ID:slot,정답값:r.truth,채점가능:eligible,미지정정답:truthy(r.placeholder),지정값채점가능:eligible&&!placeholder(r.truth),원시모델_Top1:r.raw_model_value,원시모델_Top10:r.raw,최종표시값:r.displayed_value,최종코드:full.displayed_code,상태:r.status};
    for(const k of KS)row['원시모델_적중_Top'+k]=eligible?hit(r.raw,r.truth,k):null;
    rows.push(row);
  }
  emit('private/predictions/'+slot+'.csv',rows);
}
console.log('행별 결과 CSV 7개와 Top-K 비교 표를 저장했습니다.');
const tools=await renderer(),graphs={};
for(const view of Object.keys(VIEWS))for(const population of Object.keys(POPULATIONS)){
  const id=view+'_'+population;
  graphs[id]=await saveFigure(tools,slotSpec(metrics,view,population),path.join(out,'figures',id));
}
for(const population of Object.keys(POPULATIONS)){
  graphs['delta_'+population]=await saveFigure(tools,slotSpec(metrics,'raw',population,{delta:true}),path.join(out,'figures','delta_'+population));
  graphs['table_'+population]=await saveFigure(tools,tableSpec(metrics,population),path.join(out,'figures','table_'+population));
  for(const slot of SLOTS)await saveFigure(tools,slotSpec(metrics,'raw',population,{slot}),path.join(out,'figures','slots',slot+'_'+population));
}
graphs.whole=await saveFigure(tools,wholeSpec(wholeRows),path.join(out,'figures','whole_code_topk'));
const escape=x=>String(x??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const percent=x=>x==null?'—':(x*100).toFixed(2)+'%';
function table(view,population){
  const header='<tr><th>코드 항목</th><th>특징 인코더</th><th>N</th>'+DISPLAY_K.map(k=>'<th>Top-'+k+'</th>').join('')+'</tr>';
  const rows=SLOTS.flatMap(slot=>VARIANTS.map((variant,i)=>'<tr><th>'+escape(LABELS[slot])+'</th><td class="'+variant+'">'+escape(MODELS[variant])+'</td><td>'+lookup(view,population,slot,variant,1).n+'</td>'+DISPLAY_K.map(k=>'<td>'+percent(lookup(view,population,slot,variant,k).rate)+'</td>').join('')+'</tr>')).join('');
  return '<div class="table"><table><thead>'+header+'</thead><tbody>'+rows+'</tbody></table></div>';
}
const sections=Object.keys(VIEWS).flatMap(view=>Object.keys(POPULATIONS).map(population=>{
 const id=view+'_'+population;
 return '<details '+(id==='raw_designated'?'open':'')+'><summary>'+escape(VIEWS[view]+' · '+POPULATIONS[population])+'</summary>'+table(view,population)+graphs[id]+'<p><a href="figures/'+id+'.png">PNG 이미지</a> · <a href="figures/'+id+'.svg">SVG 벡터 이미지</a></p></details>';
})).join('\n');
const created=new Intl.DateTimeFormat('ko-KR',{timeZone:'Asia/Seoul',dateStyle:'medium',timeStyle:'short'}).format(new Date());
const html='<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>ERP 인코더 유무 Top-K 평가</title><style>body{font-family:system-ui,"Malgun Gothic",sans-serif;margin:0;background:#f3f6fb;color:#20334e}main{max-width:1280px;margin:auto;padding:24px}h1{font-size:28px}p,li{line-height:1.65}article,details{background:white;border:1px solid #dfe6f0;border-radius:12px;padding:20px;margin:18px 0}summary{font-weight:700;font-size:20px;cursor:pointer;margin-bottom:16px}svg{max-width:100%;height:auto;display:block}a{color:#174fac}.table{overflow-x:auto}table{width:100%;border-collapse:collapse;min-width:780px}th,td{padding:11px;border-bottom:1px solid #e4eaf3;text-align:right;white-space:nowrap}th:first-child,td:nth-child(2){text-align:left}.no_encoder{color:#2563eb}.encoder{color:#bd510d}.muted{color:#5c6c83;font-size:14px}.downloads{display:flex;gap:18px;flex-wrap:wrap}</style><main><h1>학습 특징 인코더 유무 · 6개 코드 항목별 Top-K</h1><p class="muted">생성: '+escape(created)+' · 기존 v6 / 문자 CNN 추가 v7 비교 · 평가 2,850행</p><article><strong>평가 기준</strong><ul><li>두 방식 모두 공통 전처리·규칙 검증·학습 라벨 마스크를 사용합니다. 비교되는 인코더는 추가 학습형 문자 CNN입니다.</li><li>원시 칸 모델은 각 모델의 후보를 직접 채점합니다. 규칙 반영 후보·완성 코드 칸·완성 코드 전체의 정답률은 별도로 표시합니다.</li><li>미지정 제외는 정답이 -, -- 등인 행만 제외합니다. 지정 정답에 대한 미지정 예측은 오답입니다. 후보가 부족해도 K별 분모 N은 같습니다.</li><li>기존 진단 데이터의 저장된 예측을 재채점했습니다. 학습이나 새 추론은 실행하지 않았고, 현재 웹의 v8 가중치 모델 성능과 섞지 않았습니다.</li><li>품번 원시 점수에는 숫자형과 문자형이 함께 들어 있습니다. Vendor 구간에는 코드 체계에 따른 제조사·버전 구간이 포함됩니다.</li><li>미지정 제외로 행 집합이 달라졌고 후보 기록이 부족한 지표는 —로 표시합니다. 행 집합이 같으면 기존 집계를 사용합니다. 0%나 다른 실험으로 채우지 않습니다.</li></ul><div class="downloads"><a href="topk_summary.csv">Top-1·3·5·10 비교 표</a><a href="topk_by_slot.csv">K=1~10 전체 지표</a><a href="paired_topk_changes.csv">인코더로 바뀐 정답·오답</a><a href="topk_by_component.csv">부품 분류별 세부 지표</a><span>행별 상세 CSV는 서버 private/predictions에 보관</span></div></article>'+sections+'<details><summary>인코더 추가 효과 · 원시 칸 모델 · 미지정 제외</summary>'+graphs.delta_designated+'</details><details><summary>완성 품목코드 전체 Top-K</summary>'+graphs.whole+'<p><a href="whole_code_topk.csv">완성 코드 지표 CSV</a> · <a href="figures/whole_code_topk.png">PNG</a></p></details><article><h2>항목별 결과 파일</h2><ul>'+SLOTS.map(s=>'<li>'+escape(LABELS[s])+': 서버 상세 CSV '+escape(s)+'.csv · <a href="figures/slots/'+s+'_designated.png">미지정 제외 그래프</a></li>').join('')+'</ul><p class="muted">출처·파일 해시·검증 횟수·실행 환경: <a href="run_manifest.json">run_manifest.json</a>. 인터넷 연결 없이 이 HTML의 내장 표와 그래프를 볼 수 있으며, CSV와 개별 이미지는 함께 생성된 파일을 사용합니다.</p></article></main></html>';
fs.writeFileSync(path.join(out,'report.html'),html);
let evaluationHashVerified=false;
const evaluationCSV=path.resolve(input,'../../dataset/ERP_원재료_inference.csv');
if(fs.existsSync(evaluationCSV)){assert.equal(sha256(fs.readFileSync(evaluationCSV)),experiment.comparison.protocol.evaluation_csv_sha256);evaluationHashVerified=true;}
const walk=directory=>fs.readdirSync(directory,{withFileTypes:true}).flatMap(entry=>entry.isDirectory()?walk(path.join(directory,entry.name)):[path.join(directory,entry.name)]);
const manifest={
 created_at:new Date().toISOString(),timezone:'Asia/Seoul',source_input:input,output:out,
 experiment:'동일 규칙 검증에 문자 CNN 특징만 추가한 v6/v7 비교',evaluation_rows:experiment.whole.no_encoder.length,
 ks:KS,table_ks:DISPLAY_K,training_or_new_inference_executed:false,v8_class_balance_results_mixed:false,
 input_files:experiment.files,evaluation_csv_sha256_verified:evaluationHashVerified,
 source_protocol:experiment.comparison.protocol,validation_checks:experiment.checks,
 metric_rows:metrics.length,component_metric_rows:componentRows.length,whole_prediction_rows:wholePredictions.length,
 slot_prediction_rows_per_file:experiment.whole.no_encoder.length*2,
 unavailable_metric_rows:metrics.filter(r=>r.status==='unavailable').length,
 runtime:{node:process.version,...tools.versions},elapsed_seconds:(Date.now()-started)/1000,peak_memory_mib:process.resourceUsage().maxRSS/1024,
 outputs:walk(out).filter(f=>!f.endsWith('run_manifest.json')).map(f=>({path:path.relative(out,f),bytes:fs.statSync(f).size,sha256:sha256(fs.readFileSync(f))})),
};
fs.writeFileSync(path.join(out,'run_manifest.json'),JSON.stringify(manifest,null,2)+'\n');
console.log(JSON.stringify({status:'complete',output:out,checks:manifest.validation_checks,metric_rows:metrics.length,prediction_csv_files:7,figures:manifest.outputs.filter(f=>f.path.endsWith('.png')).length,seconds:manifest.elapsed_seconds,peak_memory_mib:manifest.peak_memory_mib}));
