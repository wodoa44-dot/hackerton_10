import fs from 'node:fs';
import path from 'node:path';
import assert from 'node:assert/strict';
import crypto from 'node:crypto';

export const SLOTS = ['classification', 'trait', 'part_number', 'case', 'packing', 'vendor'];
export const LABELS = {classification:'분류',trait:'특성',part_number:'세분류·품번',case:'형상',packing:'Packing',vendor:'Vendor'};
export const VARIANTS = ['no_encoder', 'encoder'];
export const MODELS = {no_encoder:'특징 인코더 미사용 (v6)', encoder:'문자 CNN 특징 사용 (v7)'};
export const VIEWS = {raw:'원시 칸 모델',joint:'완성 코드 후보에서 해석',marginal:'규칙 반영 칸별 후보'};
export const POPULATIONS = {all:'미지정 포함',designated:'미지정 정답 제외'};
export const KS = Array.from({length:10}, (_,i)=>i+1);
export const DISPLAY_K = [1,3,5,10];
export const truthy = x => x === true || /^(true|1)$/i.test(String(x));
export const placeholder = x => x == null || String(x).replaceAll('-','').trim() === '';
export const sha256 = x => crypto.createHash('sha256').update(x).digest('hex');
export const equalRate = (a,b) => assert.ok(a == null && b == null || Number.isFinite(a) && Number.isFinite(b) && Math.abs(a-b)<1e-10, String(a)+' != '+String(b));
export const json = filename => JSON.parse(fs.readFileSync(filename,'utf8'));

export function parseCSV(text) {
  const records=[];let row=[],value='',quoted=false;
  for(let i=0;i<text.length;i++){
    const c=text[i];
    if(quoted){if(c==='"'){if(text[i+1]==='"'){value+='"';i++;}else quoted=false;}else value+=c;}
    else if(c==='"'){assert.equal(value,'','CSV 따옴표 위치 오류');quoted=true;}
    else if(c===','){row.push(value);value='';}
    else if(c==='\n'){row.push(value.endsWith('\r')?value.slice(0,-1):value);records.push(row);row=[];value='';}
    else value+=c;
  }
  assert.equal(quoted,false,'닫히지 않은 CSV 따옴표');
  if(value||row.length){row.push(value);records.push(row);}
  const header=records.shift()??[];if(header.length)header[0]=header[0].replace(/^\uFEFF/,'');
  assert.equal(new Set(header).size,header.length,'중복 CSV 열');
  return records.filter(r=>r.length>1||r[0]!=='').map(r=>{
    assert.equal(r.length,header.length,'CSV 열 수 불일치');
    return Object.fromEntries(header.map((key,i)=>[key,r[i]]));
  });
}
export function saveCSV(filename,rows,columns) {
  fs.mkdirSync(path.dirname(filename),{recursive:true});
  const keys=columns??Object.keys(rows[0]??{});
  const cell=x=>{const s=x==null?'':typeof x==='object'?JSON.stringify(x):String(x);return /[",\r\n]/.test(s)?'"'+s.replaceAll('"','""')+'"':s;};
  const text='\uFEFF'+keys.map(cell).join(',')+'\r\n'+rows.map(r=>keys.map(k=>cell(r[k])).join(',')).join('\r\n')+'\r\n';
  fs.writeFileSync(filename,text);
  assert.equal(parseCSV(text).length,rows.length,'CSV 재읽기 행 수 불일치');
}
export function macroF1(rows,prediction) {
  if(!rows.length)return null;
  const counts=new Map();
  const get=k=>{if(!counts.has(k))counts.set(k,{tp:0,fp:0,fn:0});return counts.get(k);};
  for(const row of rows){const t=String(row.truth),p=String(prediction(row)??'');get(t);get(p);if(t===p)get(t).tp++;else{get(t).fn++;get(p).fp++;}}
  return [...counts.values()].reduce((sum,c)=>sum+(2*c.tp+c.fp+c.fn?2*c.tp/(2*c.tp+c.fp+c.fn):0),0)/counts.size;
}
export const hit = (candidates,truth,k) => candidates.slice(0,k).some(v=>v!=null&&String(v)===String(truth));

export function loadExperiment(input) {
  const comparison=json(path.join(input,'comparison.json'));
  assert.deepEqual(Object.keys(comparison.variants).sort(),[...VARIANTS].sort(),'인코더 유무 비교 자료만 사용합니다.');
  const files=[],read=relative=>{const filename=path.join(input,relative),bytes=fs.readFileSync(filename);files.push({path:relative,bytes:bytes.length,sha256:sha256(bytes)});return parseCSV(bytes.toString('utf8'));};
  files.push({path:'comparison.json',sha256:sha256(fs.readFileSync(path.join(input,'comparison.json')))});
  const whole={},parts={},partIndex={};
  let checks=0;
  for(const variant of VARIANTS){
    const report=comparison.variants[variant];
    whole[variant]=read(variant+'/prediction_parts.csv');
    assert.equal(whole[variant].length,report.evaluation_rows);
    assert.equal(new Set(whole[variant].map(r=>r.source_row)).size,report.evaluation_rows);
    for(const row of whole[variant]){
      row.candidates=JSON.parse(row.candidate_codes);
      assert.equal(new Set(row.candidates).size,row.candidates.length,'완성 코드 후보 중복');
      assert.ok(row.candidates.length<=10);
      for(const k of KS){assert.equal(truthy(row['hit_at_'+k]),hit(row.candidates,row.truth,k));checks++;}
    }
    for(const [group,data] of Object.entries(report.candidate_k_metrics.groups)){
      const subset=whole[variant].filter(r=>group==='all'||truthy(r.manual_parseable)===(group==='manual'));
      for(const point of data.points){
        const correct=subset.filter(r=>hit(r.candidates,r.truth,point.k)).length;
        assert.equal(point.n,subset.length);assert.equal(point.correct,correct);equalRate(point.rate,subset.length?correct/subset.length:null);checks+=3;
      }
    }
    parts[variant]=read(variant+'/six_slot_predictions.csv');
    partIndex[variant]=new Map();
    assert.equal(parts[variant].length,whole[variant].length*6);
    for(const row of parts[variant]){
      assert.ok(SLOTS.includes(row.slot));
      const key=row.source_row+'|'+row.slot;
      assert.equal(partIndex[variant].has(key),false,'중복된 항목 행');
      row.raw=JSON.parse(row.raw_model_candidates_top10);
      assert.ok(Array.isArray(row.raw));assert.ok(row.raw.length<=10);
      assert.equal(String(row.raw[0]??''),row.raw_model_value);
      assert.equal(truthy(row.placeholder),truthy(row.eligible)&&placeholder(row.truth));
      partIndex[variant].set(key,row);
    }
    for(const slot of SLOTS){
      const rows=parts[variant].filter(r=>r.slot===slot&&truthy(r.eligible)),raw=report.raw_heads[slot],final=report.slots[slot];
      assert.equal(rows.length,raw.n);assert.equal(rows.length,final.n);
      equalRate(macroF1(rows,r=>r.raw_model_value),raw.macro_f1);
      equalRate(macroF1(rows,r=>r.displayed_value),final.macro_f1);
      equalRate(rows.length?rows.filter(r=>r.truth===r.displayed_value).length/rows.length:null,final.accuracy);checks+=5;
      for(const k of KS){
        const correct=rows.filter(r=>hit(r.raw,r.truth,k)).length,point=raw.at_k[String(k)];
        assert.equal(point.n,rows.length);assert.equal(point.correct,correct);equalRate(point.rate,rows.length?correct/rows.length:null);checks+=3;
        if(Object.hasOwn(rows[0]??{},'joint_hit_at_'+k)){
          assert.equal(rows.filter(r=>truthy(r['joint_hit_at_'+k])).length,final.joint_at_k[String(k)].correct);checks++;
        }
      }
    }
  }
  const mapB=new Map(whole.encoder.map(r=>[r.source_row,r]));
  for(const row of whole.no_encoder){
    const b=mapB.get(row.source_row);assert.ok(b);
    for(const key of ['품목명','규격명','truth','manual_parseable'])assert.equal(row[key],b[key]);checks+=4;
  }
  for(const [key,a] of partIndex.no_encoder){
    const b=partIndex.encoder.get(key);assert.ok(b);
    for(const field of ['truth','eligible','placeholder','품목명','규격명'])assert.equal(a[field],b[field]);checks+=5;
  }
  return {comparison,whole,parts,partIndex,files,checks};
}
export function buildMetrics(experiment) {
  const {comparison,parts}=experiment,metrics=[];
  for(const view of Object.keys(VIEWS))for(const population of Object.keys(POPULATIONS))for(const slot of SLOTS)for(const variant of VARIANTS){
    const eligible=parts[variant].filter(r=>r.slot===slot&&truthy(r.eligible));
    const rows=eligible.filter(r=>population==='all'||!placeholder(r.truth));
    const v=comparison.variants[variant],samePopulation=rows.length===eligible.length;
    for(const k of KS){
      let correct=null,rate=null,basis='행별 후보 기록 부족';
      if(view==='raw'){correct=rows.filter(r=>hit(r.raw,r.truth,k)).length;basis='행별 Top-10 후보 재계산';}
      else if(view==='joint'&&rows.every(r=>Object.hasOwn(r,'joint_hit_at_'+k))){correct=rows.filter(r=>truthy(r['joint_hit_at_'+k])).length;basis='행별 완성 코드 칸 적중 기록';}
      else if(samePopulation){
        const point=view==='joint'?v.slots[slot].joint_at_k[String(k)]:v.slots[slot].marginal_at_k[String(k)];
        assert.equal(point.n,rows.length);correct=point.correct;basis='저장된 동일 모집단 집계';
      }
      if(correct!=null)rate=rows.length?correct/rows.length:null;
      metrics.push({view,population,slot,variant,k,n:rows.length,all_n:eligible.length,excluded_n:eligible.length-rows.length,correct,rate,basis,status:correct==null?'unavailable':rows.length?'available':'empty',macro_f1:view==='raw'&&k===1?macroF1(rows,r=>r.raw_model_value):null});
    }
  }
  return metrics;
}
