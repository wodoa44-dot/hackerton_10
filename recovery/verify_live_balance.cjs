/* 서버에서 공개 평가 결과를 내려받아 행 단위로 독립 재검산합니다. */
const fs=require("fs"),path=require("path"),assert=require("assert/strict");
const ROOT="/data/main",BASE="https://devopen-gpu.jntp-ax.or.kr/i/OsAdz8rQR7JSXeW6";
const SLOTS=["classification","trait","part_number","case","packing","vendor"];
function csv(text){
 const rows=[];let row=[],v="",quoted=false;
 for(let i=0;i<text.length;i++){
  const c=text[i];
  if(quoted){if(c==='"'){if(text[i+1]==='"'){v+='"';i++;}else quoted=false;}else v+=c;}
  else if(c==='"')quoted=true;
  else if(c===','){row.push(v);v="";}
  else if(c==='\n'){row.push(v.endsWith("\r")?v.slice(0,-1):v);rows.push(row);row=[];v="";}
  else v+=c;
 }
 if(v||row.length){row.push(v);rows.push(row);}
 const header=rows.shift();header[0]=header[0].replace(/^\uFEFF/,"");
 return rows.filter(r=>r.length>1).map(r=>Object.fromEntries(header.map((k,i)=>[k,r[i]??""])));
}
const yes=v=>v==="True"||v==="true"||v==="1";
const designated=v=>v!=null&&String(v).replaceAll("-","").trim().length>0;
const eq=(a,b)=>assert.ok((a==null&&b==null)||Math.abs(a-b)<1e-10, String(a)+" != "+String(b));
function f1(rows,predKey){
 const labels=new Set(rows.flatMap(r=>[r.truth,r[predKey]??""]));if(!rows.length)return null;
 let total=0;
 for(const label of labels){
  let tp=0,fp=0,fn=0;
  for(const r of rows){const t=r.truth===label,p=(r[predKey]??"")===label;if(t&&p)tp++;else if(p)fp++;else if(t)fn++;}
  total+=(2*tp+fp+fn)?2*tp/(2*tp+fp+fn):0;
 }return total/labels.size;
}
async function fetchText(url){const r=await fetch(url);assert.equal(r.status,200,url);return r.text();}
(async()=>{
 assert.equal(require("os").hostname(),"proxy-ssh-server-5d5b67c46b-jhc85");
 const out=path.join(ROOT,"artifacts/balance_comparison");fs.mkdirSync(out,{recursive:true});
 const comparison=JSON.parse(await fetchText(BASE+"/comparison-downloads/comparison.json"));
 assert.equal(comparison.comparison_kind,"class_balance");
 const proof={checked_at:new Date().toISOString(),server:require("os").hostname(),source_url:BASE,checks:0,slots:[],note:"학습 모델은 복사하지 않고 평가 근거만 보관했습니다."};
 for(const [variant,report] of Object.entries(comparison.variants)){
  assert.equal(report.evaluation_rows,2850);
  fs.mkdirSync(path.join(out,variant),{recursive:true});
  for(const slot of SLOTS){
   const filename=slot+"_predictions.csv";
   const text=await fetchText(BASE+"/comparison-downloads/"+variant+"/"+filename);
   fs.writeFileSync(path.join(out,variant,filename),text);
   const all=csv(text);assert.equal(all.length,2850);
   for(const r of all)assert.equal(yes(r.designated),yes(r.eligible)&&designated(r.truth));
   for(const population of ["all","designated"]){
    const rows=all.filter(r=>yes(r.eligible)&&(population==="all"||designated(r.truth)));
    const target=population==="all"?report:report.designated;
    const raw=target.raw_heads[slot],parts=target.slots[slot];
    assert.equal(rows.length,raw.n);assert.equal(rows.length,parts.n);proof.checks+=2;
    eq(raw.accuracy,rows.length?rows.filter(r=>r.truth===r.raw_model_value).length/rows.length:null);
    eq(parts.accuracy,rows.length?rows.filter(r=>r.truth===r.displayed_value).length/rows.length:null);
    eq(raw.macro_f1,f1(rows,"raw_model_value"));eq(parts.macro_f1,f1(rows,"displayed_value"));proof.checks+=4;
    for(const [prefix,listKey,metric] of [
     ["raw","raw_model_candidates_top10",raw.at_k],
     ["joint","joint_slot_values_top10",parts.joint_at_k],
     ["marginal","marginal_candidates_top10",parts.marginal_at_k]]){
      for(let k=1;k<=10;k++){
       let correct=0;
       for(const r of rows){
        const values=JSON.parse(r[listKey]);
        const hit=values.slice(0,k).some(v=>v!=null&&String(v)===r.truth);
        assert.equal(yes(r[prefix+"_hit_at_"+k]),hit);
        if(hit)correct++;proof.checks++;
       }
       assert.equal(metric[String(k)].n,rows.length);
       assert.equal(metric[String(k)].correct,correct);
       eq(metric[String(k)].rate,rows.length?correct/rows.length:null);proof.checks+=3;
      }
    }
    proof.slots.push({variant,slot,population,n:rows.length,raw_top1:raw.accuracy,raw_top5:raw.at_k["5"].rate});
   }
  }
 }
 for(const f of ["comparison.json","candidate_k_curve.csv","top_k_comparison.csv","six_slot_comparison.csv","designated_six_slot_comparison.csv","paired_predictions.csv"]){
  const text=await fetchText(BASE+"/comparison-downloads/"+f);fs.writeFileSync(path.join(out,f),text);
 }
 proof.status="passed";fs.writeFileSync(path.join(out,"independent_server_verification.json"),JSON.stringify(proof,null,2)+"\n");
 console.log(JSON.stringify({status:proof.status,checks:proof.checks,slot_populations:proof.slots.length,saved:out}));
})().catch(e=>{console.error(e);process.exitCode=1});
