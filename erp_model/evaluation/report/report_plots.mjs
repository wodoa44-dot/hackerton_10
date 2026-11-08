import fs from 'node:fs';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import {SLOTS,LABELS,MODELS,POPULATIONS,VIEWS} from './report_data.mjs';

const here=path.dirname(fileURLToPath(import.meta.url));
const local=path.join(here,'node_modules');
const runtime=process.env.ERP_REPORT_RUNTIME?path.join(process.env.ERP_REPORT_RUNTIME,'node_modules'):fs.existsSync(local)?local:'/data/tools/report-runtime/node_modules';
export async function renderer(){
  const vega=await import(path.join(runtime,'vega/build/vega.module.js'));
  const {Resvg}=await import(path.join(runtime,'@resvg/resvg-js/index.js'));
  const font=path.join(here,'fonts/NanumGothic-Regular.ttf');
  return {vega,Resvg,font,versions:{vega:JSON.parse(fs.readFileSync(path.join(runtime,'vega/package.json'))).version,resvg:JSON.parse(fs.readFileSync(path.join(runtime,'@resvg/resvg-js/package.json'))).version}};
}
const FONT='NanumGothic';
const text=(x,y,value,size=14,weight='normal',color='#34455e')=>({type:'text',encode:{enter:{x:{value:x},y:{value:y},text:{value},font:{value:FONT},fontSize:{value:size},fontWeight:{value:weight},fill:{value:color}}}});
function legend(delta){
  if(delta)return [text(80,84,'보라색: 문자 CNN 사용 - 미사용 · 양수는 개선, 음수는 하락',15,'normal','#7c3aed')];
  return [
    {type:'rule',encode:{enter:{x:{value:80},x2:{value:113},y:{value:79},stroke:{value:'#2563eb'},strokeWidth:{value:3}}}},
    text(123,84,MODELS.no_encoder,15),
    {type:'rule',encode:{enter:{x:{value:505},x2:{value:538},y:{value:79},stroke:{value:'#e66b18'},strokeWidth:{value:3},strokeDash:{value:[6,3]}}}},
    text(548,84,MODELS.encoder,15),
  ];
}
function panel({rows,title,n,x,y,width,height,delta=false,domain=[0,1]}){
  const values=rows.filter(r=>r.rate!=null).map(r=>({...r,variant_label:MODELS[r.variant]}));
  return {type:'group',encode:{enter:{x:{value:x},y:{value:y},width:{value:width},height:{value:height}}},
    data:[{name:'points',values}],
    scales:[
      {name:'x',type:'linear',domain:[1,10],range:[0,width],zero:false},
      {name:'y',type:'linear',domain,range:[height,0],zero:!delta},
      {name:'color',type:'ordinal',domain:['no_encoder','encoder'],range:['#2563eb','#e66b18']},
    ],
    axes:[
      {orient:'bottom',scale:'x',values:[1,2,3,4,5,6,7,8,9,10],title:'Top-K (K)',labelFont:FONT,titleFont:FONT,labelFontSize:12,titleFontSize:13,domainColor:'#b4c1d4',tickColor:'#b4c1d4',titlePadding:10},
      {orient:'left',scale:'y',format:delta?undefined:'.0%',encode:delta?{labels:{update:{text:{signal:"(datum.value < 0 ? '-' : datum.value > 0 ? '+' : '') + format(abs(datum.value) * 100, '.0f') + '%p'"}}}}:undefined,tickCount:6,grid:true,gridScale:'x',gridColor:'#e5eaf2',domain:false,ticks:false,labelFont:FONT,titleFont:FONT,labelFontSize:12,titleFontSize:13,title:delta?'정답률 차이 (%p)':'정답률',titlePadding:9},
    ],
    marks:[
      text(0,-36,title,19,'bold'),text(0,-15,'평가 N = '+n.toLocaleString('en-US'),13),
      ...(delta?[{type:'rule',encode:{enter:{x:{value:0},x2:{value:width},y:{scale:'y',value:0},stroke:{value:'#8795aa'},strokeDash:{value:[4,3]}}}}]:[]),
      {type:'group',from:{facet:{name:'series',data:'points',groupby:'variant'}},marks:[
        {type:'line',from:{data:'series'},sort:{field:'k'},encode:{update:{x:{scale:'x',field:'k'},y:{scale:'y',field:'rate'},stroke:delta?{value:'#7c3aed'}:{scale:'color',field:'variant'},strokeWidth:{value:2.5},strokeDash:{signal:"datum.variant === 'encoder' && "+String(!delta)+" ? [6,3] : [1,0]"}}}},
      ]},
      {type:'symbol',from:{data:'points'},encode:{enter:{x:{scale:'x',field:'k'},y:{scale:'y',field:'rate'},shape:{signal:"datum.variant === 'encoder' ? 'square' : 'circle'"},size:{value:42},fill:{value:'white'},stroke:delta?{value:'#7c3aed'}:{scale:'color',field:'variant'},strokeWidth:{value:1.8}}}},
      ...(values.length?[]:[text(width/2-85,height/2,'행별 후보 기록 부족',16,'normal','#8a6070')]),
    ],
  };
}
export function slotSpec(metrics,view,population,{slot=null,delta=false}={}){
  const slots=slot?[slot]:SLOTS;
  let data=metrics.filter(r=>r.view===view&&r.population===population);
  let domain=[0,1];
  if(delta){
    data=SLOTS.flatMap(s=>Array.from({length:10},(_,i)=>{
      const a=data.find(r=>r.slot===s&&r.variant==='no_encoder'&&r.k===i+1),b=data.find(r=>r.slot===s&&r.variant==='encoder'&&r.k===i+1);
      return {slot:s,k:i+1,n:a?.n??0,variant:'encoder',rate:a?.rate!=null&&b?.rate!=null?b.rate-a.rate:null};
    }));
    const limit=Math.max(.05,Math.ceil(Math.max(...data.map(r=>Math.abs(r.rate??0)))*20)/20);domain=[-limit,limit];
  }
  const single=slots.length===1,width=single?760:1260,height=single?530:1160;
  const marks=[text(65,40,(delta?'인코더 추가 효과 · ':VIEWS[view]+' · ')+POPULATIONS[population],25,'bold','#172a45'),...legend(delta)];
  for(const [i,s] of slots.entries()){
    const rows=data.filter(r=>r.slot===s),n=rows[0]?.n??0;
    marks.push(panel({rows,title:LABELS[s],n,x:80+(single?0:i%2*620),y:165+(single?0:Math.floor(i/2)*320),width:single?580:480,height:single?245:215,delta,domain}));
  }
  marks.push(text(65,height-35,'두 모델은 동일한 규칙 검증·학습 마스크를 사용합니다. 후보에 정답이 없으면 K가 커져도 오답입니다.',12));
  marks.push(text(65,height-15,view==='joint'&&population==='designated'?'미지정 제외로 N이 달라지는 항목은 K=1·3·5·10을 재계산하며, N이 같으면 기존 집계를 사용합니다.':'미지정 제외: 정답이 -, -- 등인 행만 제외하며, 지정 정답에 대한 미지정 예측은 오답입니다.',12));
  return {$schema:'https://vega.github.io/schema/vega/v6.json',width,height,padding:0,background:'white',autosize:'none',marks};
}
export function wholeSpec(rows){
  const groups=[['all','전체'],['manual','매뉴얼 해석 가능'],['unmapped','매뉴얼 미해석']];
  return {$schema:'https://vega.github.io/schema/vega/v6.json',width:1260,height:530,padding:0,background:'white',autosize:'none',
    marks:[text(65,40,'완성 품목코드 Top-K · ERP 정답 코드와 완전일치',25,'bold','#172a45'),...legend(false),
      ...groups.map(([id,title],i)=>panel({rows:rows.filter(r=>r.group===id),title,n:rows.find(r=>r.group===id)?.n??0,x:80+i*410,y:180,width:305,height:225})),
      text(65,490,'전체 N=2,850 · 매뉴얼 해석 가능 N=473 · 미해석 N=2,377. 상세 칸과 완성 코드의 분모가 다릅니다.',13),
      text(65,514,'기존 진단 데이터의 저장된 비교 결과이며, 신규 블라인드 시험 또는 현재 v8 모델 성능을 뜻하지 않습니다.',12)]};
}
export function tableSpec(metrics,population){
  const columns=[['항목',35],['특징 인코더',185],['N',425],['Top-1',540],['Top-3',690],['Top-5',840],['Top-10',990]];
  const marks=[text(30,40,'원시 칸 모델 · '+POPULATIONS[population]+' · Top-K 정답률',24,'bold','#172a45')];
  marks.push({type:'rect',encode:{enter:{x:{value:20},y:{value:65},width:{value:1120},height:{value:40},fill:{value:'#eaf0f9'}}}});
  for(const [label,x] of columns)marks.push(text(x,91,label,16,'bold'));
  for(const [si,s] of SLOTS.entries())for(const [vi,v] of ['no_encoder','encoder'].entries()){
    const y=109+(si*2+vi)*40;
    marks.push({type:'rect',encode:{enter:{x:{value:20},y:{value:y},width:{value:1120},height:{value:39},fill:{value:si%2?'#f4f7fb':'#ffffff'}}}});
    const values=[vi?'':LABELS[s],vi?'문자 CNN (v7)':'미사용 (v6)',String(metrics.find(r=>r.view==='raw'&&r.population===population&&r.slot===s&&r.variant===v)?.n??0),
      ...[1,3,5,10].map(k=>{const r=metrics.find(r=>r.view==='raw'&&r.population===population&&r.slot===s&&r.variant===v&&r.k===k);return r?.rate==null?'—':(r.rate*100).toFixed(2)+'%';})];
    values.forEach((value,i)=>marks.push(text(columns[i][1],y+26,value,15,vi?'bold':'normal',i===1?(vi?'#bc4d08':'#1e54b8'):'#34455e')));
  }
  marks.push(text(30,622,'특징 인코더 미사용 모델에도 규칙 검증·전처리는 적용되어 있습니다. 수치는 같은 행 집합에서 비교했습니다.',12));
  return {$schema:'https://vega.github.io/schema/vega/v6.json',width:1160,height:645,padding:0,background:'white',autosize:'none',marks};
}
export async function saveFigure(tools,spec,base){
  fs.mkdirSync(path.dirname(base),{recursive:true});
  const view=new tools.vega.View(tools.vega.parse(spec),{renderer:'none'});
  try{
    const svg=await view.toSVG();
    fs.writeFileSync(base+'.svg',svg);
    const png=new tools.Resvg(svg,{background:'white',font:{fontFiles:[tools.font],loadSystemFonts:false,defaultFontFamily:FONT}}).render().asPng();
    fs.writeFileSync(base+'.png',png);
    fs.writeFileSync(base+'.vega.json',JSON.stringify(spec,null,2)+'\n');
    return svg;
  }finally{view.finalize();}
}
