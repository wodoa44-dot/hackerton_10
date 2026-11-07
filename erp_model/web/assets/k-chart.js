'use strict';
window.renderKChart=function(metrics) {
  const panel=document.getElementById('k-curve-panel');
  if(!metrics?.groups){panel.hidden=true;return;}
  panel.hidden=false;
  const specs=[['all','전체 품목','#16848b'],['manual','매뉴얼 해석 가능','#365d87'],['unmapped','규칙 미확인 · 추정 포함','#b96b3f']];
  const active=new Set(specs.map(s=>s[0]));
  const host=document.getElementById('k-chart');
  const legend=document.getElementById('k-chart-legend');
  const fmt=v=>v==null?'—':(v*100).toFixed(2)+'%';
  const esc=v=>String(v).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const point=(name,k)=>metrics.groups[name].points.find(p=>p.k===k);
  function detail(k) {
    const items=specs.filter(s=>active.has(s[0])).map(([id,label])=>{
      const p=point(id,k);return label+' '+fmt(p.rate)+' ('+p.correct+'/'+p.n+')';
    });
    document.getElementById('k-chart-detail').textContent='K='+k+' · '+items.join(' · ');
    const line=host.querySelector('[data-crosshair]');
    if(line){const circle=host.querySelector('[data-k="'+k+'"]');if(circle){line.setAttribute('x1',circle.getAttribute('cx'));line.setAttribute('x2',circle.getAttribute('cx'));line.style.display='';}}
  }
  function draw() {
    const w=Math.max(280,Math.round(host.clientWidth || 760)),h=330,l=53,r=22,t=28,b=54;
    const x=k=>l+(k-1)*(w-l-r)/9,y=v=>t+(1-v)*(h-t-b);
    let svg='<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 '+w+' '+h+'" width="100%" role="img" aria-labelledby="k-chart-title k-chart-description" font-family="sans-serif"><title id="k-chart-title">후보 개수 K에 따른 정답 포함률</title><desc id="k-chart-description">가로축은 후보 수 1부터 10, 세로축은 원본 정답 코드가 후보에 포함되는 행의 비율입니다. 그룹별 분모는 고정됩니다.</desc>';
    for(let v=0;v<=100;v+=20) {
      const yy=y(v/100);
      svg+='<line x1="'+l+'" x2="'+(w-r)+'" y1="'+yy+'" y2="'+yy+'" stroke="#e0e8ec" stroke-width="1"/><text x="'+(l-9)+'" y="'+(yy+4)+'" text-anchor="end" font-size="12" fill="#688195">'+v+'%</text>';
    }
    svg+='<text x="'+l+'" y="15" font-size="12" fill="#476475">정답 포함률 (%)</text>';
    for(let k=1;k<=10;k++)svg+='<text x="'+x(k)+'" y="'+(h-b+23)+'" text-anchor="middle" font-size="12" fill="#688195">'+k+'</text>';
    svg+='<text x="'+((l+w-r)/2)+'" y="'+(h-8)+'" text-anchor="middle" font-size="13" fill="#476475">K · 확인할 후보 개수</text>';
    svg+='<line data-crosshair x1="0" x2="0" y1="'+t+'" y2="'+(h-b)+'" stroke="#94aab5" stroke-width="1" stroke-dasharray="3 4" style="display:none" pointer-events="none"/>';
    for(const [id,label,color] of specs) {
      if(!active.has(id))continue;
      const values=metrics.groups[id].points.filter(p=>p.rate!=null);
      const d=values.map((p,i)=>(i?'L':'M')+x(p.k)+' '+y(p.rate)).join(' ');
      svg+='<path data-curve="'+id+'" d="'+d+'" fill="none" stroke="'+color+'" stroke-width="2.5"/>';
      for(const p of values)svg+='<circle data-series="'+id+'" data-k="'+p.k+'" cx="'+x(p.k)+'" cy="'+y(p.rate)+'" r="5" fill="'+color+'" stroke="white" stroke-width="1"><title>'+esc(label+' K='+p.k+' '+fmt(p.rate)+' ('+p.correct+'/'+p.n+')')+'</title></circle>';
    }
    svg+='</svg>';host.innerHTML=svg;
    host.querySelectorAll('[data-k]').forEach(c=>{c.addEventListener('mouseenter',()=>detail(Number(c.dataset.k)));c.addEventListener('click',()=>detail(Number(c.dataset.k)));});
    detail(5);
  }
  legend.innerHTML=specs.map(([id,label,color])=>'<button type="button" class="k-toggle" data-k-series="'+id+'" aria-pressed="true"><span class="k-swatch" style="background:'+color+'"></span>'+label+' <small>n='+metrics.groups[id].n.toLocaleString()+' · K10 '+fmt(point(id,10).rate)+'</small></button>').join('');
  legend.querySelectorAll('[data-k-series]').forEach(button=>button.addEventListener('click',()=>{
    const key=button.dataset.kSeries;
    if(active.has(key)){if(active.size===1)return;active.delete(key);}else active.add(key);
    button.setAttribute('aria-pressed',String(active.has(key)));draw();
  }));
  document.getElementById('k-chart-table').innerHTML=Array.from({length:10},(_,i)=>{
    const k=i+1;return '<tr><td>'+k+'</td>'+specs.map(([id])=>{const p=point(id,k);return '<td>'+fmt(p.rate)+' <small>'+p.correct+' / '+p.n+'</small></td>';}).join('')+'<td>'+point('all',k).average_candidates.toFixed(2)+'</td></tr>';
  }).join('');
  document.getElementById('download-k-svg').onclick=()=>{
    const svg=host.querySelector('svg');if(!svg)return;
    const url=URL.createObjectURL(new Blob([svg.outerHTML],{type:'image/svg+xml;charset=utf-8'}));
    const a=document.createElement('a');a.href=url;a.download='후보개수별_정답포함률.svg';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
  };
  draw();
  if(host._kObserver)host._kObserver.disconnect();
  host._kObserver=new ResizeObserver(()=>draw());host._kObserver.observe(host);
};
