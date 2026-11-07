'use strict';
// Every rate below comes from the measured, paired comparison report.
window.renderEncoderComparison = function(info) {
  const page = document.getElementById('page-evaluation');
  let panel = document.getElementById('encoder-comparison');
  const esc = value => String(value ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const fmt = value => value == null ? '—' : (value * 100).toFixed(2) + '%';
  const count = value => Number(value ?? 0).toLocaleString('ko-KR');
  const delta = (a,b) => a == null || b == null ? '—' : ((b-a)*100 > 0 ? '+' : '') + ((b-a)*100).toFixed(2) + ' %p';
  const series = [['no_encoder','학습 인코더 없음','#526a88'],['encoder','학습 인코더 사용','#087f78']];
  const slotLabels = [['classification','분류'],['trait','특성'],['part_number','세분류 · 품번'],['case','형상'],['packing','Packing'],['vendor','Vendor']];
  const scopes = {all:'전체 품목',manual:'매뉴얼 해석 가능',unmapped:'구간 규칙 미확인'};
  const report = info.encoder_comparison;

  if (!document.getElementById('model-variant')) {
    const anchor = document.getElementById('candidate-limit');
    const label = document.createElement('label');
    label.htmlFor = 'model-variant'; label.className = 'encoder-input-label'; label.textContent = '단일 예측 모델';
    const select = document.createElement('select'); select.id = 'model-variant';
    anchor.after(label,select);
  }
  const variantSelect = document.getElementById('model-variant');
  const priorVariant = variantSelect.value;
  variantSelect.innerHTML = '<option value="">현재 운영 모델</option>' + series.filter(([id]) => info.model_variants?.[id]?.available).map(([id,label]) => '<option value="'+id+'">'+label+'</option>').join('');
  if ([...variantSelect.options].some(option => option.value === priorVariant)) variantSelect.value = priorVariant;
  const lookup = document.getElementById('lookup-policy');
  const batchNote = document.getElementById('batch-variant-note') || document.createElement('p');
  batchNote.id = 'batch-variant-note'; batchNote.className = 'muted';
  batchNote.textContent = 'CSV 일괄 채움 모델: ' + (info.active_model_variant === 'encoder' ? '학습 인코더 사용' : '학습 인코더 없음') + '. 단일 예측 모델 선택과 별도로 현재 운영 모델을 사용합니다.';
  lookup.after(batchNote);

  if (!panel) {
    panel = document.createElement('section'); panel.id = 'encoder-comparison'; panel.className = 'encoder-comparison';
    page.querySelector('.page-heading').after(panel);
  }
  for (const child of page.children) if (child !== panel && !child.classList.contains('page-heading')) child.hidden = true;
  if (!report?.variants?.no_encoder || !report?.variants?.encoder) {
    panel.innerHTML = '<div class="panel table-panel"><h2>학습 인코더 유무 · Top-K 비교</h2><p class="muted">두 방식의 학습과 동일 평가가 끝나면 이곳에 실제 수치가 표시됩니다. 결과가 준비되기 전에는 정확도를 표시하지 않습니다.</p></div>';
    return;
  }
  page.querySelector('.page-heading p').textContent = '분류·특성·세분류/품번·형상·Packing·Vendor의 독립 결과를 각각 비교합니다. 각 칸의 K와 평가 건수를 따로 확인하세요.';
  const badge = page.querySelector('.page-heading .outline-badge');
  if (badge) badge.textContent = '동일 조건 · 실제 비교 평가';
  panel.innerHTML = `
    <section class="panel comparison-intro">
      <div class="context-heading"><div><div class="eyebrow">SIX OUTPUTS · ENCODER A/B</div><h2>6개 코드의 결과를 각각 확인하세요.</h2></div><span class="outline-badge">칸별 독립 Top-K</span></div>
      <p>공통 전처리 → TF-IDF·구조화 특징 → XGBoost·LightGBM·Forest → 코드 규칙과 후보 조합. 인코더 사용 방식은 여기에 <strong>직접 학습한 문자 CNN 특징</strong>을 추가합니다. 칸별 모델 선택·가중치와 조회·규칙은 동일하게 유지했습니다.</p>
    </section>
    <section class="slot-results-section" id="encoder-slot-section">
      <div class="context-heading"><h2>6개 코드별 독립 평가 결과</h2><label class="comparison-field-label">평가 기준<select id="encoder-slot-mode"><option value="raw">각 칸 학습 모델의 후보</option><option value="part">규칙을 반영한 각 칸의 후보</option><option value="joint">완성 코드에서 해석한 각 칸</option></select></label></div>
      <p id="encoder-slot-note" class="muted"></p>
      <div class="comparison-legend">${series.map(([,label,color])=>'<span><i style="background:'+color+'"></i>'+label+'</span>').join('')}</div>
      <div class="independent-slot-grid">${slotLabels.map(([slot,label],index)=>`
        <article class="panel independent-slot-panel" data-slot-panel="${slot}" aria-labelledby="encoder-slot-title-${slot}">
          <div class="context-heading"><div><span class="slot-number">0${index+1}</span><h3 id="encoder-slot-title-${slot}">${label}</h3></div><label class="slot-k-label" for="encoder-slot-k-${slot}">이 칸의 K<select id="encoder-slot-k-${slot}" data-slot-k-select="${slot}">${Array.from({length:10},(_,i)=>'<option value="'+(i+1)+'"'+(i===4?' selected':'')+'>Top-'+(i+1)+'</option>').join('')}</select></label></div>
          <p class="slot-scope muted" id="encoder-slot-scope-${slot}"></p>
          <div class="slot-score-cards" id="encoder-slot-scores-${slot}"></div>
          <p class="slot-macro muted" id="encoder-slot-macro-${slot}"></p>
          <div class="slot-k-chart" id="encoder-slot-chart-${slot}" data-slot-chart="${slot}"></div>
          <p class="slot-chart-detail" id="encoder-slot-detail-${slot}" role="status" aria-live="polite"></p>
          <p class="comparison-table-hint muted">표를 좌우로 밀어 비교하세요.</p>
          <div class="table-scroll"><table><thead><tr><th>K</th><th>인코더 없음</th><th>인코더 사용</th><th>차이</th></tr></thead><tbody id="encoder-slot-table-${slot}"></tbody></table></div>
          <div class="slot-downloads"><button type="button" class="batch-secondary" data-slot-svg="${slot}">이 칸 그래프 ↓</button><button type="button" class="batch-secondary" data-slot-csv="${slot}">이 칸 Top-K CSV ↓</button></div>
          <div class="slot-row-downloads"><a href="comparison-downloads/no_encoder/${slot}_predictions.csv" download>없음 · 행별 결과 ↓</a><a href="comparison-downloads/encoder/${slot}_predictions.csv" download>사용 · 행별 결과 ↓</a></div>
          ${slot==='part_number'?'<p class="muted slot-special-note">수치형 세분류도 평가에 포함됩니다. 학습 모델 후보 모드에서는 수치 계산 규칙을 적용하기 전의 품번 후보를 채점합니다.</p>':''}
        </article>`).join('')}</div>
    </section>
    <details class="whole-code-results" id="encoder-whole-results"><summary>완성 코드 조합 결과도 보기 <span>6개 칸을 합친 코드의 Top-K</span></summary><section class="panel table-panel">
      <div class="context-heading"><h2>완성 코드 · K에 따른 정답 포함률</h2><button id="encoder-chart-save" class="batch-secondary" type="button">그래프 저장 ↓</button></div>
      <p class="muted">Hit@K = 앞의 K개 완성 코드 안에 원본 정답 코드가 있는 행 수 ÷ 해당 범위의 전체 행 수. 코드가 나오지 않은 행도 분모에 포함합니다. K보다 후보가 적으면 실제 생성된 후보까지만 확인합니다.</p>
      <div class="comparison-controls"><label>완성 코드 평가 범위<select id="encoder-scope"><option value="all">전체 품목</option><option value="manual">매뉴얼 해석 가능</option><option value="unmapped">구간 규칙 미확인</option></select></label><label>완성 코드의 K<select id="encoder-k">${Array.from({length:10},(_,i)=>'<option value="'+(i+1)+'"'+(i===4?' selected':'')+'>Top-'+(i+1)+'</option>').join('')}</select></label></div>
      <p id="encoder-scope-note" class="muted"></p><div class="comparison-cards" id="encoder-summary"></div>
      <div id="encoder-legend" class="comparison-legend"></div><div id="encoder-chart" class="comparison-chart"></div><p id="encoder-chart-detail" class="k-chart-detail" role="status" aria-live="polite"></p>
      <div class="table-scroll"><table><thead><tr><th>후보 수 K</th><th>학습 인코더 없음</th><th>학습 인코더 사용</th><th>차이 · 사용 − 없음</th><th>평균 생성 후보 수 · 없음 / 사용</th></tr></thead><tbody id="encoder-k-table"></tbody></table></div>
    </section></details>
    <section class="panel table-panel">
      <h2>예측 결과 다운로드</h2><p class="muted">같은 행의 두 방식 결과와 K별 정답 여부를 직접 대조할 수 있습니다.</p>
      <div class="downloads"><a class="download-link" href="comparison-downloads/paired_predictions.csv" download>행별 A/B 결과 CSV ↓</a><a class="download-link" href="comparison-downloads/top_k_comparison.csv" download>Top-1·3·5·10 CSV ↓</a><a class="download-link" href="comparison-downloads/candidate_k_curve.csv" download>K=1~10 전체 수치 CSV ↓</a><a class="download-link" href="comparison-downloads/six_slot_comparison.csv" download>6개 칸 평가 CSV ↓</a></div>
      <div class="comparison-controls"><label>칸별 결과 파일<select id="encoder-part-variant"><option value="no_encoder">학습 인코더 없음</option><option value="encoder">학습 인코더 사용</option></select></label></div><div id="encoder-part-downloads" class="downloads"></div>
      <details class="comparison-protocol"><summary>비교 조건과 평가 범위</summary><div id="encoder-protocol"></div></details>
    </section>`;
  const $c = id => document.getElementById(id);
  const active = {scope:'all',k:5,mode:'raw',slotK:Object.fromEntries(slotLabels.map(([slot])=>[slot,5]))};
  const group = id => report.variants[id].candidate_k_metrics.groups[active.scope];
  const point = (id,k) => group(id).points.find(value => value.k === k) || {};
  function chart() {
    const host = $c('encoder-chart');
    const width = Math.max(300,Math.round(host.clientWidth || 780)), height = 340, left=54,right=24,top=30,bottom=52;
    const x = k => left+(k-1)*(width-left-right)/9, y = value => top+(1-value)*(height-top-bottom);
    let svg = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 '+width+' '+height+'" width="100%" role="img" aria-labelledby="encoder-chart-title encoder-chart-desc" font-family="sans-serif"><title id="encoder-chart-title">'+esc(scopes[active.scope])+' · 학습 인코더 유무별 Top-K 정답 포함률</title><desc id="encoder-chart-desc">가로축 K는 1부터 10. 세로축은 0부터 100퍼센트. 학습 인코더 없음과 사용을 같은 평가 행으로 비교합니다.</desc>';
    for (let v=0;v<=100;v+=20) svg += '<line x1="'+left+'" x2="'+(width-right)+'" y1="'+y(v/100)+'" y2="'+y(v/100)+'" stroke="#e0e8ec"/><text x="'+(left-9)+'" y="'+(y(v/100)+4)+'" text-anchor="end" font-size="12" fill="#607586">'+v+'%</text>';
    svg += '<text x="'+left+'" y="16" fill="#476475" font-size="12">완성 코드 정답 포함률</text>';
    for (let k=1;k<=10;k++) svg += '<text x="'+x(k)+'" y="'+(height-bottom+23)+'" text-anchor="middle" fill="#607586" font-size="12">'+k+'</text>';
    svg += '<text x="'+((left+width-right)/2)+'" y="'+(height-7)+'" text-anchor="middle" fill="#476475" font-size="12">K · 확인할 후보 개수</text><line x1="'+x(active.k)+'" x2="'+x(active.k)+'" y1="'+top+'" y2="'+(height-bottom)+'" stroke="#aac0ca" stroke-dasharray="4 4"/>';
    for (const [id,label,color] of series) {
      const points = group(id).points.filter(p=>p.rate != null);
      svg += '<path d="'+points.map((p,i)=>(i?'L':'M')+x(p.k)+' '+y(p.rate)).join(' ')+'" stroke="'+color+'" fill="none" stroke-width="3"'+(id==='no_encoder'?' stroke-dasharray="7 4"':'')+'/>';
      for (const p of points) svg += '<circle data-encoder-k="'+p.k+'" cx="'+x(p.k)+'" cy="'+y(p.rate)+'" r="'+(p.k===active.k?6:4)+'" fill="'+color+'" stroke="white" stroke-width="1.5" tabindex="0" role="button" aria-label="'+esc(label+' Top-'+p.k+' '+fmt(p.rate))+'"><title>'+esc(label+' · Top-'+p.k+': '+fmt(p.rate)+' ('+p.correct+'/'+p.n+')')+'</title></circle>';
    }
    svg += '</svg>';host.innerHTML=svg;
    host.querySelectorAll('[data-encoder-k]').forEach(el=>{
      const choose=()=>{active.k=Number(el.dataset.encoderK);$c('encoder-k').value=String(active.k);render();};
      el.addEventListener('click',choose);el.addEventListener('keydown',event=>{if(event.key==='Enter'||event.key===' '){event.preventDefault();choose();}});
    });
    $c('encoder-chart-detail').textContent = 'Top-'+active.k+' · '+series.map(([id,label])=>{const p=point(id,active.k);return label+' '+fmt(p.rate)+' ('+count(p.correct)+'/'+count(p.n)+'건)';}).join(' · ');
  }
  const modeLabels={raw:'각 칸 학습 모델의 후보',part:'규칙을 반영한 각 칸의 후보',joint:'완성 코드에서 해석한 각 칸'};
  function slotPoints(variant,slot) {
    const data=report.variants[variant];
    return active.mode==='raw' ? data.raw_heads?.[slot]?.at_k : data.slots[slot][active.mode==='part'?'marginal_at_k':'joint_at_k'];
  }
  const slotPoint=(variant,slot,k)=>slotPoints(variant,slot)?.[String(k)] || {};
  function drawSlot(slot) {
    const host=$c('encoder-slot-chart-'+slot),label=slotLabels.find(([key])=>key===slot)[1],selected=active.slotK[slot];
    const w=Math.max(260,Math.round(host.clientWidth || 480)),h=250,l=44,r=14,t=24,b=42;
    const x=k=>l+(k-1)*(w-l-r)/9,y=v=>t+(1-v)*(h-t-b);
    let svg='<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 '+w+' '+h+'" width="100%" role="img" aria-labelledby="slot-title-'+slot+' slot-desc-'+slot+'" font-family="sans-serif"><title id="slot-title-'+slot+'">'+esc(label+' · '+modeLabels[active.mode]+' · 인코더 유무 Top-K')+'</title><desc id="slot-desc-'+slot+'">이 칸의 K 1부터 10에 따른 정답 포함률. 인코더 없음은 파란 점선, 사용은 초록 실선입니다.</desc>';
    for(let v=0;v<=100;v+=20) svg+='<line x1="'+l+'" x2="'+(w-r)+'" y1="'+y(v/100)+'" y2="'+y(v/100)+'" stroke="#e0e8ec"/><text x="'+(l-7)+'" y="'+(y(v/100)+4)+'" text-anchor="end" font-size="10" fill="#607586">'+v+'%</text>';
    svg+='<text x="'+l+'" y="12" fill="#476475" font-size="10">'+esc(label)+' 정답 포함률</text>';
    for(let k=1;k<=10;k++)svg+='<text x="'+x(k)+'" y="'+(h-b+19)+'" text-anchor="middle" font-size="10" fill="#607586">'+k+'</text>';
    svg+='<text x="'+((l+w-r)/2)+'" y="'+(h-3)+'" text-anchor="middle" font-size="11" fill="#476475">K · 이 칸의 후보 수</text><line x1="'+x(selected)+'" x2="'+x(selected)+'" y1="'+t+'" y2="'+(h-b)+'" stroke="#aac0ca" stroke-dasharray="3 4"/>';
    for(const [variant,name,color] of series) {
      const points=Object.entries(slotPoints(variant,slot)||{}).map(([k,p])=>({...p,k:Number(k)})).filter(p=>p.rate!=null).sort((a,b)=>a.k-b.k);
      svg+='<path data-slot-curve="'+variant+'" d="'+points.map((p,i)=>(i?'L':'M')+x(p.k)+' '+y(p.rate)).join(' ')+'" fill="none" stroke="'+color+'" stroke-width="2.5"'+(variant==='no_encoder'?' stroke-dasharray="6 4"':'')+'/>';
      for(const p of points)svg+='<circle data-slot-k="'+p.k+'" data-variant="'+variant+'" cx="'+x(p.k)+'" cy="'+y(p.rate)+'" r="'+(p.k===selected?5:3.5)+'" fill="'+color+'" stroke="white" stroke-width="1" role="button" tabindex="0" aria-pressed="'+String(p.k===selected)+'" aria-label="'+esc(label+' '+name+' Top-'+p.k+' '+fmt(p.rate))+'"><title>'+esc(label+' · '+name+' Top-'+p.k+' '+fmt(p.rate)+' ('+p.correct+'/'+p.n+')')+'</title></circle>';
    }
    host.innerHTML=svg+'</svg>';
    host.querySelectorAll('[data-slot-k]').forEach(el=>{
      const choose=()=>{active.slotK[slot]=Number(el.dataset.slotK);$c('encoder-slot-k-'+slot).value=String(active.slotK[slot]);renderSlot(slot);};
      el.addEventListener('click',choose);el.addEventListener('keydown',event=>{if(event.key==='Enter'||event.key===' '){event.preventDefault();choose();}});
    });
  }
  function renderSlot(slot) {
    const k=active.slotK[slot],a=slotPoint('no_encoder',slot,k),b=slotPoint('encoder',slot,k);
    const audit=report.variants.no_encoder.slots[slot].audit || {};
    $c('encoder-slot-scope-'+slot).textContent='채점 가능 '+count(a.n)+' / '+count(report.variants.no_encoder.evaluation_rows)+'행 · 미지정 정답(- 등) '+count(audit.placeholder_n)+'행 포함';
    $c('encoder-slot-scores-'+slot).innerHTML=series.map(([variant,label,color])=>{
      const p=slotPoint(variant,slot,k);
      return '<div data-variant="'+variant+'" style="--comparison-color:'+color+'"><span>'+label.replace('학습 ','')+' · Top-'+k+'</span><strong>'+fmt(p.rate)+'</strong><small>'+count(p.correct)+' / '+count(p.n)+'행</small></div>';
    }).join('')+'<div class="slot-score-delta"><span>사용 − 없음</span><strong>'+delta(a.rate,b.rate)+'</strong></div>';
    const macro=$c('encoder-slot-macro-'+slot);macro.hidden=active.mode!=='raw';
    macro.textContent='모델 Macro F1 · 없음 '+fmt(report.variants.no_encoder.raw_heads?.[slot]?.macro_f1)+' / 사용 '+fmt(report.variants.encoder.raw_heads?.[slot]?.macro_f1);
    $c('encoder-slot-table-'+slot).innerHTML=[1,3,5,10].map(kk=>{
      const pa=slotPoint('no_encoder',slot,kk),pb=slotPoint('encoder',slot,kk);
      return '<tr'+(kk===k?' class="comparison-selected"':'')+'><th scope="row">Top-'+kk+'</th>'+[pa,pb].map(p=>'<td>'+fmt(p.rate)+'<small>'+count(p.correct)+' / '+count(p.n)+'</small></td>').join('')+'<td>'+delta(pa.rate,pb.rate)+'</td></tr>';
    }).join('');
    $c('encoder-slot-detail-'+slot).textContent='Top-'+k+' · '+modeLabels[active.mode]+' · 없음 '+fmt(a.rate)+' ('+count(a.correct)+'/'+count(a.n)+') · 사용 '+fmt(b.rate)+' ('+count(b.correct)+'/'+count(b.n)+')';
    drawSlot(slot);
  }
  function renderSlots() {
    $c('encoder-slot-note').textContent={
      raw:'각 칸 학습 모델이 낸 독립 후보를 채점합니다. 예를 들어 Vendor Top-5는 Vendor 후보 5개 안의 정답 포함률입니다. 완성 코드 5개의 일치율과 다릅니다. 각 패널의 K는 다른 칸과 독립적으로 바뀝니다.',
      part:'규칙과 모델을 반영한 각 칸의 독립 후보를 채점합니다. 여섯 칸을 합친 코드의 순위는 사용하지 않습니다. 후보가 K개보다 적으면 실제 있는 후보까지만 확인합니다.',
      joint:'앞의 K개 완성 코드에서 해석할 수 있는 해당 칸에 정답이 있는지 채점합니다. 이는 각 칸 모델 자체의 순위와 다릅니다. 구간 의미가 미확인인 후보에는 상세 칸 정답을 임의로 채우지 않습니다.'
    }[active.mode]+' 칸별 정답을 확인할 수 있는 행을 분모로 고정하며, 미지정 정답(-, -- 등)도 포함합니다.';
    slotLabels.forEach(([slot])=>renderSlot(slot));
  }
  function saveBlob(content,type,filename) {
    const url=URL.createObjectURL(new Blob([content],{type}));
    const a=document.createElement('a');a.href=url;a.download=filename;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
  }
  function render() {
    const a=point('no_encoder',active.k), b=point('encoder',active.k);
    $c('encoder-scope-note').textContent=scopes[active.scope]+' · 고정 분모 '+count(group('no_encoder').n)+'행 · 결과 후보가 없는 행: 인코더 없음 '+count(group('no_encoder').n-group('no_encoder').predicted_rows)+' / 사용 '+count(group('encoder').n-group('encoder').predicted_rows)+'행';
    $c('encoder-summary').innerHTML=series.map(([id,label,color])=>{const p=point(id,active.k);return '<div class="metric-card" style="--comparison-color:'+color+'"><span>'+label+' · Top-'+active.k+'</span><strong>'+fmt(p.rate)+'</strong><small>'+count(p.correct)+' / '+count(p.n)+'행 · '+scopes[active.scope]+'</small></div>';}).join('')+'<div class="metric-card"><span>학습 인코더 사용에 따른 차이</span><strong class="comparison-delta">'+delta(a.rate,b.rate)+'</strong><small>Top-'+active.k+' · '+count(b.correct-a.correct)+'행 차이</small></div>';
    $c('encoder-legend').innerHTML=series.map(([id,label,color])=>'<span><i style="background:'+color+'"></i>'+label+' · n='+count(group(id).n)+'</span>').join('');
    $c('encoder-k-table').innerHTML=[1,3,5,10].map(k=>{const pa=point('no_encoder',k),pb=point('encoder',k);return '<tr'+(k===active.k?' class="comparison-selected"':'')+'><th scope="row">Top-'+k+'</th>'+[pa,pb].map(p=>'<td>'+fmt(p.rate)+'<small>'+count(p.correct)+' / '+count(p.n)+'행</small></td>').join('')+'<td>'+delta(pa.rate,pb.rate)+'</td><td>'+Number(pa.average_candidates??0).toFixed(2)+' / '+Number(pb.average_candidates??0).toFixed(2)+'</td></tr>';}).join('');
    chart();
  }
  function downloads() {
    const variant=$c('encoder-part-variant').value;
    $c('encoder-part-downloads').innerHTML=slotLabels.map(([slot,label])=>'<a class="download-link" href="comparison-downloads/'+variant+'/'+slot+'_predictions.csv" download>'+label+' 결과 CSV ↓</a>').join('');
  }
  $c('encoder-protocol').innerHTML='<p>동일 평가 분모: 전체 '+count(report.protocol?.fixed_denominators?.all)+'행 / 매뉴얼 해석 가능 '+count(report.protocol?.fixed_denominators?.manual)+'행 / 구간 규칙 미확인 '+count(report.protocol?.fixed_denominators?.unmapped)+'행.</p><p>기준 모델의 학습 내부 그룹 검증으로 칸별 모델과 가중치를 고정했습니다. 인코더 학습 횟수도 학습 내부 검증으로 정했고, 최종 평가 CSV는 학습이나 선택에 사용하지 않았습니다. 인코더 추가 모델의 내부 교차 검증 성능이나 확률 보정 성능을 측정했다고 주장하지 않습니다.</p><p>이 평가 CSV는 이전에 분석한 진단 데이터로, 처음 보는 블라인드 시험은 아닙니다. 규칙 미확인 품목은 원본 완성 코드 일치만 채점하며 상세 칸의 의미를 임의로 정답 처리하지 않습니다.</p><p>두 방식의 6개 칸 학습 행·클래스·트리 설정과 공통 규칙 파일의 일치 여부는 <a href="comparison-downloads/comparison.json" download>비교 원본 JSON</a>에서 확인할 수 있습니다.</p>';
  $c('encoder-scope').addEventListener('change',event=>{active.scope=event.target.value;render();});
  $c('encoder-k').addEventListener('change',event=>{active.k=Number(event.target.value);render();});
  $c('encoder-slot-mode').addEventListener('change',event=>{active.mode=event.target.value;renderSlots();});
  panel.querySelectorAll('[data-slot-k-select]').forEach(select=>select.addEventListener('change',()=>{
    const slot=select.dataset.slotKSelect;active.slotK[slot]=Number(select.value);renderSlot(slot);
  }));
  panel.querySelectorAll('[data-slot-svg]').forEach(button=>button.addEventListener('click',()=>{
    const slot=button.dataset.slotSvg,svg=$c('encoder-slot-chart-'+slot).querySelector('svg');
    if(svg)saveBlob(svg.outerHTML,'image/svg+xml;charset=utf-8',slot+'_'+active.mode+'_top_k.svg');
  }));
  panel.querySelectorAll('[data-slot-csv]').forEach(button=>button.addEventListener('click',()=>{
    const slot=button.dataset.slotCsv,label=slotLabels.find(([key])=>key===slot)[1];
    const rows=[['slot','label','mode','variant','k','correct','n','rate']];
    for(const [variant] of series)for(let k=1;k<=10;k++){
      const p=slotPoint(variant,slot,k);rows.push([slot,label,active.mode,variant,k,p.correct??'',p.n??'',p.rate??'']);
    }
    const csv='\uFEFF'+rows.map(row=>row.map(v=>'"'+String(v).replaceAll('"','""')+'"').join(',')).join('\r\n');
    saveBlob(csv,'text/csv;charset=utf-8',slot+'_'+active.mode+'_top_k.csv');
  }));
  $c('encoder-part-variant').addEventListener('change',downloads);
  $c('encoder-chart-save').addEventListener('click',()=>{
    const svg=$c('encoder-chart').querySelector('svg');if(!svg)return;
    const url=URL.createObjectURL(new Blob([svg.outerHTML],{type:'image/svg+xml;charset=utf-8'}));
    const link=document.createElement('a');link.href=url;link.download='인코더유무_TopK_'+active.scope+'.svg';link.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
  });
  render();renderSlots();downloads();
  if (panel._encoderObserver) panel._encoderObserver.disconnect();
  panel._encoderObserver=new ResizeObserver(chart);panel._encoderObserver.observe($c('encoder-chart'));
  if(panel._slotObserver)panel._slotObserver.disconnect();
  panel._slotObserver=new ResizeObserver(entries=>entries.forEach(entry=>drawSlot(entry.target.dataset.slotChart)));
  panel.querySelectorAll('[data-slot-chart]').forEach(host=>panel._slotObserver.observe(host));
};
