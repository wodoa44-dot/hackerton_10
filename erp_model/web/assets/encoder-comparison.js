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
  page.querySelector('.page-heading p').textContent = '동일한 평가 행과 후보 수로 학습 인코더의 효과를 비교합니다. 학습 내부 검증으로 모델과 학습 횟수를 정했습니다.';
  const badge = page.querySelector('.page-heading .outline-badge');
  if (badge) badge.textContent = '동일 조건 · 실제 비교 평가';
  panel.innerHTML = `
    <section class="panel comparison-intro">
      <div class="context-heading"><div><div class="eyebrow">ENCODER A/B · TOP-K</div><h2>학습 인코더를 더했을 때 얼마나 달라졌나요?</h2></div><span class="outline-badge">후보 상한 10개 · 두 방식 동일</span></div>
      <p>공통 전처리 → TF-IDF·구조화 특징 → XGBoost·LightGBM·Forest → 코드 규칙과 후보 조합. 인코더 사용 방식은 여기에 <strong>직접 학습한 문자 CNN 특징</strong>을 추가합니다. 칸별 모델 선택·가중치와 조회·규칙은 동일하게 유지했습니다.</p>
      <div class="comparison-controls"><label>평가 범위<select id="encoder-scope"><option value="all">전체 품목</option><option value="manual">매뉴얼 해석 가능</option><option value="unmapped">구간 규칙 미확인</option></select></label><label>K · 확인할 완성 코드 후보 수<select id="encoder-k">${Array.from({length:10},(_,i)=>'<option value="'+(i+1)+'"'+(i===4?' selected':'')+'>Top-'+(i+1)+'</option>').join('')}</select></label></div>
      <p id="encoder-scope-note" class="muted"></p>
    </section>
    <div class="comparison-cards" id="encoder-summary"></div>
    <section class="panel table-panel">
      <div class="context-heading"><h2>완성 코드 · K에 따른 정답 포함률</h2><button id="encoder-chart-save" class="batch-secondary" type="button">그래프 저장 ↓</button></div>
      <p class="muted">Hit@K = 앞의 K개 완성 코드 안에 원본 정답 코드가 있는 행 수 ÷ 해당 범위의 전체 행 수. 코드가 나오지 않은 행도 분모에 포함합니다. K보다 후보가 적으면 실제 생성된 후보까지만 확인합니다.</p>
      <div id="encoder-legend" class="comparison-legend"></div><div id="encoder-chart" class="comparison-chart"></div><p id="encoder-chart-detail" class="k-chart-detail" role="status" aria-live="polite"></p>
      <div class="table-scroll"><table><thead><tr><th>후보 수 K</th><th>학습 인코더 없음</th><th>학습 인코더 사용</th><th>차이 · 사용 − 없음</th><th>평균 생성 후보 수 · 없음 / 사용</th></tr></thead><tbody id="encoder-k-table"></tbody></table></div>
    </section>
    <section class="panel table-panel">
      <div class="context-heading"><h2>6개 칸별 비교</h2><label class="comparison-field-label">평가 결과<select id="encoder-slot-mode"><option value="final">최종 코드 결과</option><option value="raw">각 칸 학습 모델 자체</option></select></label></div>
      <p id="encoder-slot-note" class="muted"></p>
      <div class="table-scroll"><table><thead><tr><th>예측 칸</th><th>채점 가능 / 전체</th><th>없음 · Top-1</th><th>사용 · Top-1</th><th id="encoder-a-k">없음 · Top-5</th><th id="encoder-b-k">사용 · Top-5</th><th>Top-K 차이</th><th>Macro F1 · 없음 / 사용</th></tr></thead><tbody id="encoder-slot-table"></tbody></table></div>
      <p class="muted">이 표는 각 칸 정답이 확인되는 행만 채점합니다. 위의 완성 코드 그래프는 선택한 범위 전체를 채점합니다. 지정값 정확도는 ‘-’, ‘--’ 등 미지정 정답을 제외한 별도 수치입니다.</p>
    </section>
    <section class="panel table-panel">
      <h2>예측 결과 다운로드</h2><p class="muted">같은 행의 두 방식 결과와 K별 정답 여부를 직접 대조할 수 있습니다.</p>
      <div class="downloads"><a class="download-link" href="comparison-downloads/paired_predictions.csv" download>행별 A/B 결과 CSV ↓</a><a class="download-link" href="comparison-downloads/top_k_comparison.csv" download>Top-1·3·5·10 CSV ↓</a><a class="download-link" href="comparison-downloads/candidate_k_curve.csv" download>K=1~10 전체 수치 CSV ↓</a><a class="download-link" href="comparison-downloads/six_slot_comparison.csv" download>6개 칸 평가 CSV ↓</a></div>
      <div class="comparison-controls"><label>칸별 결과 파일<select id="encoder-part-variant"><option value="no_encoder">학습 인코더 없음</option><option value="encoder">학습 인코더 사용</option></select></label></div><div id="encoder-part-downloads" class="downloads"></div>
      <details class="comparison-protocol"><summary>비교 조건과 평가 범위</summary><div id="encoder-protocol"></div></details>
    </section>`;
  const $c = id => document.getElementById(id);
  const active = {scope:'all',k:5,mode:'final'};
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
  function renderSlots() {
    const k=active.k, raw=active.mode==='raw';
    $c('encoder-a-k').textContent='없음 · Top-'+k;
    $c('encoder-b-k').textContent='사용 · Top-'+k;
    $c('encoder-slot-note').textContent=raw ? '입력 규칙·조회·코드 조합 전, 각 칸 학습 모델의 후보를 따로 채점합니다. Top-K는 독립 칸 후보 K개 안의 정답 포함률이며 완성 코드 Top-K와 다릅니다. 수치형 세분류도 같은 정답 범위에 포함하므로 규칙으로 계산해야 하는 값은 원시 품번 랭커에서 오답이 될 수 있습니다. 이 표는 위 범위 필터와 독립적으로 칸별 전체 채점 가능 행을 평가합니다.' : 'Top-1은 최종 표시된 칸 값의 정확도입니다. Top-K는 앞의 K개 완성 코드에서 해석 가능한 해당 칸 중 정답이 하나라도 있는 비율입니다. 각 칸의 후보를 임의로 섞지 않습니다. 이 표는 위 그래프의 범위 필터와 독립적으로 칸별 전체 채점 가능 행을 평가합니다.';
    $c('encoder-slot-table').innerHTML=slotLabels.map(([slot,label])=>{
      const values=series.map(([id])=>raw ? report.variants[id].raw_heads?.[slot] : report.variants[id].slots[slot]);
      const [a,b]=values, audits=series.map(([id])=>report.variants[id].slots[slot].audit || {});
      const rates=values.map(value=>(raw?value?.at_k:value?.joint_at_k)?.[String(k)]);
      const top1=values.map((value,i)=>'<td>'+fmt(value?.accuracy)+(raw?'':'<small>지정값 '+fmt(audits[i].designated_accuracy)+' · '+count(audits[i].designated_correct)+'/'+count(audits[i].designated_n)+'</small>')+'</td>').join('');
      return '<tr><th scope="row">'+label+'</th><td>'+count(a?.n)+' / '+count(report.variants.no_encoder.evaluation_rows)+'<small>'+fmt(audits[0].coverage)+' 채점 가능</small></td>'+top1+rates.map(value=>'<td>'+fmt(value?.rate)+'<small>'+count(value?.correct)+' / '+count(value?.n)+'</small></td>').join('')+'<td>'+delta(rates[0]?.rate,rates[1]?.rate)+'</td><td>'+fmt(a?.macro_f1)+' / '+fmt(b?.macro_f1)+'</td></tr>';
    }).join('');
  }
  function render() {
    const a=point('no_encoder',active.k), b=point('encoder',active.k);
    $c('encoder-scope-note').textContent=scopes[active.scope]+' · 고정 분모 '+count(group('no_encoder').n)+'행 · 결과 후보가 없는 행: 인코더 없음 '+count(group('no_encoder').n-group('no_encoder').predicted_rows)+' / 사용 '+count(group('encoder').n-group('encoder').predicted_rows)+'행';
    $c('encoder-summary').innerHTML=series.map(([id,label,color])=>{const p=point(id,active.k);return '<div class="metric-card" style="--comparison-color:'+color+'"><span>'+label+' · Top-'+active.k+'</span><strong>'+fmt(p.rate)+'</strong><small>'+count(p.correct)+' / '+count(p.n)+'행 · '+scopes[active.scope]+'</small></div>';}).join('')+'<div class="metric-card"><span>학습 인코더 사용에 따른 차이</span><strong class="comparison-delta">'+delta(a.rate,b.rate)+'</strong><small>Top-'+active.k+' · '+count(b.correct-a.correct)+'행 차이</small></div>';
    $c('encoder-legend').innerHTML=series.map(([id,label,color])=>'<span><i style="background:'+color+'"></i>'+label+' · n='+count(group(id).n)+'</span>').join('');
    $c('encoder-k-table').innerHTML=[1,3,5,10].map(k=>{const pa=point('no_encoder',k),pb=point('encoder',k);return '<tr'+(k===active.k?' class="comparison-selected"':'')+'><th scope="row">Top-'+k+'</th>'+[pa,pb].map(p=>'<td>'+fmt(p.rate)+'<small>'+count(p.correct)+' / '+count(p.n)+'행</small></td>').join('')+'<td>'+delta(pa.rate,pb.rate)+'</td><td>'+Number(pa.average_candidates??0).toFixed(2)+' / '+Number(pb.average_candidates??0).toFixed(2)+'</td></tr>';}).join('');
    chart();renderSlots();
  }
  function downloads() {
    const variant=$c('encoder-part-variant').value;
    $c('encoder-part-downloads').innerHTML=slotLabels.map(([slot,label])=>'<a class="download-link" href="comparison-downloads/'+variant+'/'+slot+'_predictions.csv" download>'+label+' 결과 CSV ↓</a>').join('');
  }
  $c('encoder-protocol').innerHTML='<p>동일 평가 분모: 전체 '+count(report.protocol?.fixed_denominators?.all)+'행 / 매뉴얼 해석 가능 '+count(report.protocol?.fixed_denominators?.manual)+'행 / 구간 규칙 미확인 '+count(report.protocol?.fixed_denominators?.unmapped)+'행.</p><p>기준 모델의 학습 내부 그룹 검증으로 칸별 모델과 가중치를 고정했습니다. 인코더 학습 횟수도 학습 내부 검증으로 정했고, 최종 평가 CSV는 학습이나 선택에 사용하지 않았습니다. 인코더 추가 모델의 내부 교차 검증 성능이나 확률 보정 성능을 측정했다고 주장하지 않습니다.</p><p>이 평가 CSV는 이전에 분석한 진단 데이터로, 처음 보는 블라인드 시험은 아닙니다. 규칙 미확인 품목은 원본 완성 코드 일치만 채점하며 상세 칸의 의미를 임의로 정답 처리하지 않습니다.</p><p>두 방식의 6개 칸 학습 행·클래스·트리 설정과 공통 규칙 파일의 일치 여부는 <a href="comparison-downloads/comparison.json" download>비교 원본 JSON</a>에서 확인할 수 있습니다.</p>';
  $c('encoder-scope').addEventListener('change',event=>{active.scope=event.target.value;render();});
  $c('encoder-k').addEventListener('change',event=>{active.k=Number(event.target.value);render();});
  $c('encoder-slot-mode').addEventListener('change',event=>{active.mode=event.target.value;renderSlots();});
  $c('encoder-part-variant').addEventListener('change',downloads);
  $c('encoder-chart-save').addEventListener('click',()=>{
    const svg=$c('encoder-chart').querySelector('svg');if(!svg)return;
    const url=URL.createObjectURL(new Blob([svg.outerHTML],{type:'image/svg+xml;charset=utf-8'}));
    const link=document.createElement('a');link.href=url;link.download='인코더유무_TopK_'+active.scope+'.svg';link.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
  });
  render();downloads();
  if (panel._encoderObserver) panel._encoderObserver.disconnect();
  panel._encoderObserver=new ResizeObserver(chart);panel._encoderObserver.observe($c('encoder-chart'));
};
