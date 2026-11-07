'use strict';
const slots = [
  ['classification', '코드 분류', 'CODE CLASS', 1],
  ['trait', '특성', 'TRAIT', 1],
  ['part_number', '세분류 · 품번', 'PART NUMBER', 7],
  ['case', '형상', 'CASE', 2],
  ['packing', 'Packing', 'PACKING', 1],
  ['vendor', 'Vendor', 'VENDOR', 1],
];
const names = Object.fromEntries(slots.map(s => [s[0], s[1]]));
const $ = id => document.getElementById(id);
const escape = value => String(value ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const percent = value => value == null ? '—' : `${(value * 100).toFixed(2)}%`;
function candidateScore(candidate) {
  if (candidate.score == null) return candidate.score_type?.startsWith('registered') ? '등록 코드 근거' : '입력·코드 규칙 근거';
  if (candidate.score_type === 'relative_rank_score_not_probability') return '상대 순위 점수 ' + Number(candidate.score).toFixed(3);
  if (candidate.score_type === 'mean_rank_lower_is_better') return '평균 순위 ' + Number(candidate.score).toFixed(1) + '위';
  return '모델 점수 ' + percent(candidate.score);
}
const states = {confirmed:'확정', review:'예측 완료', error:'오류', unsupported_schema:'구간 규칙 미확인', not_applicable:'해당 없음', idle:'입력 대기', loading:'예측 중'};
let active = false;
let results = {};
let completed = 0;
let requested = [];
let assembledCode = null;
let materialContext = null;
let selectedRegistryId = null;
let lastAssembly = null;
let selectedSpectrumRank = 0;
let selectedHypothesis = null;
function navigate(page) {
  document.querySelectorAll('.page').forEach(el => { el.hidden = el.id !== `page-${page}`; });
  document.querySelectorAll('.nav-item').forEach(el => el.classList.toggle('active', el.dataset.page === page));
  $('breadcrumb').textContent = {workspace:'품목코드 예측',evaluation:'모델 평가 결과',guide:'사용 가이드'}[page];
  window.scrollTo({top:0, behavior:'smooth'});
}
document.querySelectorAll('[data-page]').forEach(el => el.addEventListener('click', () => navigate(el.dataset.page)));
function makeCard([slot, title, label], i) {
  const card = document.createElement('article');
  card.className = 'result-card'; card.dataset.slot = slot; card.dataset.status = 'idle'; card.id = `card-${slot}`;
  card.innerHTML = `<div class="card-title"><span class="slot-title">${title}</span><span class="slot-index">0${i+1}</span></div><div class="slot-key">${label}</div><div class="slot-value-label">입력 대기</div><div class="slot-value">—</div><div class="candidate-preview" hidden></div><p class="result-reason" hidden></p><div class="card-footer"><span class="status">입력 대기</span><button type="button" class="detail-button" disabled aria-expanded="false">후보·근거 ＋</button></div><div class="details" hidden></div>`;
  const detailButton = card.querySelector('button');
  detailButton.addEventListener('click', () => {
    const details = card.querySelector('.details'); details.hidden = !details.hidden;
    detailButton.setAttribute('aria-expanded', String(!details.hidden));
    detailButton.textContent = details.hidden ? '후보·근거 ＋' : '접기 －';
  });
  return card;
}
slots.forEach((slot,i) => $('result-grid').append(makeCard(slot,i)));
function renderSegments() {
  if (selectedHypothesis) {
    $('code-segments').hidden = !selectedHypothesis.semantic_layout_verified;
    $('code-segments').innerHTML = selectedHypothesis.parts.map(piece =>
      '<div class="code-segment" data-code-position="'+piece.start+'"><div class="segment-value predicted">'+escape(piece.value)+'</div><div class="segment-label">'+escape(piece.label)+'</div></div>').join('');
    return;
  }
  $('code-segments').hidden = Boolean(materialContext && !materialContext.generation_supported);
  const pieces = [];
  const layout = materialContext?.slot_layout;
  if (layout) {
    for (const [slot, description] of Object.entries(layout)) {
      const result = results[slot];
      const value = result?.value ?? result?.predicted_value;
      let offset = 0;
      for (const span of description.spans) {
        const length = span.end - span.start;
        pieces.push({...span, value: value == null ? '·'.repeat(length) : value.slice(offset, offset + length), confirmed: result?.status === 'confirmed'});
        offset += length;
      }
    }
    for (const item of materialContext.fixed_segments || []) pieces.push({...item, confirmed:true});
    pieces.sort((a,b) => a.start - b.start);
  } else {
    slots.forEach(([slot,label,,length],i) => pieces.push({start:i,label,value:'·'.repeat(length)}));
  }
  $('code-segments').innerHTML = pieces.map(piece =>
    '<div class="code-segment" data-code-position="' + piece.start + '"><div class="segment-value ' + (piece.confirmed?'filled':'predicted') + '">' + escape(piece.value) + '</div><div class="segment-label">' + escape(piece.label) + '</div></div>').join('');
}
renderSegments();
function resetCard(slot, status) {
  const card = $(`card-${slot}`); card.dataset.status = status;
  card.querySelector('.slot-value').textContent = status === 'loading' ? '…' : '—';
  card.querySelector('.slot-value-label').textContent = status === 'loading' ? '예측 결과를 기다리는 중' : '입력 대기';
  card.querySelector('.result-reason').hidden = true;
  card.querySelector('.candidate-preview').hidden = true;
  card.querySelector('.status').className = 'status';
  card.querySelector('.status').textContent = states[status];
  card.querySelector('.details').hidden = true;
  card.querySelector('.details').innerHTML = '';
  const button = card.querySelector('button'); button.disabled = true; button.setAttribute('aria-expanded','false');button.textContent='후보·근거 ＋';
}
function humanReason(reason) {
  if (!reason) return '';
  const translated = {
    no_candidates:'입력에서 품번 후보를 찾지 못했습니다.',
    rank_tie:'두 개 이상의 후보가 같은 순위를 받아 확인이 필요합니다.',
    insufficient_candidate_evidence:'형상·단위 등으로 보이거나 품번 근거가 부족합니다.',
    insufficient_oof_rank_accuracy:'학습 검증에서 품번의 자동 확정 기준을 충족하지 못했습니다.',
    weak_rank_agreement:'두 랭커의 순위 일치도가 확정 기준에 미치지 못했습니다.',
    below_oof_confidence_threshold:'학습 검증으로 정한 점수 또는 1·2위 차이 기준에 미치지 못했습니다.',
    no_explicit_manufacturer_evidence:'입력에 명확한 제조사 단서가 없습니다.',
    conflicting_manufacturer_evidence:'입력에 여러 제조사 단서가 함께 있어 확인이 필요합니다.',
    manufacturer_evidence_disagrees_with_model_or_unmapped:'제조사 단서가 예측값과 다르거나 코드표에 연결되지 않습니다.',
    label_absent_in_cv_training_fold:'일부 학습 폴드에 없는 희귀 라벨로, 확인이 필요합니다.',
    'OOF score and margin thresholds passed':'학습 검증으로 정한 점수와 1·2위 차이 기준을 통과했습니다.',
    'Unique rank winner with input evidence and OOF-validated agreement':'입력 근거와 랭커 일치 기준을 통과한 단독 1순위 후보입니다.',
    'Validated code tables and 1+1+7+2+1+1 lengths':'코드표와 자리 수 검증을 통과했습니다.',
    'Code table/length validation failed':'코드표 또는 자리 수 검증을 통과하지 못했습니다.'
  };
  return reason.split('; ').map(part=>translated[part]||part).join(' ');
}
function resultEvent(event) {
  if (event.slot === 'context') { renderContext(event); return; }
  if (event.context) renderContext(event.context);
  if (event.slot === 'assembly') { lastAssembly=event; selectedSpectrumRank=0; renderSpectrum(event); renderExploratory(event); selectHypothesis(0); return; }
  if (!names[event.slot]) return;
  const isNew = !results[event.slot]; results[event.slot] = event;
  if (isNew) completed += 1;
  $('counter').textContent = `${completed} / ${requested.length} 완료`;
  const card = $(`card-${event.slot}`); card.dataset.status = event.status;
  const topCandidate = (event.top_candidates || event.candidates)?.[0];
  card.querySelector('.slot-value').textContent = event.requires_input_review ? '입력 확인' : event.value ?? event.predicted_value ?? topCandidate?.value ?? (event.status === 'error' ? '오류' : event.status === 'unsupported_schema' ? '해석 보류' : event.status === 'not_applicable' ? '해당 없음' : '후보 없음');
  card.querySelector('.slot-value-label').textContent = event.status === 'confirmed' ? '예측값 · 자동 검증 통과' : topCandidate ? '1순위 예측값' : '예측 결과';
  const reason = card.querySelector('.result-reason');
  reason.textContent = humanReason(event.reason); reason.hidden = !event.requires_input_review && !['error','unsupported_schema','not_applicable'].includes(event.status) && !event.reason?.includes('conflicting_manufacturer_evidence'); reason.title = reason.textContent;
  const status = card.querySelector('.status'); status.className = `status ${event.status}`; status.textContent = states[event.status]||'확인 필요';
  const candidates = event.candidates || [];
  const preview = card.querySelector('.candidate-preview'); preview.hidden = !candidates.length;
  preview.textContent = topCandidate ? candidateScore(topCandidate) : '';
  const detail = card.querySelector('.details');
  const rankScore = candidates[0]?.score_type === 'mean_rank_lower_is_better';
  detail.innerHTML = `<h4>후보와 판단 근거</h4>${candidates.slice(0,5).map(c=>`<div class="candidate-row"><code>${escape(c.value)}</code><span>${escape(candidateScore(c))}</span></div>`).join('') || '<p>후보가 없습니다.</p>'}<h4>판단 근거</h4><p>${escape(humanReason(event.reason))}</p><h4>입력에서 찾은 단서</h4>${(event.evidence||[]).slice(0,12).map(e=>`<span class="evidence-chip" title="${escape(e.transformation)}">${escape(e.source_column)} · ${escape(e.matched_text)} → ${escape(e.normalized_value)}</span>`).join('') || '<p>명시적인 구조화 단서가 없습니다.</p>'}${candidates.some(c=>c.score_type==='probability_uncalibrated')?'<p style="margin-top:9px">모델 확률은 별도로 보정하지 않은 점수입니다.</p>':''}${event.fields?.length?'<h4>코드 구간</h4>'+event.fields.map(f=>'<div class="candidate-row"><span>'+escape(f.label)+'</span><code>'+escape(f.value)+'</code></div>').join(''):''}`;
  card.querySelector('button').disabled = false;
  renderSegments();
  const counts = {confirmed:0,review:0,error:0,unsupported_schema:0,not_applicable:0};
  Object.values(results).forEach(r => { if (r.status in counts) counts[r.status] += 1; });
  $('result-summary').hidden = false;
  $('result-summary').textContent = `${completed}개 칸 예측 완료 · 자동 검증 통과 ${counts.confirmed} · 검토 권장 ${counts.review} · 구간 규칙 미확인 ${counts.unsupported_schema} · 해당 없음 ${counts.not_applicable} · 오류 ${counts.error}`;
}
function renderContext(context) {
  materialContext=context;
  $('context-panel').hidden=false;
  $('context-family').textContent='품목군 '+context.component_family;
  $('context-scheme').textContent=(context.code_scheme_title || '품목코드')+' · 하나의 예측 흐름에서 등록 근거와 새 코드 후보를 함께 비교합니다.';
  const matches=context.lookup_candidates || context.registry_candidates || [];
  $('registry-list').innerHTML=matches.length?'<details><summary>등록 근거 '+matches.length+'건 확인</summary>'+
    matches.slice(0,8).map(m=>'<article class="registry-card"><strong class="registry-code">'+escape(m.code)+'</strong><p>'+escape(m.registered_name)+' · '+escape(m.registered_spec)+'</p><p class="muted">'+
    (m.retrieval_basis==='normalized_name_and_specification'?'품목명·규격명 정규화 일치':m.retrieval_basis==='manufacturer_part_agreement'?'입력 품번과 등록 품번 일치':'텍스트 유사 후보 · 실제 사양 확인 필요')+'</p></article>').join('')+'</details>':'<p class="registry-empty">입력과 가까운 등록 근거가 없습니다.</p>';
  $('required-info').innerHTML=(context.required_information||[]).map(x=>'<p class="muted">'+escape(x)+'</p>').join('');
  renderSegments();
}
function renderSpectrum(event) {
  const list=event.spectrum || [];
  $('spectrum-panel').hidden=!list.length;
  $('spectrum-note').textContent='후보 '+list.length+'개 · 상대 점수는 정답 확률이 아닙니다. 후보를 선택하면 여섯 칸과 세부 파트가 함께 바뀝니다.';
  $('spectrum-body').innerHTML=list.map((c,i)=>'<tr data-spectrum-row="'+i+'"><td><button type="button" data-hypothesis="'+i+'">'+(i+1)+'순위 선택</button></td><td><code>'+escape(c.code)+'</code></td><td>'+Number(c.relative_score).toFixed(3)+'</td><td>'+escape(c.source==='registry_exact'?'등록 일치':c.source==='registry_candidate'?'등록 후보':'규칙·모델 조립')+'</td><td>'+(c.input_numeric_consistent===true?'수치 일치':c.input_numeric_consistent===false?'입력 수치와 충돌':'수치 판독 근거 없음')+'</td></tr>').join('');
  $('spectrum-body').querySelectorAll('[data-hypothesis]').forEach(b=>b.addEventListener('click',()=>selectHypothesis(Number(b.dataset.hypothesis))));
}
function selectHypothesis(index) {
  const candidate=lastAssembly?.spectrum?.[index];
  selectedSpectrumRank=index;selectedHypothesis=candidate || null;
  if(candidate) {
    slots.forEach(([slot])=>{
      const value=candidate.values[slot];
      const applicable=value !== null && value !== undefined && value !== '';
      const fields=candidate.parts.filter(p=>p.slot===slot);
      const status=applicable?(candidate.auto_confirmed?'confirmed':'review'):(candidate.semantic_layout_verified?'not_applicable':'unsupported_schema');
      const candidates=(lastAssembly.part_spectra?.[slot] || []).filter(c=>c.value !== null && c.value !== '').map(c=>({value:c.value,score:c.relative_score,score_type:'relative_rank_score_not_probability'}));
      const selectedCandidates=applicable?[{value,score:null,score_type:'selected_hypothesis'},...candidates.filter(c=>c.value!==value)]:[];
      resultEvent({...results[slot],slot,value:candidate.auto_confirmed?value:null,predicted_value:applicable?value:null,status,fields,
        requires_input_review:false,candidates:selectedCandidates,top_candidates:selectedCandidates,
        reason:applicable?'선택한 '+(index+1)+'순위 코드의 해당 구간입니다.':candidate.semantic_layout_verified?'이 코드 체계에는 해당 구간이 없습니다.':'매뉴얼에서 구간 의미를 확인하지 못했습니다.'});
    });
    $('parts-body').innerHTML=candidate.parts.map(p=>'<tr><td>'+escape(names[p.slot] || (p.slot==='fixed'?'고정 구분자':'미해석'))+'</td><td>'+escape(p.label)+'</td><td>'+p.position_start+'–'+p.position_end+'</td><td><code>'+escape(p.value)+'</code></td><td>'+escape(p.meaning || (p.slot==='unmapped'?'구간 의미 미확인':'—'))+'</td></tr>').join('');
  } else {
    $('parts-body').innerHTML='';
  }
  document.querySelectorAll('[data-spectrum-row]').forEach(row=>row.classList.toggle('selected-hypothesis',Number(row.dataset.spectrumRow)===index));
  renderAssembly(lastAssembly || {});
}
function renderAssembly(event) {
  const selected=selectedHypothesis;
  const code=selected?.code ?? event.display_code ?? event.value ?? event.predicted_code;
  const source=selected?.source ?? event.code_source;
  const auto=selected?.auto_confirmed ?? event.auto_confirmed ?? event.status==='confirmed';
  const badge=$('assembly-badge');
  badge.textContent=code?(auto?'등록 코드 · 완전 일치':'후보 '+(selectedSpectrumRank+1)+' · 검토 권장'):'예측값 확인 필요';
  badge.className='assembly-badge '+(auto?'confirmed':'review');
  $('predicted-code').textContent=code||'—';
  $('code-title').textContent=code?'ERP 품목코드 · '+code.length+'자리':'ERP 품목코드';
  $('code-provenance').textContent=source?({registry_exact:'출처 · 등록 품목명과 규격명 일치',registry_candidate:'출처 · 등록 품목코드 후보',model_assembled:'출처 · 회사 규칙 + 여섯 칸 예측 조립',reference_candidate:'출처 · 유사한 등록 품목의 코드 후보'}[source]||source):'';
  $('assembly-reason').textContent=selected?'선택한 후보의 여섯 칸과 실제 코드 위치별 파트를 아래에서 확인하세요.':humanReason(event.reason || '품번과 코드 규칙을 확인하세요.');
  assembledCode=code||null;$('copy-code').disabled=!assembledCode;
  renderSegments();
}

$('copy-code').addEventListener('click', async () => {
  if (!assembledCode) return;
  try { await navigator.clipboard.writeText(assembledCode); $('copy-code').textContent = '복사 완료 ✓'; }
  catch { $('assembly-reason').textContent = `완성 코드: ${assembledCode} · 직접 선택해 복사하세요.`; }
});
document.querySelectorAll('[data-sample]').forEach(button=>button.addEventListener('click',()=>{
  if(active)return;
  const samples = {thermistor:['THERMISTOR','IDU,TWI/TWO/THO'],basic:['DIODE','1N4148, DO-35, TAPING, VISHAY'],conflict:['DIODE','BZT52-C3V3, 3.3V, 0.5W, SOD-123, SAMYANG, PANJIT'],resistor:['RES CHIP','1608 F 12.3K, REEL, YAGEO'],capacitor:['CAP CHIP','1608 X7R 0.47uF K 50V, REEL'],thermistor_ntc:['THERMISTOR NTC','NTX9234 BULK'],fuse:['FUSE','SERIES 217 3.15A GLASS 5X20'],varistor:['VARISTOR','INR14D681K CNR']};
  [$('name').value, $('spec').value] = samples[button.dataset.sample];
  $('scheme').value='';
  $('slot').value='';$('name').focus();
}));
$('predict-form').addEventListener('submit',async e=>{
  e.preventDefault();if(active)return;
  if(!$('name').value.trim())return;
  active=true;results={};completed=0;assembledCode=null;materialContext=null;selectedRegistryId=null;lastAssembly=null;selectedSpectrumRank=0;selectedHypothesis=null;$('exploratory-panel').hidden=true;$('spectrum-panel').hidden=true;$('parts-body').innerHTML='';$('context-panel').hidden=true;$('code-title').textContent='ERP 품목코드';$('code-provenance').textContent='';
  requested=$('slot').value?[$('slot').value]:slots.map(s=>s[0]);
  slots.forEach(([s])=>resetCard(s,requested.includes(s)?'loading':'idle'));
  renderSegments();$('predicted-code').textContent='—';$('result-summary').hidden=true;$('form-error').hidden=true;$('copy-code').disabled=true;$('copy-code').innerHTML='코드 복사 <span>⧉</span>';
  $('assembly-badge').textContent=requested.length===1?'단일 칸 실행':'예측 중';$('assembly-badge').className='assembly-badge';
  $('assembly-reason').textContent=requested.length===1?'단일 칸 실행에서는 완성 코드를 조립하지 않습니다.':'각 칸의 확정 여부를 확인하고 있습니다.';
  $('counter').textContent=`0 / ${requested.length} 완료`;
  $('run-button').disabled=true;$('run-button').firstElementChild.textContent='모델 예측 중…';
  let gotAssembly=false;
  function consume(line){if(!line.trim())return;const item=JSON.parse(line);if(item.slot==='assembly')gotAssembly=true;resultEvent(item);}
  try {
    const response=await fetch('predict',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({name:$('name').value,spec:$('spec').value,slot:$('slot').value||null,code_scheme_id:$('scheme').value||null,candidate_limit:Number($('candidate-limit').value)})});
    if(!response.ok)throw new Error(`예측 요청에 실패했습니다. (HTTP ${response.status})`);
    if(!response.body)throw new Error('이 브라우저는 스트리밍 응답을 지원하지 않습니다.');
    const reader=response.body.getReader();const decoder=new TextDecoder();let buffer='';
    while(true){const {done,value}=await reader.read();if(done)break;buffer+=decoder.decode(value,{stream:true});let newline;while((newline=buffer.indexOf('\n'))>=0){consume(buffer.slice(0,newline));buffer=buffer.slice(newline+1);}}
    buffer+=decoder.decode();if(buffer.trim())consume(buffer);
    if(completed!==requested.length || (requested.length===6&&!gotAssembly))throw new Error('일부 결과를 받지 못했습니다. 다시 실행해 주세요.');
  } catch(err) {
    $('form-error').textContent=err.message;$('form-error').hidden=false;
    requested.filter(s=>!results[s]).forEach(s=>resultEvent({slot:s,value:null,status:'error',candidates:[],evidence:[],reason:'서버 응답을 받지 못했습니다.'}));
    if(requested.length===6&&!gotAssembly)renderAssembly({status:'review',value:null,reason:'일부 칸의 응답이 없어 조립을 보류합니다.'});
  } finally {active=false;$('run-button').disabled=false;$('run-button').firstElementChild.textContent='품목코드 예측하기';}
});
async function loadModelInfo() {
  try {
    const response = await fetch('model-info');
    if (!response.ok) throw new Error('모델 정보 응답 오류');
    const info = await response.json();
    $('batch-panel').hidden=!info.csv_upload;
    const metrics = info.generation_metrics;
    const spectrum = info.spectrum_metrics;
    const counts = info.training_counts;
    const curves=info.candidate_k_metrics;
    window.renderKChart(curves);
    const topOne=curves?.groups?.all?.points?.[0] || spectrum?.exact_at_k?.['1'];
    const topFive=curves?.groups?.all?.points?.[4] || spectrum?.exact_at_k?.['5'];
    $('connection').textContent = '전체 코드 모델 연결됨';
    const select = $('scheme');
    select.innerHTML = '<option value="">품목명·규격명으로 자동 선택</option>' +
      info.schemes.map(s => `<option value="${escape(s.id)}">${escape(s.prefix)} · ${escape(s.title)} · ${s.length}자리</option>`).join('');
    $('stat-precision').textContent = percent(metrics?.slots?.classification?.accuracy);
    $('stat-recall').textContent = counts.classification.classes + '종';
    $('stat-part').textContent = percent(metrics?.code_available_rate);
    const cards = spectrum ? [
      ['전체 코드 Top 1 포함률', topOne.rate, topOne.correct+' / '+spectrum.evaluation_rows+'건 · 원문 완전 일치'],
      ['추정 포함 전체 Top 5 포함률', topFive.rate, topFive.correct+' / '+spectrum.evaluation_rows+'건 · 원문 완전 일치'],
      ['매뉴얼 일치행 Top 5 포함률', spectrum.manual_exact_at_k['5'].rate, spectrum.manual_exact_at_k['5'].correct+' / '+spectrum.company_layout_rows+'건'],
      ['상세 코드 채점 가능 범위', spectrum.company_layout_rows/spectrum.evaluation_rows, spectrum.company_layout_rows+' / '+spectrum.evaluation_rows+'건 · 나머지는 상세 정답 판정 불가']
    ] : [
      ['실제 코드 분류 정확도', metrics?.slots?.classification?.accuracy, metrics?.evaluation_rows+'건'],
      ['코드 후보 제공률', metrics?.code_available_rate, metrics?.code_available+'건'],
      ['완성 코드 일치율', metrics?.full_code_exact_rate, '학습 참조만 사용'],
      ['매뉴얼 일치행 완성 코드 정확도', metrics?.manual_full_code_exact_rate, metrics?.company_layout_rows+'건']
    ];
    $('metric-cards').innerHTML = cards.map(([title, value, note]) =>
      `<div class="metric-card"><span>${escape(title)}</span><strong>${percent(value)}</strong><small>${escape(note)}</small></div>`).join('');
    $('evaluation-body').innerHTML = slots.map(([slot, title]) => {
      const m = metrics?.slots?.[slot]; const audit=spectrum?.slot_audit?.[slot];
      const scope=(m?.n ?? 0).toLocaleString()+' / '+(metrics?.evaluation_rows ?? '—').toLocaleString();
      return '<tr><td>'+escape(title)+'</td><td>'+scope+'<small class="metric-note">'+percent(audit?.coverage)+' 채점 가능</small></td><td>'+percent(m?.accuracy)+'<small class="metric-note">'+(audit?.correct ?? '—')+' / '+(m?.n ?? '—')+'건</small></td><td>'+percent(audit?.designated_accuracy)+'<small class="metric-note">'+(audit?.designated_correct ?? '—')+' / '+(audit?.designated_n ?? '—')+'건</small></td><td>'+percent(m?.macro_f1)+'</td><td>'+percent(audit?.training_majority_baseline)+'</td><td>'+percent(m?.spectrum_at_k?.['5'])+'</td></tr>';
    }).join('');
    const lookup=info.web_lookup_audit;
    $('lookup-policy').textContent=lookup?'웹·CSV 조회는 학습 참조 '+lookup.current_lookup_rows.toLocaleString()+'행만 사용합니다. inference와 같은 입력 '+lookup.evaluation_input_overlap+'건 · 같은 등록 코드 '+lookup.evaluation_code_overlap+'건.':'학습·조회 범위를 확인하고 있습니다.';
    $('actual-code-evaluation').innerHTML = '<p>입력은 품목명과 규격명 두 열입니다. 분류는 실제 코드 접두값을 학습하며, 상세 칸은 회사 규칙에 따라 분해한 값으로 학습합니다.</p>' +
      '<div class="table-scroll"><table><thead><tr><th>칸</th><th>학습 행</th><th>실제 라벨 수</th><th>XGBoost GPU 학습</th></tr></thead><tbody>' +
      slots.map(([slot, title]) => `<tr><td>${escape(title)}</td><td>${counts[slot].rows.toLocaleString()}</td><td>${counts[slot].classes}</td><td>${info.selection[slot]?.gpu_training_verified ? 'CUDA 확인' : '확인 필요'}</td></tr>`).join('') +
      '</tbody></table></div><p>세분류의 수치형 용량·저항값은 입력에서 계산하고, 4·5·6·7·8·10자리 품번은 입력에서 만든 후보를 두 랭커가 평가합니다. 일부 희귀 라벨은 검증 폴드의 학습에 없어 평가에서 오답으로 집계될 수 있습니다.</p>';
    const vendor = metrics?.slots?.vendor;
    const vendorAudit=spectrum?.slot_audit?.vendor;
    $('vendor-diagnosis').innerHTML='<p>전체 Vendor 정확도 '+percent(vendor?.accuracy)+'는 미지정 표기(-) '+(vendorAudit?.placeholder_n ?? '—')+'건을 포함합니다. 지정값이 있는 '+(vendorAudit?.designated_n ?? '—')+'건의 정확도는 <strong>'+percent(vendorAudit?.designated_accuracy)+'</strong> ('+(vendorAudit?.designated_correct ?? '—')+'건 일치)입니다.</p><p>학습 참조의 최빈값만 고르는 기준 '+percent(vendorAudit?.training_majority_baseline)+' · Macro F1 '+percent(vendor?.macro_f1)+'. 최종 결과에는 입력 규칙·모델·학습 참조가 함께 작용합니다. 순수 모델 성능으로 읽지 마세요.</p>';
    $('assembly-evaluation').innerHTML = `<div class="plain-stat"><span>전체 평가의 완성 코드 일치</span><strong>${metrics?.full_code_correct ?? '—'} / ${metrics?.evaluation_rows ?? '—'}</strong><span>${percent(metrics?.full_code_exact_rate)}</span></div><div class="plain-stat"><span>회사 규칙으로 조립한 코드 후보</span><strong>${metrics?.model_generated_codes ?? '—'}건</strong><span>전체 코드 후보 제공 ${metrics?.code_available ?? '—'}건</span></div><p>웹 예측·CSV 채움·성능 평가 모두 학습 참조만 사용합니다. inference 행은 등록 조회에서 제외했습니다.</p>`;
    $('assembly-evaluation').innerHTML += '<p>매뉴얼과 같은 코드 규칙의 행만 비교한 완성 코드 일치: '+(metrics?.manual_full_code_correct ?? '—')+' / '+(metrics?.company_layout_rows ?? '—')+'건 ('+percent(metrics?.manual_full_code_exact_rate)+'). 전체 평가와 구분해 표시합니다.</p>';
    if (spectrum) {
      if(curves) {
        const point=(key,k)=>curves.groups[key].points[k-1];
        $('assembly-evaluation').innerHTML += '<div class="table-scroll"><table><thead><tr><th>후보 수 상한</th><th>전체 정답 포함률</th><th>매뉴얼 해석 가능</th><th>규칙 미확인 · 추정 포함</th><th>평균 후보 수</th></tr></thead><tbody>'+[1,3,5,10].map(k=>'<tr><td>Top '+k+'</td><td>'+percent(point('all',k).rate)+'</td><td>'+percent(point('manual',k).rate)+'</td><td>'+percent(point('unmapped',k).rate)+'</td><td>'+point('all',k).average_candidates.toFixed(2)+'</td></tr>').join('')+'</tbody></table></div><p>위 표와 그래프는 규칙 미확인 품목의 확장 추정 후보를 포함합니다. 칸별 평가는 기존 해석값 기준이며, 미해석 코드의 세부 칸을 정답으로 간주하지 않습니다.</p>';
      }
      const validation=info.pipeline_validation;
      if(validation) $('assembly-evaluation').innerHTML += '<p>정책 선택용 학습 검증 '+validation.scores.baseline.n+'행: 완성 코드 정확도 '+percent(validation.scores.baseline.accuracy)+' → '+percent(validation.scores[validation.selected].accuracy)+'. 최종 평가 CSV는 정책 선택에 사용하지 않았습니다.</p>';
      $('downloads').innerHTML=(info.downloads || []).map(name=>'<a class="download-link" href="downloads/'+encodeURIComponent(name)+'" download>'+escape({'original_code_parts.csv':'원본 코드 파트 전체','prediction_parts.csv':'1순위 예측 파트','spectrum_parts_top5.csv':'기존 Top 5 후보별 파트','exploratory_predictions.csv':'규칙 미확인 추정 후보'}[name] || name)+' CSV ↓</a>').join('');
    }
    const comparison = info.manual_comparison;
    if(comparison) $('actual-code-evaluation').innerHTML += '<p>매뉴얼·CSV 대조 '+comparison.total_rows.toLocaleString()+'행: 코드표 일치 '+comparison.status_counts.matched.toLocaleString()+'행 · 구간 규칙 불일치 '+comparison.status_counts.layout_conflict.toLocaleString()+'행 · 매뉴얼 외 형식 '+comparison.status_counts.outside_manual.toLocaleString()+'행. 상세 모델은 코드표 검증을 통과한 라벨만 학습합니다.</p>';
    const audit = info.data_audit;
    $('training-data-flow').innerHTML = [
      [audit.train_original, '원본 학습 행'], [audit.excluded_overlap, '평가 입력 중복 제외'],
      [counts.classification.rows, '분류 학습 행'], [audit.company_layout_training_rows, '상세 구간 라벨 행']
    ].map(([value, title]) => `<div><strong>${value.toLocaleString()}</strong><span>${escape(title)}</span></div>`).join('<b>→</b>');
  } catch (error) {
    $('connection').textContent = '연결 확인 필요';
    $('connection').classList.add('offline');
    $('evaluation-body').innerHTML = '<tr><td colspan="7">현재 모델 평가 정보를 불러오지 못했습니다.</td></tr>';
    console.error(error);
  }
}
loadModelInfo();

let batchActive=false;
function showBatchError(message) {
  $('batch-error').textContent=message;$('batch-error').hidden=false;
  batchActive=false;$('batch-upload').disabled=false;$('batch-file').disabled=false;
}
async function pollBatch(jobId) {
  try {
    const response=await fetch('batch-csv/'+encodeURIComponent(jobId),{cache:'no-store'});
    const job=await response.json();
    if(!response.ok)throw new Error(job.detail || '처리 정보를 가져오지 못했습니다.');
    $('batch-progress').hidden=false;$('batch-progress').max=job.total || 1;$('batch-progress').value=job.processed;
    $('batch-status').textContent=(job.status==='queued'?'처리 대기 · ':job.status==='completed'?'완료 · ':'처리 중 · ')+job.processed+' / '+job.total+'행 · 기존 코드 '+job.kept+' · 새로 채움 '+job.filled+' · 미생성 '+job.unfilled+' · 오류 '+job.errors;
    $('batch-preview').hidden=!(job.preview || []).length;
    $('batch-preview-body').innerHTML=(job.preview || []).map(r=>'<tr><td>'+r.row+'</td><td>'+escape(r.name)+'</td><td><code>'+escape(r.code)+'</code></td><td>'+escape(r.status)+'</td></tr>').join('');
    if(job.status==='failed')throw new Error(job.error || 'CSV 처리에 실패했습니다.');
    if(job.status==='completed') {
      batchActive=false;$('batch-upload').disabled=false;$('batch-file').disabled=false;
      $('batch-download').href='batch-csv/'+encodeURIComponent(job.id)+'/download';$('batch-download').hidden=false;
      $('batch-status').textContent+=' · 검토 필요 '+job.review+'행';
      return;
    }
    setTimeout(()=>pollBatch(jobId),1000);
  } catch(error) {showBatchError(error.message);}
}
$('batch-upload').addEventListener('click',async()=>{
  if(batchActive)return;
  const file=$('batch-file').files?.[0];
  if(!file){showBatchError('CSV 파일을 선택해 주세요.');return;}
  if(file.size>20*1024*1024){showBatchError('CSV 파일은 20MB 이하로 올려주세요.');return;}
  batchActive=true;$('batch-upload').disabled=true;$('batch-file').disabled=true;
  $('batch-error').hidden=true;$('batch-download').hidden=true;$('batch-preview').hidden=true;
  $('batch-status').textContent='파일을 올리고 있습니다…';$('batch-progress').hidden=true;
  try {
    const response=await fetch('batch-csv',{method:'POST',headers:{'Content-Type':'text/csv','X-Filename':encodeURIComponent(file.name)},body:file});
    const job=await response.json();
    if(!response.ok)throw new Error(job.detail || 'CSV를 올리지 못했습니다.');
    try{localStorage.setItem('erp-last-csv-job',job.id);}catch{}
    await pollBatch(job.id);
  } catch(error) {showBatchError(error.message);}
});
$('batch-template').addEventListener('click',()=>{
  const text='\uFEFF품목명,규격명,품목코드\nRES CHIP,"1608 F 12.3K REEL YAGEO",\nCAP CHIP,"1608 X7R 0.47uF K 50V REEL",\n';
  const url=URL.createObjectURL(new Blob([text],{type:'text/csv;charset=utf-8'}));
  const a=document.createElement('a');a.href=url;a.download='빈코드_입력양식.csv';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
});
try {
  const last=localStorage.getItem('erp-last-csv-job');
  if(last){$('batch-details').open=true;batchActive=true;$('batch-upload').disabled=true;pollBatch(last);}
}catch{}

function renderExploratory(event) {
  const extra=event.exploratory;const panel=$('exploratory-panel');
  panel.hidden=!extra?.active;
  if(!extra?.active)return;
  $('exploratory-note').textContent=extra.reason || '규칙 미확인 품목의 추정 후보입니다.';
  const candidates=extra.candidates || [];
  $('exploratory-body').innerHTML=candidates.map((c,i)=>'<tr><td>'+(i+1)+'</td><td><code>'+escape(c.code)+'</code></td><td>'+escape({training_pattern_adaptation:'학습 코드 형태 + 입력 식별자',training_reference:'학습 유사 품목',registry_candidate:'학습 코드 참고',registry_exact:'학습 등록 일치',training_frequency_prior:'입력 근거 부족 · 학습 빈도 참고'}[c.source] || c.source)+'</td><td>'+escape(c.input_identifier || '—')+'</td><td><button type="button" data-copy-estimate="'+i+'">추정 코드 복사</button></td></tr>').join('') || '<tr><td colspan="5">후보를 만들 학습 근거가 없습니다.</td></tr>';
  $('exploratory-body').querySelectorAll('[data-copy-estimate]').forEach(button=>button.addEventListener('click',async()=>{
    const code=candidates[Number(button.dataset.copyEstimate)].code;
    try{await navigator.clipboard.writeText(code);button.textContent='복사 완료 ✓';}
    catch{button.textContent=code;}
  }));
  $('exploratory-heads').innerHTML=slots.map(([slot,label])=>{
    const item=extra.head_estimates?.[slot];const values=(item?.candidates || []).map(c=>c.value);
    return '<tr><td>'+escape(label)+'</td><td>'+values.map(v=>'<code>'+escape(v)+'</code>').join(' / ')+'</td><td>'+escape(values.length?'적용 규칙 미검증':'추정할 입력·학습 근거 부족')+'</td></tr>';
  }).join('');
}
