/* =====================================================================
 * 품목코드 생성 엔진 (engine.js)
 *  - 값 변환(정전용량/저항치), 코드 조립·분해, 중복검사
 *  - 내장 AI: TF-IDF 유사 품목 검색 + 규칙 기반 부품정보 해석기
 *  - 외부 통신 코드 없음 (로컬 LLM 연동은 llm.js 에서 127.0.0.1 루프백만 사용)
 * ===================================================================== */
(function () {
  'use strict';
  const R = window.CODE_RULES;
  const LEN = R.codeLength;

  /* ---------------- 공통 유틸 ---------------- */
  const U = {
    up: s => String(s == null ? '' : s).toUpperCase(),
    // 전각 문자·특수기호 정규화
    norm: s => String(s == null ? '' : s)
      .replace(/[／]/g, '/').replace(/[，]/g, ',').replace(/[（]/g, '(').replace(/[）]/g, ')')
      .replace(/[％]/g, '%').replace(/[＊×]/g, '*').replace(/[µμ]/g, 'U').replace(/Ω/g, 'OHM')
      .replace(/[‘’′`]/g, "'").replace(/[＂]/g, '"').replace(/\s+/g, ' ').trim(),
    num: s => { const v = parseFloat(String(s).replace(/,/g, '')); return isFinite(v) ? v : NaN; },
    trimNum: v => { let s = (+v).toPrecision(6); s = String(parseFloat(s)); return s; },
    esc: s => String(s == null ? '' : s).replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]))
  };

  /* ---------------- 정전용량 ---------------- */
  // 문자열 → pF 값
  function parseCapPF(str) {
    if (str == null) return NaN;
    let s = U.up(U.norm(str)).replace(/\s+/g, '');
    if (!s) return NaN;
    // 이미 3자리 코드인 경우 (472, 4R7, 0R5, R47)
    if (/^\d{3}$/.test(s)) return decodeCap3(s);
    if (/^\d?R\d{1,2}$/.test(s)) return decodeCap3(s);
    const m = s.match(/^(\d+(?:\.\d+)?)(PF|P|NF|N|UF|U|MFD|MF|F)?$/);
    if (!m) return NaN;
    const v = parseFloat(m[1]); const unit = m[2] || 'PF';
    const mult = { PF: 1, P: 1, NF: 1e3, N: 1e3, UF: 1e6, U: 1e6, MFD: 1e6, MF: 1e6, F: 1e12 }[unit];
    return v * mult;
  }
  // pF → EIA 3자리 코드
  function capCode(pf) {
    if (!isFinite(pf) || pf <= 0) return '';
    if (pf < 10) { // 0.5 → 0R5, 4.7 → 4R7
      const s = (Math.round(pf * 10) / 10).toFixed(1).split('.');
      return (s[0] === '0' ? '0' : s[0]) + 'R' + s[1];
    }
    let e = Math.floor(Math.log10(pf)) - 1;
    let sig = Math.round(pf / Math.pow(10, e));
    if (sig >= 100) { sig = Math.round(sig / 10); e += 1; }
    if (e > 9) return '';
    return String(sig).padStart(2, '0') + e;
  }
  // uF 기준 3자리 (매뉴얼 전해콘덴서 예시: 0.47uF R47, 4.7uF 4R7, 47uF 470, 470uF 471)
  function capCodeUF(pf) {
    if (!isFinite(pf) || pf <= 0) return '';
    const uf = pf / 1e6;
    if (uf < 1) return 'R' + String(Math.round(uf * 100)).padStart(2, '0');
    if (uf < 10) { const s = (Math.round(uf * 10) / 10).toFixed(1).split('.'); return s[0] + 'R' + s[1]; }
    let e = Math.floor(Math.log10(uf)) - 1; let sig = Math.round(uf / Math.pow(10, e));
    if (sig >= 100) { sig = Math.round(sig / 10); e += 1; }
    return String(sig) + e;
  }
  function decodeCap3(c) {
    c = U.up(c);
    if (/^\d{3}$/.test(c)) return parseInt(c.slice(0, 2), 10) * Math.pow(10, parseInt(c[2], 10));
    if (/^\d?R\d{1,2}$/.test(c)) return parseFloat(c.replace('R', '.').replace(/^\./, '0.'));
    return NaN;
  }
  function decodeCapUF(c) {
    c = U.up(c);
    if (/^R\d{2}$/.test(c)) return parseFloat('0.' + c.slice(1)) * 1e6;
    if (/^\dR\d$/.test(c)) return parseFloat(c.replace('R', '.')) * 1e6;
    if (/^\d{3}$/.test(c)) return parseInt(c.slice(0, 2), 10) * Math.pow(10, parseInt(c[2], 10)) * 1e6;
    return NaN;
  }
  // 매뉴얼: 기본 pF, 큰 값은 uF (nF 사용 안 함)
  function capDisp(pf) {
    if (!isFinite(pf)) return '';
    if (pf < 10000) return U.trimNum(pf) + 'pF';
    return U.trimNum(pf / 1e6) + 'uF';
  }

  /* ---------------- 저항치 ---------------- */
  function parseOhm(str) {
    if (str == null) return NaN;
    const lm = String(str).match(/^\s*(\d+(?:\.\d+)?)\s*m(?:Ω|ohm|OHM|R|r)?\s*$/); // 소문자 m = 밀리옴
    if (lm) return parseFloat(lm[1]) / 1000;
    let s = U.up(U.norm(str)).replace(/\s+/g, '').replace(/OHMS?$/, '');
    if (!s) return NaN;
    let m;
    if ((m = s.match(/^(\d+)([RKM])(\d+)$/))) { // 4K7, 1R8, 2M2
      const v = parseFloat(m[1] + '.' + m[3]); return v * { R: 1, K: 1e3, M: 1e6 }[m[2]];
    }
    if ((m = s.match(/^(\d+(?:\.\d+)?)(MR|MOHM|MΩ)$/))) return parseFloat(m[1]) / 1000; // 밀리옴
    if ((m = s.match(/^(\d+(?:\.\d+)?)(R|K|KOHM|M|MEG|G)?$/))) {
      const v = parseFloat(m[1]); const u = m[2] || 'R';
      return v * ({ R: 1, K: 1e3, KOHM: 1e3, M: 1e6, MEG: 1e6, G: 1e9 }[u]);
    }
    return NaN;
  }
  // 4자리: 10Ω→0100, 1.8K→1801, 100K→1003, 1.8Ω→1R80, 0.5Ω→0R50, 62mΩ→62M0
  function res4(ohm) {
    if (!isFinite(ohm) || ohm < 0) return '';
    if (ohm === 0) return '0000';
    if (ohm < 0.01 || (ohm < 1 && Math.abs(ohm * 100 - Math.round(ohm * 100)) > 1e-6)) {
      const m = ohm * 1000; // mΩ
      const s = U.trimNum(m).replace('.', 'M');
      const out = (s.includes('M') ? s : s + 'M').padEnd(4, '0');
      return out.slice(0, 4);
    }
    if (ohm < 10 && Math.abs(ohm - Math.round(ohm)) > 1e-6) return (Math.round(ohm * 100) / 100).toFixed(2).replace('.', 'R').slice(0, 4);
    if (ohm < 100 && Math.abs(ohm - Math.round(ohm)) > 1e-6) return (Math.round(ohm * 10) / 10).toFixed(1).replace('.', 'R');
    for (let e = 0; e <= 9; e++) {
      const man = ohm / Math.pow(10, e);
      if (man < 1000 && Math.abs(man - Math.round(man)) < 1e-6) return String(Math.round(man)).padStart(3, '0') + e;
    }
    // 유효숫자 3자리 반올림
    let e = Math.max(0, Math.floor(Math.log10(ohm)) - 2); const man = Math.round(ohm / Math.pow(10, e));
    return String(man).padStart(3, '0') + e;
  }
  function decodeRes4(c) {
    c = U.up(c);
    if (/^\d{4}$/.test(c)) return parseInt(c.slice(0, 3), 10) * Math.pow(10, parseInt(c[3], 10));
    if (/^\d*R\d*$/.test(c)) return parseFloat(c.replace('R', '.'));
    if (/^\d*M\d*$/.test(c)) return parseFloat(c.replace('M', '.')) / 1000;
    return NaN;
  }
  // 3자리: 10K→103, 3.9K→392, 4.7Ω→4R7
  function res3(ohm) {
    if (!isFinite(ohm) || ohm < 0) return '';
    if (ohm === 0) return '000';
    if (ohm < 10) return (Math.round(ohm * 10) / 10).toFixed(1).replace('.', 'R');
    let e = Math.floor(Math.log10(ohm)) - 1; let sig = Math.round(ohm / Math.pow(10, e));
    if (sig >= 100) { sig = Math.round(sig / 10); e++; }
    return String(sig) + e;
  }
  function decodeRes3(c) {
    c = U.up(c);
    if (/^\d{3}$/.test(c)) return parseInt(c.slice(0, 2), 10) * Math.pow(10, parseInt(c[2], 10));
    if (/^\dR\d$/.test(c)) return parseFloat(c.replace('R', '.'));
    return NaN;
  }
  function ohmDisp(o) {
    if (!isFinite(o)) return '';
    if (o < 1 && o > 0) return U.trimNum(o) + 'R';
    if (o < 1000) return U.trimNum(o) + 'R';
    if (o < 1e6) return U.trimNum(o / 1e3) + 'K';
    return U.trimNum(o / 1e6) + 'M';
  }

  /* ---------------- 문자열 필드 정규화 ---------------- */
  function cleanTxt(raw, keepDash) {
    let s = U.up(U.norm(raw)).replace(/[^A-Z0-9\-]/g, '');
    if (!keepDash) s = s.replace(/-/g, '');
    return s;
  }
  // 8자 초과 시 구분 가능한 길이로 압축 (영문 앞부분 축약 + 숫자/접미부 유지)
  function compress(s, n) {
    if (s.length <= n) return { v: s, compressed: false };
    const m = s.match(/^([A-Z]+)(.*)$/);
    if (m && m[2].length > 0 && m[2].length < n) return { v: m[1].slice(0, n - m[2].length) + m[2], compressed: true };
    return { v: s.slice(0, n), compressed: true };
  }
  function padField(s, n, ch) { return (s + ch.repeat(n)).slice(0, n); }

  /* ---------------- 분류 조회 ---------------- */
  const CAT = {};
  R.categories.forEach(c => { CAT[c.id] = c; c.prefixLen = c.prefix.length; c.total = c.prefix.length + c.fields.reduce((a, f) => a + f.n, 0); });
  function catOfCode(code) {
    code = U.up(code);
    if (code.length !== LEN) return null;
    if (/^J[A-Z0-9]{2}[A-Z0-9]{6}S\d{3}$/.test(code)) return CAT.JACC;
    let best = null;
    for (const c of R.categories) {
      if (!c.prefix || c.id === 'JACC') continue;
      if (code.startsWith(c.prefix) && (!best || c.prefix.length > best.prefix.length)) best = c;
    }
    return best;
  }
  function optMeaning(f, code) { if (!f.o) return ''; const o = f.o.find(x => x[0] === code); return o ? o[1] : ''; }

  // 코드 → 필드 분해
  function decode(code, cat) {
    code = U.up(code); cat = cat || catOfCode(code);
    if (!cat || code.length !== cat.total) return null;
    let p = cat.prefix.length; const vals = {};
    for (const f of cat.fields) {
      const seg = code.substr(p, f.n); p += f.n;
      const v = { code: seg, raw: '', fromCode: true };
      if (f.t === 'cap') { const pf = decodeCap3(seg); v.raw = isFinite(pf) ? capDisp(pf) : seg; }
      else if (f.t === 'res4') { const o = decodeRes4(seg); v.raw = isFinite(o) ? ohmDisp(o) : seg; }
      else if (f.t === 'res3') { const o = decodeRes3(seg); v.raw = isFinite(o) ? ohmDisp(o) : seg; }
      else if (f.t === 'txt') v.raw = seg.replace(/-+$/, '');
      vals[f.k] = v;
    }
    return { cat, vals };
  }

  /* ---------------- 데이터(마스터) ---------------- */
  const DB = { items: [], codeMap: new Map(), source: '', date: '', posCount: {} };
  function setItems(rows, source, date) {
    DB.items = rows; DB.source = source; DB.date = date || '';
    DB.codeMap = new Map(); rows.forEach((r, i) => DB.codeMap.set(U.up(r[0]), i));
    DB.posCount = {}; IDX.ready = false;
  }
  // 분류별 각 자리 문자 사용빈도 (드롭다운 "ERP n건" 표시용)
  function posCount(cat) {
    if (DB.posCount[cat.id]) return DB.posCount[cat.id];
    const res = {}; cat.fields.forEach(f => res[f.k] = {});
    for (const r of DB.items) {
      const code = U.up(r[0]); if (code.length !== cat.total || !code.startsWith(cat.prefix)) continue;
      if (catOfCode(code) !== cat) continue;
      let p = cat.prefix.length;
      for (const f of cat.fields) { const seg = code.substr(p, f.n); p += f.n; res[f.k][seg] = (res[f.k][seg] || 0) + 1; }
    }
    DB.posCount[cat.id] = res; return res;
  }
  function nextSerial(prefix, width, extraCodes) {
    let max = 0; const re = new RegExp('^' + prefix.replace(/[.*+?^${}()|[\]\\]/g, '\\$&') + '(\\d{' + width + '})');
    const scan = c => { const m = U.up(c).match(re); if (m) max = Math.max(max, parseInt(m[1], 10)); };
    DB.items.forEach(r => scan(r[0])); (extraCodes || []).forEach(scan);
    return String(max + 1).padStart(width, '0');
  }

  /* ---------------- 내장 AI ①: TF-IDF 유사 품목 검색 ---------------- */
  const IDX = { ready: false, building: false, post: null, norms: null, idf: null, N: 0 };
  function featurize(text, code) {
    const t = U.up(U.norm(text));
    const feats = new Map();
    const add = (k, w) => feats.set(k, (feats.get(k) || 0) + w);
    const toks = t.split(/[^A-Z0-9가-힣.\-+]+/).filter(Boolean);
    for (let tok of toks) {
      tok = tok.replace(/^[.\-]+|[.\-]+$/g, '');
      if (!tok) continue;
      add('W:' + tok, 2);
      const comp = tok.replace(/[.\-]/g, '');
      if (comp !== tok && comp.length > 1) add('W:' + comp, 1);
      const g = '^' + comp + '$';
      if (comp.length >= 3) for (let i = 0; i + 3 <= g.length; i++) add('G:' + g.substr(i, 3), 1);
    }
    // 값 정규화 특징 (4.7nF = 4700pF = 472)
    const capRe = /(\d+(?:\.\d+)?)\s*(PF|NF|UF|MFD|MF)\b/g; let m;
    while ((m = capRe.exec(t))) { const pf = parseCapPF(m[1] + m[2]); if (isFinite(pf)) add('CAP:' + capCode(pf), 3); }
    const vRe = /(\d+(?:\.\d+)?)\s*(K?)V\b/g;
    while ((m = vRe.exec(t))) add('V:' + (parseFloat(m[1]) * (m[2] ? 1000 : 1)), 2);
    const rRe = /(\d+(?:\.\d+)?)\s*(K|M|R|OHM)\b|\b(\d+)([RK])(\d+)\b/g;
    while ((m = rRe.exec(t))) { const o = parseOhm(m[0].replace(/\s/g, '')); if (isFinite(o)) add('RES:' + res4(o), 3); }
    if (code) {
      const c = U.up(code); add('P2:' + c.slice(0, 2), 1); add('P3:' + c.slice(0, 3), 1);
      const d = decode(c);
      if (d) {
        const cf = d.cat.fields.find(f => f.t === 'cap'); if (cf) add('CAP:' + d.vals[cf.k].code, 3);
        const rf = d.cat.fields.find(f => f.t === 'res4'); if (rf) add('RES:' + d.vals[rf.k].code, 3);
      }
    }
    return feats;
  }
  function buildIndex(onProgress) {
    return new Promise(resolve => {
      IDX.building = true;
      const items = DB.items, N = items.length; const post = new Map(); const docFeats = new Array(N);
      let i = 0;
      function step() {
        const end = Math.min(N, i + 2500);
        for (; i < end; i++) {
          const r = items[i];
          const f = featurize(r[0] + ' ' + r[1] + ' ' + r[2], r[0]);
          docFeats[i] = f;
          f.forEach((tf, k) => { let p = post.get(k); if (!p) { p = []; post.set(k, p); } p.push(i, tf); });
        }
        onProgress && onProgress(i / N);
        if (i < N) { setTimeout(step, 0); return; }
        const idf = new Map(); post.forEach((p, k) => idf.set(k, Math.log(1 + N / (p.length / 2))));
        const norms = new Float32Array(N);
        for (let d = 0; d < N; d++) { let s = 0; docFeats[d].forEach((tf, k) => { const w = (1 + Math.log(tf)) * idf.get(k); s += w * w; }); norms[d] = Math.sqrt(s) || 1; }
        Object.assign(IDX, { post, idf, norms, N, ready: true, building: false });
        resolve();
      }
      step();
    });
  }
  function similar(text, topK) {
    if (!IDX.ready) return [];
    const q = featurize(text, null); const scores = new Float32Array(IDX.N); let qn = 0;
    q.forEach((tf, k) => {
      const idf = IDX.idf.get(k); if (!idf) return;
      const p = IDX.post.get(k); if (p.length / 2 > IDX.N * 0.25) return; // 너무 흔한 특징 제외
      const wq = (1 + Math.log(tf)) * idf; qn += wq * wq;
      for (let j = 0; j < p.length; j += 2) scores[p[j]] += wq * (1 + Math.log(p[j + 1])) * idf;
    });
    qn = Math.sqrt(qn) || 1;
    const res = [];
    for (let d = 0; d < IDX.N; d++) if (scores[d] > 0) res.push([d, scores[d] / (IDX.norms[d] * qn)]);
    res.sort((a, b) => b[1] - a[1]);
    return res.slice(0, topK || 10).map(([d, s]) => ({ idx: d, score: s, row: DB.items[d] }));
  }

  /* ---------------- 내장 AI ②: 규칙 기반 부품정보 해석기 ---------------- */
  function kwList(s) { return (s || '').split(',').map(x => U.up(x.trim())).filter(Boolean); }
  function hasKw(T, kw) {
    if (/^[A-Z0-9]+$/.test(kw) && kw.length <= 3) return new RegExp('(^|[^A-Z0-9])' + kw + '([^A-Z0-9]|$)').test(T);
    return T.includes(kw);
  }
  function voltOf(meaning) {
    const m = U.up(meaning).match(/^(AC)?(\d+(?:\.\d+)?)(K?)V$/); if (!m) return null;
    return { ac: !!m[1], v: parseFloat(m[2]) * (m[3] ? 1000 : 1) };
  }
  // 텍스트에서 값 추출
  function extract(text) {
    const T = ' ' + U.up(U.norm(text)) + ' ';
    const out = { T };
    let m;
    // 정전용량
    const isRes = /RES|저항|OHM|REGIST/.test(T);
    const capRe = /(\d+(?:\.\d+)?)\s*(PF|NF|UF|MFD|MF)(?=[^A-Z]|[JKMZ](?![A-Z])|$)/;
    if ((m = T.match(capRe))) { out.capPF = parseCapPF(m[1] + m[2]); out.capRaw = m[0].trim(); }
    else if (!isRes && (m = T.match(/(?:^|[^A-Z0-9])(\d{2}[0-9]|\dR\d)([JKMZ])(?:[^A-Z0-9]|$)/))) { out.capPF = decodeCap3(m[1]); out.capRaw = m[1]; out.tolLetterCap = m[2]; }
    // 전압 (AC 판별: AC275V, 275VAC, 275V~)
    const vs = []; const vRe = /(AC|DC)?\s*(\d+(?:\.\d+)?)\s*(K?)V(AC|DC|~)?(?![A-Z])/g;
    while ((m = vRe.exec(T))) vs.push({ v: parseFloat(m[2]) * (m[3] ? 1000 : 1), ac: m[1] === 'AC' || m[4] === 'AC' || m[4] === '~' });
    if (vs.length) out.volt = vs[0];
    // 허용차
    if ((m = T.match(/±\s*(\d+(?:\.\d+)?)\s*%/))) out.tolPct = parseFloat(m[1]);
    else if ((m = T.match(/(?:^|[^0-9.])(\d+(?:\.\d+)?)\s*%/))) out.tolPct = parseFloat(m[1]);
    if ((m = T.match(/(?:^|[ ,/])([BCDFGJKMZ])(?=[ ,/]|$)/))) out.tolLetter = m[1];
    else if ((m = T.match(/(?:\d)(?:PF|UF|NF)([JKMZ])(?![A-Z])/))) out.tolLetter = m[1];
    else if (out.tolLetterCap) out.tolLetter = out.tolLetterCap;
    // 저항치
    if ((m = T.match(/(?:^|[^A-Z0-9.])(\d+(?:\.\d+)?)\s*(MR|MOHM)(?![A-Z])/))) out.ohm = parseFloat(m[1]) / 1000;
    else if ((m = T.match(/(?:^|[^A-Z0-9.])(\d+(?:\.\d+)?)\s*(K|M|R|OHM|KOHM)(?:OHMS?)?(?![A-Z0-9])/))) out.ohm = parseOhm(m[1] + (m[2] === 'OHM' ? '' : m[2] === 'KOHM' ? 'K' : m[2]));
    else if ((m = T.match(/(?:^|[^A-Z0-9])(\d+)([RK])(\d+)(?![A-Z0-9])/))) out.ohm = parseOhm(m[1] + m[2] + m[3]);
    // 전력(W)
    if ((m = T.match(/(?:^|[^0-9])(1)\s*\/\s*(2|4|6|8|16)\s*W?(?![0-9])/))) out.watt = '1/' + m[2] + 'W';
    else if ((m = T.match(/(?:^|[^0-9.\/])(\d+)\s*W(?![A-Z])/))) out.watt = m[1] + 'W';
    // 크기 코드
    const metric = ['0603', '1005', '1608', '2012', '3216', '3225', '3528', '4520', '4532', '5025', '5750', '6032', '6432', '7343'];
    const inch = { '0201': '0603', '0402': '1005', '0805': '2012', '1206': '3216', '1210': '3225', '2010': '5025', '2512': '6432', '1808': '4520', '1812': '4532', '2220': '5750' };
    if ((m = T.match(/RC(0201|0402|0603|0805|1206|1210|2010|2512)/))) out.size = m[1] === '0603' ? '1608' : inch[m[1]];
    if (!out.size) for (const s of metric) if (new RegExp('(^|[^0-9])' + s + '([^0-9]|$)').test(T)) { out.size = s; break; }
    if (!out.size) for (const k in inch) if (new RegExp('(^|[^0-9])' + k + '([^0-9]|$)').test(T)) { out.size = inch[k]; break; }
    // 치수 a x b
    if ((m = T.match(/(?:Ø|Φ|D)?\s*(\d+(?:\.\d+)?)\s*[X*]\s*(\d+(?:\.\d+)?)/))) { out.dimA = parseFloat(m[1]); out.dimB = parseFloat(m[2]); }
    if ((m = T.match(/[ØΦ∮]\s*(\d+(?:\.\d+)?)/))) out.phi = parseFloat(m[1]);
    // 품번 후보 (영문+숫자 혼합, 값/단위 표현 제외)
    const stop = /^(\d+(\.\d+)?(PF|NF|UF|MF|MFD|V|KV|VAC|VDC|W|K|M|R|OHM|MM|A|MA|HZ|KHZ|MHZ|C|℃|P|PIN|L|T|EA)?|[A-Z]+|SMD|DIP|ROHS|TAPING|REEL|BULK|SOT\d+|TO\d+|DO\d+|SOD\d+|X\d+R|C0G|NP0|Y5[PVUR]|X5R|X7R|Z5U)$/;
    const toks = T.split(/[\s,;()\[\]\/:]+/).filter(Boolean);
    out.pnCandidates = toks.filter(t => /[A-Z]/.test(t) && /\d/.test(t) && t.replace(/[^A-Z0-9]/g, '').length >= 4 && !stop.test(t.replace(/-/g, '')) && !/^\d+(\.\d+)?[X*]\d/.test(t) && !/^[\d.]+(PF|NF|UF|MF|V|KV|W|K|R|MM)/.test(t));
    return out;
  }
  function pickByKw(f, T) {
    let best = null, bestLen = 0;
    for (const o of f.o || []) {
      const kws = kwList(o[2]);
      if (/vend/.test(f.k)) kws.push(...U.up(o[1]).split(/[()\s]+/).filter(w => w.length >= 3 && w !== '미지정' && w !== 'REMARK'));
      if (f.k === 'case') kws.push(...U.up(o[1]).split(/[()\/\s]+/).filter(w => w.length >= 3 && /\d/.test(w)));
      for (const kw of kws) if (hasKw(T, kw) && kw.length > bestLen) { best = o[0]; bestLen = kw.length; }
    }
    return best;
  }
  // 분류 추정 점수
  function scoreCategories(text, neighbors) {
    const T = ' ' + U.up(U.norm(text)) + ' '; const sc = {};
    for (const c of R.categories) {
      if (c.id === 'FREE') continue;
      let s = 0;
      for (const kw of kwList(c.kw)) if (hasKw(T, kw)) s += 2 + Math.min(kw.length, 12) / 6;
      if (c.itemName && T.includes(U.up(c.itemName))) s += 3;
      sc[c.id] = s;
    }
    (neighbors || []).forEach((nb, i) => { const c = catOfCode(nb.row[0]); if (c && c.id !== 'FREE') sc[c.id] = (sc[c.id] || 0) + nb.score * (i < 3 ? 4 : 2); });
    // 보정: 저항 중 칩 판별, 콘덴서 칩/리드 판별
    const ex = extract(text);
    if (isFinite(ex.ohm) && ex.watt && !ex.size) { sc.RCP = (sc.RCP || 0) - 3; }
    if (isFinite(ex.ohm) && !isFinite(ex.capPF)) { ['CA','CB','CC','CE','CM','CF','CT'].forEach(id => sc[id] = (sc[id] || 0) - 2); if (ex.size) sc.RCP = (sc.RCP || 0) + 3; else sc.RCR = (sc.RCR || 0) + 1; }
    if (isFinite(ex.capPF)) {
      ['RCP','RCR','RNA','RMX','RXC'].forEach(id => sc[id] = (sc[id] || 0) - 2);
      if (ex.capPF >= 1e6 && !/CHIP|MLCC|X7R|X5R|TANTAL/.test(T)) { sc.CE = (sc.CE || 0) + 2; if (/SMD|CHIP/.test(T)) sc.CM = (sc.CM || 0) + 3; }
      if (ex.size && !/TANTAL/.test(T)) sc.CC = (sc.CC || 0) + 2;
    }
    return Object.entries(sc).filter(([, v]) => v > 0).sort((a, b) => b[1] - a[1]);
  }
  // 분류 확정 후 필드값 채우기
  function fillFields(cat, text, neighbors) {
    const ex = extract(text); const T = ex.T; const vals = {}; const src = {}; let m0;
    const set = (k, code, raw) => { vals[k] = { code, raw: raw || '' }; src[k] = 'rule'; };
    for (const f of cat.fields) {
      if (f.t === 'sel') {
        if (f.k === 'volt' && ex.volt) {
          const o = f.o.find(o => { const v = voltOf(o[1]); return v && v.v === ex.volt.v && v.ac === ex.volt.ac; }) || f.o.find(o => { const v = voltOf(o[1]); return v && v.v === ex.volt.v; });
          if (o) set(f.k, o[0]);
        } else if (f.k === 'tol') {
          let code = null;
          if (ex.tolLetter && f.o.some(o => o[0] === ex.tolLetter)) code = ex.tolLetter;
          if (!code && isFinite(ex.tolPct)) { const o = f.o.find(o => o[1].replace(/\s/g, '').match(new RegExp('±' + ex.tolPct + '(\\.0)?%'))); if (o) code = o[0]; }
          if (code) set(f.k, code);
        } else if (cat.id === 'CT' && f.k === 'shape' && (m0 = T.match(/\(([ABCDP])\)|TANTAL-([ABCD])|CASE\s*([ABCD])/))) {
          set(f.k, 'S' + (m0[1] || m0[2] || m0[3]));
        } else if ((f.k === 'shape' || f.k === 'size') && ex.size) {
          const o = f.o.find(o => o[1].includes(ex.size)); if (o) set(f.k, o[0]);
        } else if (f.k === 'watt' && ex.watt) {
          const o = f.o.find(o => o[1].split(' ')[0] === ex.watt); if (o) set(f.k, o[0]);
        } else if (f.k === 'tc') {
          const o = f.o.find(o => hasKw(T, U.up(o[1].split(' ')[0]))) || (/NP0|NPO/.test(T) ? f.o.find(o => /C0G|NPO/.test(o[1])) : null);
          if (o) set(f.k, o[0]);
        } else if (cat.id.match(/^C[EM]$/) && f.k === 'd' && isFinite(ex.dimA)) {
          let best = null, bd = 1e9; f.o.forEach(o => { const v = parseFloat(o[1].replace('Ø', '')); if (Math.abs(v - ex.dimA) < bd) { bd = Math.abs(v - ex.dimA); best = o[0]; } });
          if (bd <= 0.6) set(f.k, best);
        } else if (cat.id.match(/^C[EM]$/) && f.k === 'len' && isFinite(ex.dimB)) {
          const L = Math.floor(ex.dimB + 0.5); const o = f.o.find(o => parseInt(o[1], 10) === L); if (o) set(f.k, o[0]);
        } else if (cat.id === 'CA' && f.k === 'dia' && isFinite(ex.phi)) {
          const o = f.o.find(o => Math.abs(parseFloat(o[1].replace('Ø', '')) - ex.phi) < 0.35); if (o) set(f.k, o[0]);
        } else {
          const c = pickByKw(f, T); if (c) set(f.k, c);
        }
      } else if (f.t === 'cap' && isFinite(ex.capPF)) {
        set(f.k, capCode(ex.capPF), capDisp(ex.capPF));
      } else if ((f.t === 'res4' || f.t === 'res3') && isFinite(ex.ohm)) {
        set(f.k, f.t === 'res4' ? res4(ex.ohm) : res3(ex.ohm), ohmDisp(ex.ohm));
      } else if (f.t === 'txt' && cat.id === 'H' && (f.k === 'dia' || f.k === 'len')) {
        let m = T.match(/M?(\d+(?:\.\d+)?)\s*[X*]\s*(\d+(?:\.\d+)?)/);
        if (m) { const v = f.k === 'dia' ? Math.round(parseFloat(m[1]) * 10) : Math.round(parseFloat(m[2])); const c = String(v).padStart(2, '0'); set(f.k, c, c); }
        else if (f.k === 'len' && (m = T.match(/(\d+(?:\.\d+)?)\s*MM/))) { const c = String(Math.round(parseFloat(m[1]))).padStart(2, '0'); set(f.k, c, c); }
        else if (f.k === 'dia' && (m = T.match(/[ØΦ]\s*(\d+(?:\.\d+)?)/))) { const c = String(Math.round(parseFloat(m[1]) * 10)).padStart(2, '0'); set(f.k, c, c); }
      } else if (f.t === 'txt' && ['pn', 'sub', 'name', 'model'].includes(f.k) && ex.pnCandidates.length) {
        set(f.k, '', ex.pnCandidates[0]);
      }
    }
    // 스크류 끝처리: T2, T3 표기
    if (cat.id === 'H') { const m = T.match(/(?:^|[^A-Z])T([1-4])[A-Z]?/); if (m) set('tip', m[1]); }
    // 내장 AI: 텍스트에 없는 항목은 유사 ERP 품목(동일 분류)의 코드값을 가중 투표로 추정
    if (neighbors && neighbors.length) {
      const same = neighbors.filter(nb => catOfCode(nb.row[0]) === cat).slice(0, 8);
      let p = cat.prefix.length;
      for (const f of cat.fields) {
        const pos = p; p += f.n;
        if (vals[f.k] || !(f.t === 'sel' || (f.t === 'txt' && f.def))) continue;
        const vote = {}; let tot = 0;
        same.forEach((nb, i) => { const seg = U.up(nb.row[0]).substr(pos, f.n); if (f.t === 'sel' && !f.o.some(o => o[0] === seg)) return; const w = nb.score * (1 / (1 + i * 0.3)); vote[seg] = (vote[seg] || 0) + w; tot += w; });
        const best = Object.entries(vote).sort((a, b) => b[1] - a[1])[0];
        if (best && tot > 0 && best[1] / tot >= 0.45) { vals[f.k] = { code: best[0], raw: f.t === 'txt' ? best[0] : '' }; src[f.k] = 'ai'; }
      }
    }
    return { vals, src, ex };
  }

  /* ---------------- 코드 조립 ---------------- */
  // vals: {k:{code, raw}} → 결과 {code, segs:[{k,l,c,ok,msg}], issues}
  function assemble(cat, vals, opts) {
    opts = opts || {};
    const segs = []; const issues = [];
    let code = cat.prefix;
    if (cat.prefix) segs.push({ k: '_p', l: '분류', c: cat.prefix, ok: true });
    for (const f of cat.fields) {
      const v = vals[f.k] || {}; let c = '', ok = true, msg = '';
      if (v.fromCode && v.code && v.code.length === f.n && f.t !== 'fix') { c = U.up(v.code); segs.push({ k: f.k, l: f.l, c, ok: true, msg: '' }); code += c; continue; }
      if (f.t === 'fix') c = f.def;
      else if (f.t === 'sel') { c = U.up(v.code != null && v.code !== '' ? v.code : (f.def || '')); }
      else if (f.t === 'cap') {
        if (v.code && /^[0-9R]{3}$/.test(U.up(v.code)) && !v.raw) c = U.up(v.code);
        else { const pf = parseCapPF(v.raw); c = (opts.capMode === 'uf' && f.alt === 'capuf') ? capCodeUF(pf) : capCode(pf); }
        if (!c) { ok = false; msg = '용량 값을 입력하세요'; }
      } else if (f.t === 'res4' || f.t === 'res3') {
        const o = parseOhm(v.raw); c = f.t === 'res4' ? res4(o) : res3(o);
        if (!c && v.code) c = U.up(v.code);
        if (!c) { ok = false; msg = '저항 값을 입력하세요'; }
      } else if (f.t === 'txt') {
        const src = v.raw != null && v.raw !== '' ? v.raw : (v.code || f.def || '');
        let s = cleanTxt(src, !!f.keepDash || /-/.test(f.def || '') && src === f.def);
        if (f.k === 'all') s = cleanTxt(src, true);
        const cp = compress(s, f.n); s = cp.v;
        if (cp.compressed) { msg = '자릿수 초과 → 자동 축약됨(확인 필요)'; issues.push(f.l + ': ' + msg); }
        if (!s && /^-+$/.test(String(src).trim())) s = '-';
        if (!s) { ok = false; msg = '값을 입력하세요'; }
        c = f.k === 'all' ? s : padField(s, f.n, '-');
      } else if (f.t === 'serial') {
        c = v.code || '';
        if (!c) { ok = false; msg = '일련번호 계산 필요'; }
      }
      if (c.length !== f.n && f.k !== 'all') { if (ok) { msg = `${f.n}자리 필요(현재 ${c.length})`; } ok = false; c = padField(c, f.n, '?'); }
      if (!ok && msg) issues.push(f.l + ': ' + msg);
      segs.push({ k: f.k, l: f.l, c, ok, msg }); code += c;
    }
    if (code.length !== cat.total) issues.push(`코드 길이 ${code.length}자리 (기준 ${cat.total}자리)`);
    return { code, segs, issues, valid: issues.filter(x => !/확인 필요/.test(x)).length === 0 };
  }
  // 규격명/품목명용 표시 데이터
  function displayData(cat, vals, asm) {
    const d = {};
    cat.fields.forEach((f, i) => {
      const seg = asm.segs.find(s => s.k === f.k) || { c: '' }; const v = vals[f.k] || {};
      let disp = v.raw || '';
      if (f.t === 'cap') { const pf = parseCapPF(v.raw || seg.c); disp = capDisp(pf); }
      if (f.t === 'res4') disp = ohmDisp(parseOhm(v.raw) || decodeRes4(seg.c));
      if (f.t === 'res3') disp = ohmDisp(parseOhm(v.raw) || decodeRes3(seg.c));
      d[f.k] = { c: seg.c, m: optMeaning(f, seg.c) || seg.c, raw: v.raw || '', disp };
    });
    return d;
  }
  function makeNames(cat, vals, asm) {
    try {
      const d = displayData(cat, vals, asm);
      const name = cat.itemNameFn ? cat.itemNameFn(d) : cat.itemName;
      const spec = cat.spec ? cat.spec(d).replace(/\s+/g, ' ').trim() : '';
      return { name: name || '', spec };
    } catch (e) { return { name: cat.itemName || '', spec: '' }; }
  }
  // 중복/유사 검사
  function checkDup(code, historyCodes) {
    code = U.up(code);
    const idx = DB.codeMap.get(code);
    const inHist = (historyCodes || []).find(h => U.up(h.code) === code);
    const near = [];
    if (code.length === LEN) {
      const stem = code.slice(0, 11);
      for (const r of DB.items) { const c = U.up(r[0]); if (c !== code && c.length === LEN && c.startsWith(stem)) { near.push(r); if (near.length >= 8) break; } }
    }
    const alts = [];
    if (idx != null || inHist) {
      const used = new Set(DB.items.map(r => U.up(r[0]))); (historyCodes || []).forEach(h => used.add(U.up(h.code)));
      for (const ch of '-ABCDEFGHJKLMNPRSTUVWXYZ0123456789') { const c = code.slice(0, -1) + ch; if (!used.has(c)) alts.push(c); if (alts.length >= 4) break; }
    }
    return { exists: idx != null, row: idx != null ? DB.items[idx] : null, inHist, near, alts };
  }

  window.ENGINE = {
    U, CAT, DB, IDX, LEN,
    parseCapPF, capCode, capCodeUF, decodeCap3, decodeCapUF, capDisp,
    parseOhm, res4, res3, decodeRes4, decodeRes3, ohmDisp,
    cleanTxt, compress, catOfCode, decode, setItems, posCount, nextSerial,
    buildIndex, similar, extract, scoreCategories, fillFields, assemble, makeNames, checkDup, optMeaning
  };
})();
