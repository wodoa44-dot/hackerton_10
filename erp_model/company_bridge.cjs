// Manual-reviewed rules run locally without browser or network APIs.
const fs = require('node:fs');
const vm = require('node:vm');
const readline = require('node:readline');
const path = require('node:path');
const rules = JSON.parse(fs.readFileSync(process.argv[2] || path.join(__dirname, '../config/company_rules.json'), 'utf8'));
const context = {window: {CODE_RULES: rules}};
vm.runInNewContext(fs.readFileSync(path.join(__dirname, 'company_engine.js'), 'utf8'), context, {timeout: 2000});
const engine = context.window.ENGINE;
readline.createInterface({input: process.stdin, crlfDelay: Infinity}).on('line', line => {
  try {
    const input = JSON.parse(line);
    const neighbors = input.neighbors || [];
    let scores = engine.scoreCategories(input.text, neighbors);
    if (input.category && engine.CAT[input.category]) scores = [[input.category, 100]];
    const proposals = [];
    for (const [id, score] of scores) {
      if (input.allowed && !input.allowed.includes(id)) continue;
      const category = engine.CAT[id];
      if (!category || id === 'FREE' || category.generation_enabled === false) continue;
      const filled = engine.fillFields(category, input.text, neighbors);
      if (id === 'DL' && /^W[0-9][A-Z0-9]{6,7}$/.test(filled.vals.pn?.raw || '')) {
        filled.vals.pn.raw = filled.vals.pn.raw.slice(1);
        filled.src.pn = 'manual_transform';
      }
      if (id === '25') {
        const series = input.text.match(/(?:SERIES|시리즈)\s*[:=]?\s*([A-Z0-9-]{2,4})(?![A-Z0-9])/i);
        const current = input.text.match(/(?:^|[\s,;])(\d+(?:\.\d+)?)\s*(MA|A)(?![A-Z0-9])/i);
        if (series && current) {
          const amps = Number(current[1]) / (current[2].toUpperCase() === 'MA' ? 1000 : 1);
          const cents = Math.round(amps * 100);
          const value = Math.floor(cents / 100) + 'A' + String(cents % 100).padStart(2,'0');
          const raw = series[1].replace(/-/g, '') + value;
          if (raw.length <= 8 && Math.abs(amps * 100 - cents) < 0.000001) {
            filled.vals.sub = {raw}; filled.src.sub = 'manual_transform';
          }
        }
      }
      if (id === '2590') {
        const forward = input.text.match(/(?:INR|CNR|SVC)?(08|10|14)D(\d{3})/i);
        const reverse = input.text.match(/(\d{3})D[- ]?(08|10|14)/i);
        if (forward || reverse) {
          filled.vals.shape = {code:(forward ? forward[1] : reverse[2]) + 'D'};
          filled.vals.pn = {raw:forward ? forward[2] : reverse[1]};
          filled.src.shape = 'rule'; filled.src.pn = 'manual_transform';
        }
      }
      const mode = input.capacity_policy?.[id]?.mode || (['CE','CM'].includes(id) ? 'uf' : 'pf');
      const assembled = engine.assemble(category, filled.vals, {capMode: mode});
      // Numerical conversion and original text candidates retain their separate origins.
      for (const field of category.fields) {
        if (['pn','sub','model'].includes(field.k) && filled.src[field.k] === 'rule') filled.src[field.k] = 'input_candidate';
      }
      const defaults = category.fields.filter(f => !filled.src[f.k] && f.t !== 'fix').map(f => f.k);
      proposals.push({id, score, prefix: category.prefix, title: category.title,
        code: assembled.code, valid: assembled.valid, issues: assembled.issues,
        segments: assembled.segs, sources: filled.src, defaults,
        capacity_mode: ['CE','CM'].includes(id) ? mode : null});
      if (proposals.length >= 8) break;
    }
    process.stdout.write(JSON.stringify({proposals}) + '\n');
  } catch (error) {
    process.stdout.write(JSON.stringify({error: error.message}) + '\n');
  }
});
