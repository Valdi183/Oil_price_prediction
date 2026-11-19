// Simulador para inversores: predicción a N días, probabilidad de precio, posición, escenarios
// y backtest de señales. Toda la estadística viene calculada en Python (data/tools.json,
// src/decision_tools.py); aquí solo se combina con lo que escribe el usuario.
import { fmt, escapeHtml } from './format.js';
import { simOption, backtestOption } from './charts.js';

const $ = (sel) => document.querySelector(sel);
const BADGE_CLASS = { 'media-alta': 'badge--alta', media: 'badge--media', baja: 'badge--baja', desconocida: 'badge--neutral' };
const HZ_NAME = { corto: 'corto plazo', medio: 'medio plazo', largo: 'largo plazo' };
const dirClass = (x) => (x > 0 ? 'is-up' : x < 0 ? 'is-down' : '');
const clamp = (x, a, b) => Math.min(b, Math.max(a, x));
const money = (x) => `${x > 0 ? '+' : x < 0 ? '−' : ''}${fmt.num(Math.abs(x), 0)} $`;
const moneyAbs = (x) => `${fmt.num(x, 0)} $`;

export function createTools({ tools, forecast, history, models, charts, palette }) {
  const S = tools.simulator;
  const L = S.levels;
  const p0 = tools.last_price;
  const state = {
    days: 21, tool: 'prob',
    probPrice: Math.round(p0 * 1.1),
    pos: { mode: 'long', qty: 10, unit: 'contracts', price: Math.round(p0 * 100) / 100 },
    bt: { h: 'corto', capital: 10000, cost: 0.05 },
  };

  // ---------------------------------------------------------------- distribución a h días
  const lrAt = (h, level) => {                      // log-rentabilidad (frente a hoy) del percentil
    const q = S.error_q[h - 1], c = S.center_lr[h - 1];
    let j = L.findIndex((x) => x >= level);
    if (j <= 0) return c + q[0];
    if (L[j] === level) return c + q[j];
    const f = (level - L[j - 1]) / (L[j] - L[j - 1]);
    return c + q[j - 1] + f * (q[j] - q[j - 1]);
  };
  const priceAt = (h, level) => p0 * Math.exp(lrAt(h, level));
  const center = (h) => p0 * Math.exp(S.center_lr[h - 1]);
  // P(precio <= x) con la distribución empírica; fuera de los percentiles 1-99 se acota.
  const cdf = (h, x) => {
    const z = Math.log(x / p0) - S.center_lr[h - 1];
    const q = S.error_q[h - 1];
    if (z <= q[0]) return { p: L[0] / 100, bound: 'lt' };
    if (z >= q[q.length - 1]) return { p: L[L.length - 1] / 100, bound: 'gt' };
    let j = q.findIndex((v) => v >= z);
    const f = (z - q[j - 1]) / (q[j] - q[j - 1] || 1);
    return { p: (L[j - 1] + f * (L[j] - L[j - 1])) / 100, bound: null };
  };
  const fmtProb = ({ p, bound }, upper = false) => {
    // upper: probabilidad complementaria (por encima)
    const v = upper ? 1 - p : p;
    const b = upper ? (bound === 'lt' ? 'gt' : bound === 'gt' ? 'lt' : null) : bound;
    if (b === 'lt') return `< ${fmt.num(100 * v, 0)} %`;
    if (b === 'gt') return `> ${fmt.num(100 * v, 0)} %`;
    return fmt.pct(100 * v, 0, false);
  };
  const anchorFor = (h) => (h <= 18 ? 'corto' : h <= 126 ? 'medio' : 'largo');
  const months = (h) => {
    const [v, one, many] = h < 21 ? [h / 5, 'semana', 'semanas'] : [h / 21, 'mes', 'meses'];
    const r = Math.round(v * 10) / 10;
    return `~${fmt.num(r, Number.isInteger(r) ? 0 : 1)} ${r === 1 ? one : many}`;
  };

  // ---------------------------------------------------------------- simulador
  function renderSim() {
    const h = state.days;
    const date = S.target_dates[h - 1];
    const pc = center(h);
    const chg = 100 * (pc / p0 - 1);
    const key = anchorFor(h);
    const rel = forecast.horizons[key].reliability;
    const direct = Object.values(S.anchors).includes(h);

    $('#sim-days-label').textContent = h;
    $('#sim-date').textContent = `· ${months(h)} · ${fmt.date(date)}`;
    $('#sim-range').value = h;
    $('#sim-range').style.setProperty('--fill', `${(100 * (h - 1) / 251).toFixed(1)}%`);
    if (document.activeElement !== $('#sim-days')) $('#sim-days').value = h;
    document.querySelectorAll('#sim-presets button').forEach((b) => b.setAttribute('aria-pressed', Number(b.dataset.days) === h ? 'true' : 'false'));

    $('#sim-title').textContent = `Brent dentro de ${h} días hábiles (${fmt.date(date)})`;
    const badge = $('#sim-badge');
    badge.className = `badge ${BADGE_CLASS[rel.level]}`;
    badge.textContent = `Fiabilidad ${rel.label}`;
    badge.title = `Fiabilidad del horizonte evaluado más cercano (${HZ_NAME[key]}, ${forecast.horizons[key].h} días)`;

    const kpi = (k, v, cls = '', sub = '', accent = false) =>
      `<div class="kpi${accent ? ' kpi--accent' : ''}"><div class="kpi__k">${k}</div><div class="kpi__v ${cls}">${v}${sub ? `<small>${sub}</small>` : ''}</div></div>`;
    $('#sim-kpis').innerHTML = [
      kpi('Precio simulado', fmt.price(pc), '', `${fmt.pct(chg)} frente a hoy`, true),
      kpi('Rango del 80 %', `${fmt.num(priceAt(h, 10), 1)}–${fmt.num(priceAt(h, 90), 1)} $`, '', '8 de cada 10 casos'),
      kpi('Rango del 90 %', `${fmt.num(priceAt(h, 5), 1)}–${fmt.num(priceAt(h, 95), 1)} $`, '', '9 de cada 10 casos'),
      kpi('Probabilidad de subida', fmtProb(cdf(h, p0), true), '', `precio > ${fmt.price(p0)}`),
    ].join('');

    const hs = Object.values(S.anchors);
    const lo = [...hs].reverse().find((x) => x < h), hi = hs.find((x) => x > h);
    const nErr = S.n_errors[key];
    $('#sim-note').innerHTML = (direct
      ? `<strong>${h} días es uno de los tres horizontes que predice directamente la red.</strong> `
      : `<strong>Interpolación</strong> entre las predicciones de la red a ${lo ? `${lo} y ` : 'hoy y '}${hi} días: no es una salida directa de la red. `)
      + `El rango sale de ${fmt.int(nErr)} errores reales del modelo fuera de muestra (${fmt.date(S.error_period[0])} → ${fmt.date(S.error_period[1])}), escalados con la raíz del tiempo. La fiabilidad es la del horizonte evaluado más cercano (${HZ_NAME[key]}).`;

    renderSimChart();
    renderTool();
  }

  function simSeries() {
    const out = { dates: S.target_dates, center: [], p5: [], p10: [], p90: [], p95: [] };
    for (let h = 1; h <= S.days.length; h++) {
      out.center.push(center(h));
      out.p5.push(priceAt(h, 5)); out.p10.push(priceAt(h, 10));
      out.p90.push(priceAt(h, 90)); out.p95.push(priceAt(h, 95));
    }
    out.anchors = Object.values(S.anchors).map((h) => ({ h, date: S.target_dates[h - 1], price: center(h) }));
    return out;
  }
  let cachedSim = null;
  function renderSimChart() {
    if (!charts.sim) return;
    cachedSim = cachedSim || simSeries();
    const h = state.days;
    charts.sim.setOption(simOption(history, cachedSim, { date: S.target_dates[h - 1], price: center(h) }, palette()), { notMerge: true });
  }

  // ---------------------------------------------------------------- pestañas
  function renderTool() {
    if (state.tool === 'prob') renderProb();
    else if (state.tool === 'pos') renderPos();
    else renderScen();
  }

  function renderProb() {
    const h = state.days;
    const x = state.probPrice;
    if (document.activeElement !== $('#prob-price')) $('#prob-price').value = x;
    if (!(x > 0)) { $('#prob-out').innerHTML = '<p class="prob__text">Escribe un precio mayor que 0.</p>'; return; }
    const c = cdf(h, x);
    const pctile = clamp(100 * c.p, 1, 99);
    const lo = Math.min(priceAt(h, 1), x, p0) * 0.98, hi = Math.max(priceAt(h, 99), x, p0) * 1.02;
    const pos = (v) => `${(100 * (v - lo) / (hi - lo)).toFixed(2)}%`;
    const band = (a, b, cls) => `<div class="distbar__band ${cls}" style="left:${pos(a)};width:calc(${pos(b)} - ${pos(a)})"></div>`;
    const chg = 100 * (x / p0 - 1);
    const pc = center(h);
    const view = x > pc ? 'más alcista' : x < pc ? 'más bajista' : 'igual';
    $('#prob-out').innerHTML = `
      <div class="prob__pair">
        <div class="prob__box prob__box--up"><div class="kpi__k">Por encima de ${fmt.price(x)}</div><div class="kpi__v">${fmtProb(c, true)}</div></div>
        <div class="prob__box prob__box--down"><div class="kpi__k">Por debajo de ${fmt.price(x)}</div><div class="kpi__v">${fmtProb(c)}</div></div>
      </div>
      <div class="distbar" aria-hidden="true">
        <div class="distbar__track"></div>
        ${band(priceAt(h, 5), priceAt(h, 95), 'distbar__band--90')}
        ${band(priceAt(h, 10), priceAt(h, 90), 'distbar__band--80')}
        <div class="distbar__mark distbar__mark--now" style="left:${pos(p0)}"></div>
        <div class="distbar__mark distbar__mark--target" style="left:${pos(x)}"></div>
        <span class="distbar__lbl" style="left:${pos(p0)}">hoy</span>
        <span class="distbar__lbl" style="left:${pos(x)};color:var(--text)">${fmt.num(x, 1)} $</span>
      </div>
      <p class="prob__text">Un Brent de <strong>${fmt.price(x)}</strong> (${fmt.pct(chg)} sobre hoy) dentro de ${h} días hábiles queda en el <strong>percentil ${fmt.num(pctile, 0)}</strong> de los escenarios del modelo. Si es tu previsión, es <strong>${view}</strong> que la del modelo (${fmt.price(pc)}).</p>
      <p class="small muted">Distribución empírica de los errores reales del modelo fuera de muestra, no una fórmula teórica. En test, el precio real cayó dentro del rango del 80 % entre el ${coverageText()} de las veces según el horizonte. Banda oscura: 80 %; clara: 90 %.</p>`;
  }

  function coverageText() {
    const cov = models.comparison.filter((r) => r.red === models.selected).map((r) => r['cobertura_IC80_%']);
    return `${fmt.num(Math.min(...cov), 0)} y el ${fmt.num(Math.max(...cov), 0)} %`;
  }

  function barrels() {
    const q = Math.max(0, Number(state.pos.qty) || 0);
    return q * (state.pos.unit === 'contracts' ? tools.contract_barrels : 1);
  }
  function posPnl(price) {
    const e = Number(state.pos.price) || p0, b = barrels();
    if (state.pos.mode === 'long') return (price - e) * b;
    if (state.pos.mode === 'short') return (e - price) * b;
    return (price - e) * b;            // compra futura: sobrecoste frente a fijar hoy el precio
  }

  function renderPos() {
    const h = state.days, { mode } = state.pos;
    const e = Number(state.pos.price) || p0, b = barrels();
    if (document.activeElement !== $('#pos-price')) $('#pos-price').value = state.pos.price;
    document.querySelectorAll('#pos-mode button').forEach((x) => x.setAttribute('aria-pressed', x.dataset.mode === mode ? 'true' : 'false'));
    $('#pos-price-label').textContent = mode === 'buy' ? 'Precio si fijas hoy ($)' : 'Precio de entrada ($)';
    $('#pos-help').textContent = {
      long: `Compras ${fmt.int(b)} barriles hoy y los vendes dentro de ${h} días hábiles: ganas si el Brent sube.`,
      short: `Vendes ${fmt.int(b)} barriles hoy y los recompras dentro de ${h} días hábiles: ganas si el Brent baja.`,
      buy: `Necesitarás comprar ${fmt.int(b)} barriles dentro de ${h} días hábiles (una aerolínea, una refinería…). ¿Cuánto te costará si esperas, frente a fijar hoy el precio con un futuro?`,
    }[mode];
    const kpi = (k, v, cls = '', sub = '', accent = false) =>
      `<div class="kpi${accent ? ' kpi--accent' : ''}"><div class="kpi__k">${k}</div><div class="kpi__v ${cls}">${v}${sub ? `<small>${sub}</small>` : ''}</div></div>`;
    const pc = center(h), p10 = priceAt(h, 10), p90 = priceAt(h, 90);
    const above = 1 - cdf(h, e).p;
    let html;
    if (mode === 'buy') {
      const bad = priceAt(h, 95);
      html = [
        kpi('Coste si esperas (central)', moneyAbs(pc * b), '', `a ${fmt.price(pc)} el barril`, true),
        kpi('Coste si fijas hoy', moneyAbs(e * b), '', `a ${fmt.price(e)} el barril`),
        kpi('Rango del 80 % si esperas', `${fmt.compact(p10 * b)}–${fmt.compact(p90 * b)} $`),
        kpi('Escenario malo (95 %)', moneyAbs(bad * b), 'is-down', `${money((bad - e) * b)} frente a fijar hoy`),
        kpi('Probabilidad de que esperar salga más caro', fmt.pct(100 * above, 0, false), above > 0.5 ? 'is-down' : 'is-up'),
        kpi('Diferencia central', money((pc - e) * b), dirClass(-(pc - e)), pc < e ? 'esperar sale más barato' : 'esperar sale más caro'),
      ];
    } else {
      const win = mode === 'long' ? above : 1 - above;
      const pnlC = posPnl(pc);
      const r = [posPnl(p10), posPnl(p90)].sort((a, c) => a - c);
      const varP = mode === 'long' ? priceAt(h, 5) : priceAt(h, 95);
      html = [
        kpi('Resultado central', money(pnlC), dirClass(pnlC), `si el Brent llega a ${fmt.price(pc)}`, true),
        kpi('Probabilidad de ganar', fmt.pct(100 * win, 0, false), win >= 0.5 ? 'is-up' : 'is-down'),
        kpi('Rango del 80 %', `${money(r[0])} / ${money(r[1])}`),
        kpi('Pérdida en el escenario malo (VaR 95 %)', money(Math.min(0, posPnl(varP))), 'is-down', `si el Brent llega a ${fmt.price(varP)}`),
        kpi('Exposición', moneyAbs(e * b), '', `${fmt.int(b)} barriles`),
        kpi('Resultado por cada 1 $ de movimiento', moneyAbs(b), '', 'sensibilidad de la posición'),
      ];
    }
    $('#pos-out').innerHTML = html.join('');
    $('#pos-note').innerHTML = 'Aproximación con el precio spot del Brent: no incluye la base de los futuros, comisiones, márgenes de garantía ni deslizamiento. Un contrato de futuros del Brent (ICE) son 1.000 barriles. <strong>No es una recomendación de inversión.</strong>';
  }

  function renderScen() {
    const h = state.days;
    const hq = tools.historical.q[h - 1], HL = tools.historical.levels;
    const g = (lv) => hq[HL.indexOf(lv)];
    const pr = (lr) => p0 * Math.exp(lr);
    const pc = (lr) => 100 * (Math.exp(lr) - 1);
    $('#scen-intro').textContent = `Plazo: ${h} días hábiles (${months(h)}). ¿Cuánto suele moverse el Brent en ese tiempo y qué pasaría si se repitiera una gran crisis?`;
    $('#scen-base').innerHTML = `Desde ${tools.historical.since.slice(0, 4)}, en 8 de cada 10 periodos de ${h} días el Brent se movió entre <strong class="is-down">${fmt.pct(pc(g(10)), 0)}</strong> y <strong class="is-up">${fmt.pct(pc(g(90)), 0)}</strong>: hoy equivaldría a <strong>${fmt.num(pr(g(10)), 1)}–${fmt.num(pr(g(90)), 1)} $</strong>. El rango del 80 % del modelo para este plazo es <strong>${fmt.num(priceAt(h, 10), 1)}–${fmt.num(priceAt(h, 90), 1)} $</strong>.`;
    const rows = tools.scenarios.map((s) => ({ ...s, lr: s.moves[h - 1], start: s.starts[h - 1] }));
    const maxAbs = Math.max(...rows.filter((r) => r.lr !== null).map((r) => Math.abs(pc(r.lr))), 1);
    const posLabel = { long: 'tu compra', short: 'tu venta', buy: 'sobrecoste' }[state.pos.mode];
    $('#scen-list').innerHTML = rows.map((r) => {
      if (r.lr === null) {
        return `<div class="scen"><div class="scen__t">${escapeHtml(r.title)}</div><div class="scen__d">Aún no hay ${h} días de datos desde esta crisis.</div><div class="scen__move muted">—</div><div class="scen__pnl muted">—</div></div>`;
      }
      const m = pc(r.lr), price = pr(r.lr), pnl = posPnl(price);
      const w = (50 * Math.abs(m) / maxAbs).toFixed(1);
      const bar = m >= 0 ? `left:50%;width:${w}%;background:var(--up)` : `right:50%;width:${w}%;background:var(--down)`;
      return `<div class="scen" title="${escapeHtml(r.text)}">
        <div class="scen__t">${escapeHtml(r.title)}</div>
        <div class="scen__d">${r.direction === 'down' ? 'Mayor caída' : 'Mayor subida'} en ${h} días, desde el ${fmt.date(r.start)}</div>
        <div class="scen__move ${dirClass(m)}">${fmt.pct(m, 0)}<small>${fmt.price(price)}</small></div>
        <div class="scen__pnl ${state.pos.mode === 'buy' ? dirClass(-pnl) : dirClass(pnl)}">${money(pnl)}<small>${posLabel}</small></div>
        <div class="scen__bar"><i style="${bar}"></i></div>
      </div>`;
    }).join('');
  }

  // ---------------------------------------------------------------- backtest
  function strategies(pal) {
    return [
      { id: 'model', name: 'Modelo (compra o venta)', short: 'Modelo', color: pal.accent, sig: (t) => t.model, main: true },
      { id: 'long', name: 'Modelo (solo compra)', short: 'Solo compra', color: pal.up, sig: (t) => (t.model > 0 ? 1 : 0) },
      { id: 'trend', name: 'Seguir la tendencia', short: 'Tendencia', color: pal.info, sig: (t) => t.trend },
      { id: 'hold', name: 'Comprar y mantener', short: 'Mantener', color: pal.muted, sig: () => 1, hold: true, dashed: true },
    ];
  }

  function runStrategy(st, trades, capital, cost) {
    let eq = capital, peak = capital, maxDD = 0, n = 0, wins = 0, worst = 0;
    const pts = [[trades[0].start, capital]];
    const years = {};
    trades.forEach((t, i) => {
      const s = st.sig(t);
      const r = Math.exp(t.ret_lr) - 1;
      const c = st.hold ? (i === 0 ? cost : 0) : (s !== 0 ? cost : 0);
      const ret = s * r - c;
      if (s !== 0 && !st.hold) { n++; if (s * r > 0) wins++; worst = Math.min(worst, ret); }
      eq *= 1 + ret;
      const y = t.start.slice(0, 4);
      years[y] = (years[y] ?? 1) * (1 + ret);
      peak = Math.max(peak, eq);
      maxDD = Math.min(maxDD, eq / peak - 1);
      pts.push([t.end, eq]);
    });
    if (st.hold) { n = 1; wins = eq > capital ? 1 : 0; worst = Math.min(0, eq / capital - 1); }
    return { ...st, points: pts, final: eq, total: 100 * (eq / capital - 1), n, hit: n ? (100 * wins) / n : null, maxDD: 100 * maxDD, worst: 100 * worst, years };
  }

  function renderBacktest() {
    const pal = palette();
    const bt = tools.backtest[state.bt.h];
    const capital = Math.max(1, Number(state.bt.capital) || 10000);
    const cost = Math.max(0, Number(state.bt.cost) || 0) / 100;
    const res = strategies(pal).map((st) => runStrategy(st, bt.trades, capital, cost));
    document.querySelectorAll('#bt-h button').forEach((b) => b.setAttribute('aria-pressed', b.dataset.bt === state.bt.h ? 'true' : 'false'));

    $('#bt-kpis').innerHTML = res.map((r) => `
      <div class="bt-card" style="--c:${r.color}">
        <div class="bt-card__name">${r.name}</div>
        <div class="bt-card__v ${dirClass(r.total)}">${fmt.pct(r.total, 0)}</div>
        <div class="bt-card__grid">
          <span>Capital final</span><b>${moneyAbs(r.final)}</b>
          <span>${r.hold ? 'Posición' : 'Operaciones'}</span><b>${r.hold ? 'siempre comprado' : fmt.int(r.n)}</b>
          <span>Aciertos</span><b>${r.hold ? '—' : fmt.pct(r.hit, 0, false)}</b>
          <span>Máxima caída</span><b>${fmt.pct(r.maxDD, 0)}</b>
        </div>
      </div>`).join('');

    if (charts.bt) charts.bt.setOption(backtestOption(res, capital, pal), { notMerge: true });

    const years = Object.keys(res[0].years).sort();
    const cell = (v) => {
      const p = 100 * (v - 1);
      const heat = Math.min(40, 6 + Math.abs(p) * 0.25).toFixed(0);
      return `<td class="${dirClass(p)} heat--${p >= 0 ? 'up' : 'down'}" style="--heat:${heat}%">${fmt.pct(p, 0)}</td>`;
    };
    $('#bt-years').innerHTML = `<thead><tr><th>Año</th>${res.map((r) => `<th>${r.short}</th>`).join('')}</tr></thead>
      <tbody>${years.map((y) => `<tr><td>${y}</td>${res.map((r) => cell(r.years[y])).join('')}</tr>`).join('')}</tbody>`;

    const m = res[0];
    const yr = years.map((y) => [y, 100 * (m.years[y] - 1)]);
    const best = yr.reduce((a, b) => (b[1] > a[1] ? b : a));
    const neg = yr.filter((x) => x[1] < 0);
    const o = bt.offsets;
    const hName = state.bt.h === 'corto' ? 'semanales' : 'trimestrales';
    $('#bt-note').innerHTML = `
      <p>Con operaciones ${hName}, la estrategia del modelo ${m.total >= 0 ? 'gana' : 'pierde'} un <strong class="${dirClass(m.total)}">${fmt.pct(m.total, 0)}</strong> frente al <strong>${fmt.pct(res[3].total, 0)}</strong> de comprar y mantener, acertando la dirección en el ${fmt.pct(m.hit, 0, false)} de ${fmt.int(m.n)} operaciones.</p>
      <p><strong>Ojo con la lectura.</strong> El mejor año es ${best[0]} (${fmt.pct(best[1], 0)})${best[0] === '2026' ? ', con los vaivenes de la crisis de Ormuz' : ''}${neg.length ? `; ${neg.map((x) => `en ${x[0]} perdía (${fmt.pct(x[1], 0)})`).join(', ')}` : ''}. El resultado depende del día en que se empiece: con los ${o.n} días de inicio posibles y sin costes, la estrategia del modelo va de <strong>${fmt.pct(o.min, 0)}</strong> a <strong>${fmt.pct(o.max, 0)}</strong> (mediana ${fmt.pct(o.median, 0)}).${state.bt.h === 'medio' ? ` Con solo ${fmt.int(bt.trades.length)} operaciones, la muestra es muy pequeña.` : ''}</p>
      <p class="warn">Resultado hipotético: precio spot (en la práctica se operaría con futuros), sin deslizamiento, apalancamiento ni márgenes. Rentabilidades pasadas no garantizan resultados futuros. No es una recomendación de inversión.</p>`;
  }

  // ---------------------------------------------------------------- controles
  function setDays(v) {
    const h = clamp(Math.round(Number(v) || 1), 1, S.days.length);
    if (h === state.days) return;
    state.days = h;
    renderSim();
  }

  function bind() {
    $('#sim-range').addEventListener('input', (e) => setDays(e.target.value));
    $('#sim-days').addEventListener('input', (e) => { if (e.target.value !== '') setDays(e.target.value); });
    $('#sim-days').addEventListener('blur', (e) => { e.target.value = state.days; });
    $('#sim-presets').addEventListener('click', (e) => { const b = e.target.closest('button[data-days]'); if (b) setDays(b.dataset.days); });

    $('#tool-tabs').addEventListener('click', (e) => {
      const b = e.target.closest('button[data-tool]');
      if (!b) return;
      state.tool = b.dataset.tool;
      document.querySelectorAll('#tool-tabs button').forEach((x) => x.setAttribute('aria-selected', x === b ? 'true' : 'false'));
      for (const t of ['prob', 'pos', 'scen']) $(`#tool-${t}`).hidden = t !== state.tool;
      renderTool();
    });

    $('#prob-price').addEventListener('input', (e) => { state.probPrice = Number(e.target.value); renderProb(); });
    $('#prob-presets').addEventListener('click', (e) => {
      const b = e.target.closest('button[data-pct]');
      if (!b) return;
      state.probPrice = Math.round(p0 * (1 + Number(b.dataset.pct) / 100) * 2) / 2;
      $('#prob-price').value = state.probPrice;
      renderProb();
    });

    $('#pos-mode').addEventListener('click', (e) => { const b = e.target.closest('button[data-mode]'); if (b) { state.pos.mode = b.dataset.mode; renderPos(); } });
    $('#pos-qty').addEventListener('input', (e) => { state.pos.qty = e.target.value; renderPos(); });
    $('#pos-unit').addEventListener('change', (e) => { state.pos.unit = e.target.value; renderPos(); });
    $('#pos-price').addEventListener('input', (e) => { state.pos.price = e.target.value; renderPos(); });

    $('#bt-h').addEventListener('click', (e) => { const b = e.target.closest('button[data-bt]'); if (b) { state.bt.h = b.dataset.bt; renderBacktest(); } });
    $('#bt-capital').addEventListener('input', (e) => { state.bt.capital = e.target.value; renderBacktest(); });
    $('#bt-cost').addEventListener('input', (e) => { state.bt.cost = e.target.value; renderBacktest(); });
  }

  bind();
  renderSim();
  renderBacktest();
  return {
    renderCharts() { renderSimChart(); renderBacktest(); },
  };
}
