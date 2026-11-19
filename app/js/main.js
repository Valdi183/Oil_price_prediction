// Punto de entrada: carga los JSON, mantiene el estado y conecta los controles.
import { fmt, localMs, shiftMonths, prefersReducedMotion } from './format.js';
import { initTheme, palette } from './theme.js';
import { sparklineOption, mainChartOption, barsOption, testOption, rollingOption } from './charts.js';
import * as R from './render.js';
import { createTools } from './tools.js';

const HLABEL = { corto: 'Corto plazo', medio: 'Medio plazo', largo: 'Largo plazo' };

const state = {
  horizon: 'corto',
  range: '3',
  zoomWindow: null,
  selectedEventId: null,
  layers: { ma50: true, ma200: true, wti: false, events: true },
};
const data = {};
const charts = {};
let tools = null;

async function loadJson(name) {
  const res = await fetch(`data/${name}.json`, { cache: 'no-cache' });
  if (!res.ok) throw new Error(`No se pudo cargar data/${name}.json (${res.status})`);
  return res.json();
}

function initChart(id) {
  const el = document.getElementById(id);
  const chart = echarts.init(el, null, { renderer: 'canvas' });
  return chart;
}

function renderCharts() {
  const pal = palette();
  charts.spark.setOption(sparklineOption(data.history, pal), { notMerge: true });
  charts.main.setOption(mainChartOption({
    history: data.history, forecast: data.forecast, events: data.events, horizon: state.horizon,
    range: state.range, layers: state.layers, zoomWindow: state.zoomWindow, pal, selectedEventId: state.selectedEventId,
  }), { replaceMerge: ['series'] });
  charts.bars.setOption(barsOption(data.models, pal), { notMerge: true });
  charts.test.setOption(testOption(data.models, state.horizon, pal), { notMerge: true });
  charts.roll.setOption(rollingOption(data.models, state.horizon, pal), { notMerge: true });
  if (tools) tools.renderCharts();
}

function updateHorizon(h) {
  if (!HLABEL[h]) return;
  state.horizon = h;
  state.zoomWindow = null;
  state.selectedEventId = null;
  R.markSelectedCard(h);
  R.markSelectedEvent(null);
  R.renderExplanation(data.forecast, h);
  R.renderValidationHorizon(data.models, h);
  R.renderLegend(state.layers, HLABEL[h]);
  syncRangeButtons();
  renderCharts();
}

function syncRangeButtons() {
  document.querySelectorAll('#range-segmented button').forEach((b) => {
    b.setAttribute('aria-pressed', !state.zoomWindow && b.dataset.range === state.range ? 'true' : 'false');
  });
}

function selectEvent(id, { scrollChart = true, scrollCard = false } = {}) {
  const ev = data.events.find((e) => e.id === id);
  if (!ev || state.selectedEventId === id) {
    state.selectedEventId = null;
    state.zoomWindow = null;
    R.markSelectedEvent(null);
    syncRangeButtons();
    renderCharts();
    return;
  }
  state.selectedEventId = id;
  const [a, b] = ev.window;
  state.zoomWindow = [shiftMonths(localMs(a), -9), shiftMonths(localMs(b), 9)];
  if (!state.layers.events) { state.layers.events = true; document.getElementById('tg-events').checked = true; }
  R.markSelectedEvent(id, scrollCard);
  R.renderLegend(state.layers, HLABEL[state.horizon]);
  syncRangeButtons();
  renderCharts();
  if (scrollChart) {
    document.getElementById('grafico').scrollIntoView({ behavior: prefersReducedMotion() ? 'auto' : 'smooth', block: 'start' });
  }
}

function bindControls() {
  document.getElementById('horizon-segmented').addEventListener('click', (e) => {
    const b = e.target.closest('button[data-horizon]');
    if (b) updateHorizon(b.dataset.horizon);
  });
  document.getElementById('val-segmented').addEventListener('click', (e) => {
    const b = e.target.closest('button[data-horizon]');
    if (b) updateHorizon(b.dataset.horizon);
  });
  document.getElementById('test-segmented').addEventListener('click', (e) => {
    const b = e.target.closest('button[data-horizon]');
    if (b) updateHorizon(b.dataset.horizon);
  });
  const cards = document.getElementById('horizon-cards');
  cards.addEventListener('click', (e) => {
    const c = e.target.closest('.hcard');
    if (c) updateHorizon(c.dataset.horizon);
  });
  cards.addEventListener('keydown', (e) => {
    const c = e.target.closest('.hcard');
    if (c && (e.key === 'Enter' || e.key === ' ')) { e.preventDefault(); updateHorizon(c.dataset.horizon); }
  });
  document.getElementById('range-segmented').addEventListener('click', (e) => {
    const b = e.target.closest('button[data-range]');
    if (!b) return;
    state.range = b.dataset.range;
    state.zoomWindow = null;
    state.selectedEventId = null;
    R.markSelectedEvent(null);
    syncRangeButtons();
    renderCharts();
  });
  for (const [id, key] of [['tg-ma50', 'ma50'], ['tg-ma200', 'ma200'], ['tg-wti', 'wti'], ['tg-events', 'events']]) {
    document.getElementById(id).addEventListener('change', (e) => {
      state.layers[key] = e.target.checked;
      R.renderLegend(state.layers, HLABEL[state.horizon]);
      renderCharts();
    });
  }
  charts.main.on('click', (params) => {
    if (params.seriesId === 'events' && params.data && params.data.ev) {
      selectEvent(params.data.ev.id, { scrollChart: false, scrollCard: true });
    }
  });

  initTheme(document.getElementById('theme-toggle'), () => renderCharts());

  let raf = null;
  window.addEventListener('resize', () => {
    if (raf) cancelAnimationFrame(raf);
    raf = requestAnimationFrame(() => Object.values(charts).forEach((c) => c.resize()));
  });
}

function initReveal() {
  const els = document.querySelectorAll('[data-reveal]');
  if (!('IntersectionObserver' in window) || prefersReducedMotion()) {
    els.forEach((el) => el.classList.add('in'));
    return;
  }
  const io = new IntersectionObserver((entries) => {
    for (const en of entries) {
      if (en.isIntersecting) { en.target.classList.add('in'); io.unobserve(en.target); }
    }
  }, { rootMargin: '0px 0px -8% 0px', threshold: 0.08 });
  els.forEach((el) => io.observe(el));
}

function initNavSpy() {
  const links = [...document.querySelectorAll('.nav__links a')];
  const sections = links.map((a) => document.querySelector(a.getAttribute('href'))).filter(Boolean);
  if (!('IntersectionObserver' in window)) return;
  const io = new IntersectionObserver((entries) => {
    for (const en of entries) {
      if (!en.isIntersecting) continue;
      links.forEach((a) => a.classList.toggle('active', a.getAttribute('href') === `#${en.target.id}`));
    }
  }, { rootMargin: '-40% 0px -55% 0px' });
  sections.forEach((s) => io.observe(s));
}

async function main() {
  initReveal();
  initNavSpy();
  try {
    const [forecast, history, events, models, meta, toolsData] = await Promise.all(
      ['forecast', 'history', 'events', 'models', 'meta', 'tools'].map(loadJson));
    Object.assign(data, { forecast, history, events, models, meta, tools: toolsData });
  } catch (err) {
    console.error(err);
    R.showToast(`No se pudieron cargar los datos de la web. Ejecuta scripts/07_export_app_data.py. (${err.message})`);
    document.body.classList.remove('is-loading');
    return;
  }

  if (typeof echarts === 'undefined') {
    R.showToast('No se pudo cargar la librería de gráficos (ECharts). Comprueba la conexión.');
  }

  document.body.classList.remove('is-loading');
  R.renderHero(data.forecast, data.meta);
  R.renderHorizonCards(data.forecast, state.horizon);
  R.renderExplanation(data.forecast, state.horizon);
  R.renderLegend(state.layers, HLABEL[state.horizon]);
  R.renderTimeline(data.events, (id) => selectEvent(id));
  R.renderRecent(data.forecast);
  R.renderModelCards(data.models);
  R.renderWhy(data.models, data.forecast);
  R.renderSeals(data.models);
  R.renderTables(data.models);
  R.renderValidationStatic(data.models);
  R.renderValidationHorizon(data.models, state.horizon);
  R.renderMeta(data.meta);
  R.markSelectedCard(state.horizon);

  if (typeof echarts !== 'undefined') {
    charts.spark = initChart('sparkline');
    charts.main = initChart('main-chart');
    charts.bars = initChart('bars-chart');
    charts.test = initChart('test-chart');
    charts.roll = initChart('roll-chart');
    charts.sim = initChart('sim-chart');
    charts.bt = initChart('bt-chart');
    renderCharts();
  } else {
    charts.main = { on() {}, setOption() {}, resize() {} };
    charts.spark = charts.bars = charts.test = charts.roll = charts.sim = charts.bt = charts.main;
  }
  tools = createTools({ tools: data.tools, forecast: data.forecast, history: data.history, models: data.models, charts, palette });
  bindControls();
  syncRangeButtons();
  document.title = `Brent ${fmt.price(data.forecast.last_price)} · Brent Forecast`;
}

main();
