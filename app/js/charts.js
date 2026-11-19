// Gráficos con Apache ECharts: minigráfico, gráfico principal, barras y real vs predicho.
import { fmt, stars, pValue, localMs, isoLocal, shiftMonths, prefersReducedMotion, escapeHtml } from './format.js';

const HLABEL = { corto: 'Corto plazo', medio: 'Medio plazo', largo: 'Largo plazo' };
const MODEL_COLOR = (pal) => ({ RNN: pal.accent, LSTM: pal.lstm, GRU: pal.gru, 'Combinación': pal.combo });

export function tooltipBase(pal) {
  return {
    backgroundColor: pal.tooltipBg,
    borderColor: pal.borderStrong,
    borderWidth: 1,
    padding: [10, 12],
    textStyle: { color: pal.text, fontFamily: pal.fontUi, fontSize: 12 },
    extraCssText: 'box-shadow: 0 12px 30px rgba(0,0,0,.35); border-radius: 10px; backdrop-filter: blur(8px);',
    transitionDuration: 0.15,
  };
}

export function axisBase(pal) {
  return {
    axisLine: { lineStyle: { color: pal.border } },
    axisTick: { show: false },
    axisLabel: { color: pal.muted, fontFamily: pal.fontMono, fontSize: 11 },
    splitLine: { lineStyle: { color: pal.grid } },
  };
}

export function hexToRgba(hex, a) {
  const h = hex.replace('#', '');
  if (h.length !== 6) return hex;
  const n = parseInt(h, 16);
  return `rgba(${(n >> 16) & 255}, ${(n >> 8) & 255}, ${n & 255}, ${a})`;
}

export function row(color, name, value, dash = false) {
  const sw = dash
    ? `style="background: repeating-linear-gradient(90deg, ${color} 0 3px, transparent 3px 5px); width: 12px"`
    : `style="background:${color}"`;
  return `<div class="bf-tip__row"><span><i class="bf-tip__sw" ${sw}></i>${name}</span><b>${value}</b></div>`;
}

export function eventTooltipHtml(ev) {
  const kind = ev.kind === 'max' ? 'Pico' : 'Caída';
  return `<div class="bf-tip bf-tip__ev"><div class="bf-tip__date">${kind} · ${fmt.date(ev.date)} · ${fmt.price(ev.price)}</div>` +
    `<h4>${escapeHtml(ev.title)}</h4><span class="bf-tip__cycle">${escapeHtml(ev.cycle)}</span><p>${escapeHtml(ev.text)}</p></div>`;
}

// ------------------------------------------------------------------ minigráfico
export function sparklineOption(history, pal) {
  const n = Math.min(252, history.dates.length);
  const dates = history.dates.slice(-n);
  const vals = history.brent.slice(-n);
  const min = Math.min(...vals), max = Math.max(...vals);
  const last = vals[vals.length - 1], first = vals[0];
  const color = last >= first ? pal.up : pal.down;
  return {
    animation: !prefersReducedMotion(), animationDuration: 1200, animationEasing: 'cubicOut',
    grid: { left: 4, right: 4, top: 8, bottom: 4 },
    xAxis: { type: 'category', data: dates, show: false, boundaryGap: false },
    yAxis: { type: 'value', show: false, min: min - (max - min) * 0.1, max: max + (max - min) * 0.1 },
    tooltip: {
      ...tooltipBase(pal), trigger: 'axis', axisPointer: { type: 'line', lineStyle: { color: pal.muted } },
      formatter: (p) => `<div class="bf-tip"><div class="bf-tip__date">${fmt.date(p[0].axisValue)}</div>${row(color, 'Brent', fmt.price(p[0].value))}</div>`,
    },
    series: [{
      type: 'line', data: vals, showSymbol: false, smooth: 0.15,
      lineStyle: { width: 2, color },
      areaStyle: { color: { type: 'linear', x: 0, y: 0, x2: 0, y2: 1, colorStops: [{ offset: 0, color: hexToRgba(color, 0.35) }, { offset: 1, color: hexToRgba(color, 0) }] } },
      markPoint: {
        symbol: 'circle', symbolSize: 8, silent: true,
        data: [{ coord: [n - 1, last], itemStyle: { color, borderColor: pal.surface, borderWidth: 2 } }],
        label: { show: false },
      },
    }],
  };
}

// ------------------------------------------------------------------ gráfico principal
function fanPoints(lastDate, p0, h) {
  // Abanico: el centro sigue la trayectoria lineal hasta la predicción y la anchura crece
  // con la raíz del tiempo (como la incertidumbre de un paseo aleatorio).
  const t0 = localMs(lastDate);
  const t1 = localMs(h.target_date);
  const steps = 24;
  const low = [], band = [], center = [];
  for (let i = 0; i <= steps; i++) {
    const f = i / steps;
    const t = t0 + (t1 - t0) * f;
    const c = p0 + (h.price - p0) * f;
    const w = Math.sqrt(f);
    const lo = c - (h.price - h.low) * w;
    const hi = c + (h.high - h.price) * w;
    low.push([t, lo]);
    band.push([t, hi - lo]);
    center.push([t, c]);
  }
  return { low, band, center };
}

export function mainChartOption(ctx) {
  const { history, forecast, events, horizon, range, layers, zoomWindow, pal, selectedEventId } = ctx;
  const h = forecast.horizons[horizon];
  const lastDate = forecast.last_date;
  const p0 = forecast.last_price;
  const lastT = localMs(lastDate);
  const targetT = localMs(h.target_date);

  let min = null, max = null;
  if (zoomWindow) {
    [min, max] = zoomWindow;
  } else if (range === 'all') {
    max = shiftMonths(targetT, 2);
  } else {
    min = shiftMonths(lastT, -12 * Number(range));
    const span = targetT - min;
    max = targetT + span * 0.04;
  }

  const pair = (arr) => history.dates.map((d, i) => [d, arr[i]]);
  const fan = fanPoints(lastDate, p0, h);
  const series = [];

  // Abanico de incertidumbre (banda apilada: base invisible + anchura con degradado)
  series.push({
    id: 'fan-low', name: 'fan-low', type: 'line', data: fan.low, stack: 'fan', symbol: 'none', silent: true,
    lineStyle: { opacity: 0 }, areaStyle: { opacity: 0 }, z: 1, emphasis: { disabled: true },
  });
  series.push({
    id: 'fan', name: 'Rango 80 %', type: 'line', data: fan.band, stack: 'fan', symbol: 'none', silent: true, z: 1,
    lineStyle: { opacity: 0 }, emphasis: { disabled: true },
    areaStyle: {
      color: {
        type: 'linear', x: 0, y: 0, x2: 1, y2: 0,
        colorStops: [{ offset: 0, color: hexToRgba(pal.accent, 0.45) }, { offset: 1, color: hexToRgba(pal.accent, 0.12) }],
      },
    },
  });

  // Histórico
  series.push({
    id: 'brent', name: 'Brent', type: 'line', data: pair(history.brent), showSymbol: false, sampling: 'lttb',
    lineStyle: { width: 1.7, color: pal.accent }, itemStyle: { color: pal.accent }, z: 5,
    emphasis: { lineStyle: { width: 2.2 } },
    markLine: {
      silent: true, symbol: 'none', animation: false,
      lineStyle: { color: pal.muted, type: 'dashed', width: 1, opacity: 0.7 },
      label: { show: true, formatter: 'Último dato', position: 'insideEndTop', color: pal.muted, fontSize: 10, fontFamily: pal.fontMono },
      data: [{ xAxis: lastDate }],
    },
  });
  if (layers.ma50) {
    series.push({ id: 'ma50', name: 'Media 50 d', type: 'line', data: pair(history.ma50), showSymbol: false, sampling: 'lttb',
      lineStyle: { width: 1, color: pal.ma50, type: [2, 4], opacity: 0.9 }, itemStyle: { color: pal.ma50 }, z: 3, emphasis: { disabled: true } });
  }
  if (layers.ma200) {
    series.push({ id: 'ma200', name: 'Media 200 d', type: 'line', data: pair(history.ma200), showSymbol: false, sampling: 'lttb',
      lineStyle: { width: 1, color: pal.ma200, type: [2, 4], opacity: 0.9 }, itemStyle: { color: pal.ma200 }, z: 3, emphasis: { disabled: true } });
  }
  if (layers.wti) {
    series.push({ id: 'wti', name: 'WTI', type: 'line', data: pair(history.wti), showSymbol: false, sampling: 'lttb',
      lineStyle: { width: 1, color: pal.wti, opacity: 0.8 }, itemStyle: { color: pal.wti }, z: 2, emphasis: { disabled: true } });
  }

  // Otros horizontes, atenuados
  for (const [key, other] of Object.entries(forecast.horizons)) {
    if (key === horizon) continue;
    series.push({
      id: `pred-${key}`, name: `${HLABEL[key]} (otro horizonte)`, type: 'line',
      data: [[lastDate, p0], [other.target_date, other.price]], symbol: 'circle', symbolSize: 5,
      lineStyle: { width: 1, type: 'dashed', color: pal.muted, opacity: 0.45 }, itemStyle: { color: pal.muted, opacity: 0.6 },
      z: 4, emphasis: { disabled: true }, silent: true,
    });
  }

  // Predicción del horizonte elegido
  const up = h.price >= p0;
  series.push({
    id: 'pred', name: `Predicción ${HLABEL[horizon].toLowerCase()}`, type: 'line',
    data: [[lastDate, p0], [h.target_date, h.price]],
    symbol: 'circle', symbolSize: (v, p) => (p.dataIndex === 1 ? 11 : 0),
    lineStyle: { width: 2.4, type: [6, 5], color: pal.accent2 },
    itemStyle: { color: up ? pal.up : pal.down, borderColor: pal.surface, borderWidth: 2 },
    label: {
      show: true, position: 'right', fontFamily: pal.fontMono, fontWeight: 600, fontSize: 12, color: pal.text,
      backgroundColor: pal.surface, borderColor: pal.borderStrong, borderWidth: 1, borderRadius: 6, padding: [4, 7],
      formatter: (p) => (p.dataIndex === 1 ? `${fmt.price(p.value[1])}  ${fmt.pct(h.change_pct)}` : ''),
    },
    z: 8,
  });
  // Extremos del rango en la fecha objetivo
  series.push({
    id: 'pred-range', name: 'Extremos del rango', type: 'scatter', silent: true,
    data: [[h.target_date, h.low], [h.target_date, h.high]], symbol: 'rect', symbolSize: [10, 2],
    itemStyle: { color: pal.accent }, z: 7,
    label: {
      show: true, position: 'right', color: pal.muted, fontFamily: pal.fontMono, fontSize: 10,
      formatter: (p) => fmt.price(p.value[1]),
    },
  });

  // Eventos del ciclo económico
  if (layers.events) {
    series.push({
      id: 'events', name: 'Eventos', type: 'scatter', z: 9,
      data: events.map((ev) => ({
        value: [ev.date, ev.price], ev,
        symbol: 'triangle', symbolRotate: ev.kind === 'max' ? 0 : 180,
        symbolSize: ev.id === selectedEventId ? 18 : 13,
        symbolOffset: [0, ev.kind === 'max' ? -14 : 14],
        itemStyle: {
          color: ev.kind === 'max' ? pal.down : pal.up, borderColor: pal.surface, borderWidth: 1.5,
          shadowBlur: ev.id === selectedEventId ? 12 : 0, shadowColor: ev.kind === 'max' ? pal.down : pal.up,
        },
      })),
      emphasis: { scale: 1.5 },
      tooltip: { trigger: 'item', formatter: (p) => eventTooltipHtml(p.data.ev) },
    });
  }

  const formatter = (params) => {
    if (!Array.isArray(params)) {
      return params.data && params.data.ev ? eventTooltipHtml(params.data.ev) : '';
    }
    const first = params[0];
    const date = typeof first.axisValue === 'number' ? isoLocal(first.axisValue) : first.axisValue;
    const lines = [];
    let fanLow = null, fanWidth = null;
    for (const p of params) {
      const v = Array.isArray(p.value) ? p.value[1] : p.value;
      if (v === null || v === undefined) continue;
      if (p.seriesId === 'fan-low') { fanLow = v; continue; }
      if (p.seriesId === 'fan') { fanWidth = v; continue; }
      if (p.seriesId === 'events' || p.seriesId === 'pred-range') continue;
      const dash = p.seriesId.startsWith('pred') || p.seriesId.startsWith('ma');
      lines.push(row(p.color, p.seriesName, fmt.price(v), dash));
    }
    if (fanLow !== null && fanWidth !== null && fanWidth > 0.01) {
      lines.push(row(pal.accent, 'Rango 80 %', `${fmt.price(fanLow)} – ${fmt.price(fanLow + fanWidth)}`));
    }
    if (!lines.length) return '';
    return `<div class="bf-tip"><div class="bf-tip__date">${fmt.date(date)}</div>${lines.join('')}</div>`;
  };

  return {
    animation: !prefersReducedMotion(), animationDuration: 900, animationDurationUpdate: 500, animationEasing: 'cubicOut',
    grid: { left: 10, right: 92, top: 24, bottom: 36, containLabel: true },
    tooltip: {
      ...tooltipBase(pal), trigger: 'axis', formatter,
      axisPointer: {
        type: 'cross', snap: false,
        lineStyle: { color: pal.muted, type: 'dashed', opacity: 0.8 },
        crossStyle: { color: pal.muted, opacity: 0.6 },
        label: { backgroundColor: pal.surface, color: pal.text, borderColor: pal.borderStrong, borderWidth: 1, fontFamily: pal.fontMono, fontSize: 10,
          formatter: (p) => (p.axisDimension === 'x' ? fmt.date(isoLocal(p.value)) : fmt.price(p.value)) },
      },
    },
    xAxis: { type: 'time', min, max, ...axisBase(pal), splitLine: { show: false }, axisLabel: { ...axisBase(pal).axisLabel, hideOverlap: true } },
    yAxis: { type: 'value', scale: true, ...axisBase(pal), axisLine: { show: false }, axisLabel: { ...axisBase(pal).axisLabel, formatter: (v) => fmt.price(v, 0) } },
    dataZoom: [{ type: 'inside', start: 0, end: 100, zoomOnMouseWheel: 'ctrl', moveOnMouseMove: true, moveOnMouseWheel: false }],
    series,
  };
}

// ------------------------------------------------------------------ barras: mejora vs ingenuo
export function barsOption(models, pal) {
  const colors = MODEL_COLOR(pal);
  const horizons = ['corto', 'medio', 'largo'];
  const cats = ['Corto (5 d)', 'Medio (63 d)', 'Largo (252 d)'];
  const nets = ['RNN', 'LSTM', 'GRU', 'Combinación'];
  const get = (net, hz) => models.comparison.find((r) => r.red === net && r.horizonte === hz);
  const sig = (net, hz) => models.significance.find((r) => r.modelo === net && r.frente_a === 'Ingenuo' && r.horizonte === hz);
  const series = nets.map((net) => ({
    name: net, type: 'bar', barGap: '12%', barCategoryGap: '34%',
    data: horizons.map((hz) => {
      const r = get(net, hz), s = sig(net, hz);
      return { value: r ? r['mejora_RMSE_%'] : null, p: s ? s.p_valor : null, std: r ? r['std_mejora_RMSE_%'] : null };
    }),
    itemStyle: { color: colors[net], borderRadius: [4, 4, 0, 0], opacity: net === 'Combinación' ? 0.45 : net === 'RNN' ? 1 : 0.85 },
    label: {
      show: true, position: 'top', fontFamily: pal.fontMono, fontSize: 11, color: pal.accent,
      formatter: (p) => stars(p.data.p),
    },
    emphasis: { focus: 'series' },
  }));
  return {
    animation: !prefersReducedMotion(), animationDuration: 800,
    color: nets.map((n) => colors[n]),
    grid: { left: 8, right: 8, top: 40, bottom: 8, containLabel: true },
    legend: { top: 0, textStyle: { color: pal.text2, fontFamily: pal.fontUi, fontSize: 11 }, itemWidth: 12, itemHeight: 8, icon: 'roundRect' },
    tooltip: {
      ...tooltipBase(pal), trigger: 'axis', axisPointer: { type: 'shadow', shadowStyle: { color: hexToRgba(pal.text === '#121826' ? '#101828' : '#ffffff', 0.04) } },
      formatter: (params) => {
        const lines = params.map((p) => row(p.color, p.seriesName, `${fmt.pct(p.value)} ${p.data.p !== null ? `· ${pValue(p.data.p)} ${stars(p.data.p)}` : ''}`));
        return `<div class="bf-tip"><div class="bf-tip__date">${params[0].axisValue} · mejora del RMSE vs ingenuo</div>${lines.join('')}</div>`;
      },
    },
    xAxis: { type: 'category', data: cats, ...axisBase(pal), splitLine: { show: false }, axisLabel: { color: pal.text2, fontFamily: pal.fontUi, fontSize: 12 } },
    yAxis: { type: 'value', ...axisBase(pal), axisLine: { show: false }, axisLabel: { ...axisBase(pal).axisLabel, formatter: (v) => fmt.pct(v, 0, false) } },
    series,
  };
}

// ------------------------------------------------------------------ real vs predicho en test
export function testOption(models, horizon, pal) {
  const colors = MODEL_COLOR(pal);
  const tp = models.test_predictions;
  const block = tp[horizon];
  const dates = tp.dates;
  const mk = (name, vals, style) => ({
    name, type: 'line', data: dates.map((d, i) => [d, vals[i]]), showSymbol: false, sampling: 'lttb', ...style,
  });
  const series = [
    mk('Real', block.real, { lineStyle: { width: 2.2, color: pal.text }, itemStyle: { color: pal.text }, z: 6 }),
    mk('RNN', block.RNN, { lineStyle: { width: 1.8, color: colors.RNN }, itemStyle: { color: colors.RNN }, z: 5 }),
    mk('LSTM', block.LSTM, { lineStyle: { width: 1.1, color: colors.LSTM, opacity: 0.85 }, itemStyle: { color: colors.LSTM }, z: 4 }),
    mk('GRU', block.GRU, { lineStyle: { width: 1.1, color: colors.GRU, opacity: 0.85 }, itemStyle: { color: colors.GRU }, z: 4 }),
    mk('Combinación (referencia)', block['Combinación'], { lineStyle: { width: 1, color: colors['Combinación'], type: 'dashed', opacity: 0.8 }, itemStyle: { color: colors['Combinación'] }, z: 3 }),
  ];
  return {
    animation: !prefersReducedMotion(), animationDuration: 800,
    grid: { left: 8, right: 16, top: 36, bottom: 54, containLabel: true },
    legend: { top: 0, textStyle: { color: pal.text2, fontFamily: pal.fontUi, fontSize: 11 }, itemWidth: 14, itemHeight: 2, selected: { 'Combinación (referencia)': false } },
    tooltip: {
      ...tooltipBase(pal), trigger: 'axis', axisPointer: { type: 'cross', lineStyle: { color: pal.muted, type: 'dashed' }, crossStyle: { color: pal.muted },
        label: { backgroundColor: pal.surface, color: pal.text, borderColor: pal.borderStrong, borderWidth: 1, fontFamily: pal.fontMono, fontSize: 10,
          formatter: (p) => (p.axisDimension === 'x' ? fmt.date(isoLocal(p.value)) : fmt.price(p.value)) } },
      formatter: (params) => {
        const date = isoLocal(params[0].axisValue);
        const lines = params.filter((p) => p.value[1] !== null).map((p) => row(p.color, p.seriesName, fmt.price(p.value[1]), p.seriesName.startsWith('Combinación')));
        return `<div class="bf-tip"><div class="bf-tip__date">Origen ${fmt.date(date)} · objetivo a ${horizon === 'corto' ? '5' : horizon === 'medio' ? '63' : '252'} días</div>${lines.join('')}</div>`;
      },
    },
    xAxis: { type: 'time', ...axisBase(pal), splitLine: { show: false } },
    yAxis: { type: 'value', scale: true, ...axisBase(pal), axisLine: { show: false }, axisLabel: { ...axisBase(pal).axisLabel, formatter: (v) => fmt.price(v, 0) } },
    dataZoom: [
      { type: 'inside', start: 0, end: 100 },
      { type: 'slider', height: 18, bottom: 8, borderColor: pal.border, backgroundColor: 'transparent', fillerColor: hexToRgba(pal.accent, 0.12),
        handleStyle: { color: pal.accent, borderColor: pal.accent }, moveHandleStyle: { color: pal.borderStrong },
        dataBackground: { lineStyle: { color: pal.muted, opacity: 0.4 }, areaStyle: { color: pal.muted, opacity: 0.1 } },
        selectedDataBackground: { lineStyle: { color: pal.accent, opacity: 0.6 }, areaStyle: { color: pal.accent, opacity: 0.15 } },
        textStyle: { color: pal.muted, fontFamily: pal.fontMono, fontSize: 10 }, labelFormatter: (v) => fmt.date(isoLocal(v)) },
    ],
    series,
  };
}

// ------------------------------------------------------------------ error móvil frente al ingenuo
export function rollingOption(models, horizon, pal) {
  const dates = models.test_predictions.dates;
  const r = models.validation.rolling[horizon];
  const data = dates.map((d, i) => [d, r.ratio[i]]).filter((p) => p[1] !== null);
  // Degradado vertical: rojo por encima de 1 (peor que el ingenuo) y verde por debajo.
  const ys = data.map((p) => p[1]);
  const ymax = Math.ceil(Math.max(...ys, 1) * 20) / 20, ymin = Math.floor(Math.min(...ys, 1) * 20) / 20;
  const cut = ymax === ymin ? 0.5 : (ymax - 1) / (ymax - ymin);
  const grad = (a) => new echarts.graphic.LinearGradient(0, 0, 0, 1, [
    { offset: 0, color: hexToRgba(pal.down, a) }, { offset: cut, color: hexToRgba(pal.down, a) },
    { offset: cut, color: hexToRgba(pal.up, a) }, { offset: 1, color: hexToRgba(pal.up, a) }]);
  return {
    animation: !prefersReducedMotion(), animationDuration: 800,
    grid: { left: 8, right: 16, top: 16, bottom: 8, containLabel: true },
    tooltip: {
      ...tooltipBase(pal), trigger: 'axis',
      formatter: (params) => {
        const p = params.find((x) => x.seriesIndex === 0);
        if (!p || p.value[1] === null) return '';
        const ok = p.value[1] < 1;
        return `<div class="bf-tip"><div class="bf-tip__date">Ventana hasta ${fmt.date(isoLocal(p.axisValue))}</div>`
          + row(ok ? pal.up : pal.down, 'RMSE RNN / ingenuo', fmt.num(p.value[1], 3))
          + `<div class="bf-tip__row"><span>La RNN se equivoca un</span><b>${fmt.pct(Math.abs(100 * (p.value[1] - 1)), 1, false)} ${ok ? 'menos' : 'más'}</b></div></div>`;
      },
    },
    xAxis: { type: 'time', ...axisBase(pal), splitLine: { show: false } },
    yAxis: { type: 'value', min: ymin, max: ymax, ...axisBase(pal), axisLine: { show: false }, axisLabel: { ...axisBase(pal).axisLabel, formatter: (v) => fmt.num(v, 2) } },
    series: [{
      name: 'RMSE RNN / ingenuo', type: 'line', data, showSymbol: false,
      lineStyle: { width: 2, color: grad(1) }, itemStyle: { color: pal.accent },
      areaStyle: { color: grad(0.14), origin: 'start' },
    }, {
      name: 'Ingenuo = 1', type: 'line', data: [[dates[0], 1], [dates[dates.length - 1], 1]], showSymbol: false, silent: true,
      lineStyle: { color: pal.muted, type: 'dashed', width: 1 }, tooltip: { show: false },
    }],
  };
}

// ------------------------------------------------------------------ simulador: trayectoria y rangos
// sim: { dates, center, p5, p10, p90, p95, anchors: [{date, price, h}] } ya calculados (tools.js)
export function simOption(history, sim, sel, pal) {
  const n = 130;
  const hd = history.dates.slice(-n), hb = history.brent.slice(-n);
  const lastDate = hd[hd.length - 1], p0 = hb[hb.length - 1];
  const d = [lastDate, ...sim.dates];
  const z = (arr) => [p0, ...arr];
  const c = z(sim.center), p5 = z(sim.p5), p10 = z(sim.p10), p90 = z(sim.p90), p95 = z(sim.p95);
  const pts = (arr) => d.map((x, i) => [x, arr[i]]);
  const width = (hi, lo) => d.map((x, i) => [x, hi[i] - lo[i]]);
  const base = { type: 'line', symbol: 'none', silent: true, emphasis: { disabled: true }, lineStyle: { opacity: 0 } };
  const series = [
    { ...base, id: 'b90-low', name: 'b90-low', stack: 'b90', data: pts(p5), areaStyle: { opacity: 0 }, z: 1 },
    { ...base, id: 'b90', name: 'Rango 90 %', stack: 'b90', data: width(p95, p5), z: 1, areaStyle: { color: hexToRgba(pal.accent, 0.12) } },
    { ...base, id: 'b80-low', name: 'b80-low', stack: 'b80', data: pts(p10), areaStyle: { opacity: 0 }, z: 2 },
    { ...base, id: 'b80', name: 'Rango 80 %', stack: 'b80', data: width(p90, p10), z: 2, areaStyle: { color: hexToRgba(pal.accent, 0.26) } },
    { id: 'hist', name: 'Brent', type: 'line', data: hd.map((x, i) => [x, hb[i]]), showSymbol: false,
      lineStyle: { width: 1.8, color: pal.accent }, itemStyle: { color: pal.accent }, z: 5 },
    { id: 'center', name: 'Predicción simulada', type: 'line', data: pts(c), showSymbol: false,
      lineStyle: { width: 2, type: [6, 4], color: pal.accent2 }, itemStyle: { color: pal.accent2 }, z: 6,
      markLine: { silent: true, symbol: 'none', animation: false, lineStyle: { color: pal.text, type: 'solid', width: 1, opacity: 0.5 },
        label: { show: false }, data: [{ xAxis: sel.date }] } },
    { id: 'anchors', name: 'Salida directa de la red', type: 'scatter', data: sim.anchors.map((a) => [a.date, a.price]),
      symbol: 'diamond', symbolSize: 9, z: 7, itemStyle: { color: pal.surface, borderColor: pal.accent2, borderWidth: 2 } },
    { id: 'sel', name: 'Plazo elegido', type: 'scatter', data: [[sel.date, sel.price]], symbolSize: 13, z: 9,
      itemStyle: { color: sel.price >= p0 ? pal.up : pal.down, borderColor: pal.surface, borderWidth: 2 },
      label: { show: true, position: 'top', distance: 8, fontFamily: pal.fontMono, fontWeight: 600, fontSize: 12, color: pal.text,
        backgroundColor: pal.surface, borderColor: pal.borderStrong, borderWidth: 1, borderRadius: 6, padding: [3, 6],
        formatter: () => fmt.price(sel.price) } },
  ];
  return {
    animation: !prefersReducedMotion(), animationDuration: 500, animationDurationUpdate: 300,
    grid: { left: 8, right: 18, top: 28, bottom: 8, containLabel: true },
    tooltip: {
      ...tooltipBase(pal), trigger: 'axis', axisPointer: { type: 'line', lineStyle: { color: pal.muted, type: 'dashed' } },
      formatter: (params) => {
        const date = isoLocal(params[0].axisValue);
        const k = d.indexOf(date);          // > 0: día de la simulación; si no, histórico
        if (k > 0) {
          return `<div class="bf-tip"><div class="bf-tip__date">${fmt.date(date)} · ${k} días hábiles</div>`
            + row(pal.accent2, 'Predicción simulada', fmt.price(c[k]), true)
            + row(pal.accent, 'Rango 80 %', `${fmt.price(p10[k])} – ${fmt.price(p90[k])}`)
            + row(hexToRgba(pal.accent, 0.4), 'Rango 90 %', `${fmt.price(p5[k])} – ${fmt.price(p95[k])}`) + '</div>';
        }
        const v = params.find((p) => p.seriesId === 'hist');
        return v ? `<div class="bf-tip"><div class="bf-tip__date">${fmt.date(date)}</div>${row(pal.accent, 'Brent', fmt.price(v.value[1]))}</div>` : '';
      },
    },
    legend: { show: false },
    xAxis: { type: 'time', ...axisBase(pal), splitLine: { show: false }, axisLabel: { ...axisBase(pal).axisLabel, hideOverlap: true } },
    yAxis: { type: 'value', scale: true, ...axisBase(pal), axisLine: { show: false }, axisLabel: { ...axisBase(pal).axisLabel, formatter: (v) => fmt.price(v, 0) } },
    series,
  };
}

// ------------------------------------------------------------------ backtest: evolución del capital
// lines: [{ name, color, points: [[fecha, capital]], dashed }]
export function backtestOption(lines, capital, pal) {
  return {
    animation: !prefersReducedMotion(), animationDuration: 700,
    grid: { left: 8, right: 16, top: 36, bottom: 8, containLabel: true },
    legend: { top: 0, data: lines.map((l) => l.name), textStyle: { color: pal.text2, fontFamily: pal.fontUi, fontSize: 11 }, itemWidth: 14, itemHeight: 3 },
    tooltip: {
      ...tooltipBase(pal), trigger: 'axis',
      formatter: (params) => {
        const date = isoLocal(params[0].axisValue);
        const rows = params.filter((p) => p.seriesName !== 'Capital inicial').map((p) => row(p.color, p.seriesName, `${fmt.price(p.value[1], 0)} · ${fmt.pct(100 * (p.value[1] / capital - 1), 0)}`, p.seriesName === 'Comprar y mantener'));
        return `<div class="bf-tip"><div class="bf-tip__date">${fmt.date(date)}</div>${rows.join('')}</div>`;
      },
    },
    xAxis: { type: 'time', ...axisBase(pal), splitLine: { show: false } },
    yAxis: { type: 'value', scale: true, ...axisBase(pal), axisLine: { show: false },
      axisLabel: { ...axisBase(pal).axisLabel, formatter: (v) => fmt.compact(v) } },
    series: [
      ...lines.map((l) => ({
        name: l.name, type: 'line', data: l.points, showSymbol: false, step: 'end',
        lineStyle: { width: l.main ? 2.4 : 1.5, color: l.color, type: l.dashed ? [5, 4] : 'solid' }, itemStyle: { color: l.color },
        z: l.main ? 5 : 3,
        ...(l.main ? { areaStyle: { color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [{ offset: 0, color: hexToRgba(l.color, 0.2) }, { offset: 1, color: hexToRgba(l.color, 0) }]), origin: 'start' } } : {}),
      })),
      { name: 'Capital inicial', type: 'line', data: [[lines[0].points[0][0], capital], [lines[0].points[lines[0].points.length - 1][0], capital]],
        showSymbol: false, silent: true, lineStyle: { color: pal.muted, type: 'dashed', width: 1 }, tooltip: { show: false } },
    ],
  };
}
