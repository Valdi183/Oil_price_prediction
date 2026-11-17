// Renderizado del DOM a partir de los JSON exportados por scripts/export_site.py.
import { fmt, stars, pValue, mdToHtml, escapeHtml, prefersReducedMotion } from './format.js';

const $ = (sel) => document.querySelector(sel);
const BADGE_CLASS = { 'media-alta': 'badge--alta', media: 'badge--media', baja: 'badge--baja', desconocida: 'badge--neutral' };
const ARCH_DESC = {
  RNN: 'Red recurrente simple: una capa de memoria sin puertas. La más ligera y, aquí, la más precisa.',
  LSTM: 'Memoria a largo plazo con tres puertas (olvido, entrada y salida). La más estable entre semillas.',
  GRU: 'Variante de la LSTM con dos puertas. Buena a medio plazo, inestable a largo.',
};
const ARROW_UP = '<svg class="icon" viewBox="0 0 24 24" aria-hidden="true"><path d="M12 19V5M5 12l7-7 7 7"/></svg>';
const ARROW_DOWN = '<svg class="icon" viewBox="0 0 24 24" aria-hidden="true"><path d="M12 5v14M19 12l-7 7-7-7"/></svg>';

function dirClass(x) { return x > 0 ? 'is-up' : x < 0 ? 'is-down' : ''; }

function animateNumber(el, to, nd, duration = 1300) {
  if (prefersReducedMotion()) { el.textContent = fmt.num(to, nd); return; }
  const start = performance.now();
  const from = 0;
  const step = (now) => {
    const t = Math.min(1, (now - start) / duration);
    const eased = 1 - Math.pow(1 - t, 3);
    el.textContent = fmt.num(from + (to - from) * eased, nd);
    if (t < 1) requestAnimationFrame(step);
  };
  requestAnimationFrame(step);
}

// ------------------------------------------------------------------ cabecera
export function renderHero(forecast, meta) {
  $('#hero-date').textContent = `datos hasta el ${fmt.date(forecast.last_date)}`;
  animateNumber($('#hero-price'), forecast.last_price, 2);
  const c = forecast.changes;
  const chip = (id, value, abs) => {
    const el = $(id);
    el.classList.remove('is-up', 'is-down');
    el.classList.add(dirClass(value));
    el.querySelector('.chip__value').textContent = abs !== undefined ? `${fmt.signed(abs)} · ${fmt.pct(value)}` : fmt.pct(value);
  };
  chip('#chip-day', c.day_pct, c.day_abs);
  chip('#chip-month', c.month_pct);
  chip('#chip-year', c.year_pct);
  $('#spark-range').textContent = `mín ${fmt.price(c.year_low)} · máx ${fmt.price(c.year_high)}`;
  const link = meta.action_url;
  for (const id of ['#update-link', '#update-link-2', '#action-link']) $(id).href = link;
  $('#repo-link').href = meta.repo_url;
}

// ------------------------------------------------------------------ tarjetas de horizonte
export function renderHorizonCards(forecast, selected) {
  const wrap = $('#horizon-cards');
  const p0 = forecast.last_price;
  wrap.innerHTML = '';
  for (const key of ['corto', 'medio', 'largo']) {
    const h = forecast.horizons[key];
    const weak = h.reliability.level === 'baja';
    const lo = Math.min(h.low, p0), hi = Math.max(h.high, p0);
    const pos = (v) => `${(100 * (v - lo) / (hi - lo)).toFixed(1)}%`;
    const art = document.createElement('article');
    art.className = `hcard${weak ? ' hcard--weak' : ''}${key === selected ? ' is-selected' : ''}`;
    art.dataset.horizon = key;
    art.setAttribute('role', 'button');
    art.setAttribute('tabindex', '0');
    art.setAttribute('aria-pressed', key === selected ? 'true' : 'false');
    art.innerHTML = `
      <div class="hcard__top">
        <div><div class="hcard__title">${h.label}</div><div class="hcard__sub">${h.sublabel}</div></div>
        <span class="badge ${BADGE_CLASS[h.reliability.level]}">Fiabilidad ${h.reliability.label}</span>
      </div>
      <div class="hcard__price mono">${fmt.num(h.price)}<small>$</small></div>
      <div class="hcard__change mono ${dirClass(h.change_pct)}">${h.change_pct > 0 ? ARROW_UP : h.change_pct < 0 ? ARROW_DOWN : ''} ${fmt.pct(h.change_pct)} · ${fmt.signed(h.change_abs)} frente al último precio</div>
      <div class="hcard__range">
        <div class="hcard__range-label"><span>Rango del ${forecast.interval_level} %</span><span>±${fmt.num(h.width_pct / 2, 0)} %</span></div>
        <div class="rangebar">
          <div class="rangebar__fill" style="left:${pos(h.low)};right:${(100 - parseFloat(pos(h.high))).toFixed(1)}%"></div>
          <div class="rangebar__dot rangebar__dot--now" style="left:${pos(p0)}" title="Último precio"></div>
          <div class="rangebar__dot" style="left:${pos(h.price)}" title="Predicción"></div>
        </div>
        <div class="hcard__range-values mono"><span>${fmt.price(h.low)}</span><span>${fmt.price(h.high)}</span></div>
      </div>
      <div class="hcard__foot">
        <span>Objetivo: ${fmt.date(h.target_date)}</span>
        ${weak ? `<span class="hcard__warn">${ARROW_DOWN} No fiable: mira el rango</span>` : `<span class="mono">MAE test ${fmt.price(h.metrics.MAE)}</span>`}
      </div>`;
    wrap.appendChild(art);
  }
}

export function markSelectedCard(selected) {
  document.querySelectorAll('.hcard').forEach((el) => {
    const on = el.dataset.horizon === selected;
    el.classList.toggle('is-selected', on);
    el.setAttribute('aria-pressed', on ? 'true' : 'false');
  });
  document.querySelectorAll('#horizon-segmented button').forEach((b) => b.setAttribute('aria-selected', b.dataset.horizon === selected ? 'true' : 'false'));
  document.querySelectorAll('#test-segmented button, #val-segmented button').forEach((b) => b.setAttribute('aria-pressed', b.dataset.horizon === selected ? 'true' : 'false'));
}

// ------------------------------------------------------------------ explicación
export function renderExplanation(forecast, horizon) {
  const h = forecast.horizons[horizon];
  const m = h.metrics || {};
  $('#explain-title').textContent = `${h.label} · ${h.sublabel}`;
  const badge = $('#explain-badge');
  badge.className = `badge ${BADGE_CLASS[h.reliability.level]}`;
  badge.textContent = `Fiabilidad ${h.reliability.label}`;
  $('#explain-warning').hidden = h.reliability.level !== 'baja';
  $('#explain-body').innerHTML = mdToHtml(h.explanation);
  $('#chart-subtitle').textContent = `Brent spot diario · ${h.label.toLowerCase()}: predicción a ${h.h} días hábiles (objetivo ${fmt.date(h.target_date)}) con su rango del ${forecast.interval_level} %`;

  const rows = [
    ['Error medio del modelo (MAE)', fmt.price(m.MAE)],
    ['Error medio del ingenuo (MAE)', fmt.price(m.MAE_ingenuo)],
    ['RMSE modelo / ingenuo', `${fmt.num(m.RMSE)} / ${fmt.num(m.RMSE_ingenuo)}`],
    ['MSE del modelo', fmt.num(m.MSE)],
    ['R² fuera de muestra vs ingenuo', `<span class="${dirClass(m['R2_OOS_%'])}">${fmt.pct(m['R2_OOS_%'])}</span>`],
    ['Mejora del RMSE', `<span class="${dirClass(m['mejora_RMSE_%'])}">${fmt.pct(m['mejora_RMSE_%'])}</span>`],
    ['Acierto de la dirección', fmt.pct(m['acierto_direccion_%'], 0, false)],
    ['Predicciones evaluadas', fmt.int(m.n)],
  ];
  $('#explain-metrics').innerHTML = rows.map(([k, v]) => `<dt>${k}</dt><dd>${v}</dd>`).join('');
  const s = h.significance;
  const st = s ? stars(s.p_valor) : '';
  $('#explain-seal').innerHTML = s
    ? `<span class="seal__stars ${st ? '' : 'is-none'}">${st || '—'}</span><span><strong>Diebold-Mariano</strong> frente al ingenuo: ${pValue(s.p_valor)}. ${s.p_valor < 0.05 ? 'La mejora es estadísticamente significativa.' : s.p_valor < 0.10 ? 'Significativa solo al 10 %: indicio, no certeza.' : 'La mejora no es estadísticamente significativa.'}</span>`
    : '';
}

// ------------------------------------------------------------------ ciclo económico
export function renderTimeline(events, onSelect) {
  const tl = $('#timeline');
  tl.innerHTML = '';
  const sorted = [...events].sort((a, b) => a.date.localeCompare(b.date));
  for (const ev of sorted) {
    const card = document.createElement('article');
    card.className = `tl-card tl-card--${ev.kind}`;
    card.dataset.id = ev.id;
    card.setAttribute('tabindex', '0');
    card.setAttribute('role', 'button');
    card.innerHTML = `
      <div class="tl-card__top">
        <span class="tl-card__icon">${ev.kind === 'max' ? ARROW_UP : ARROW_DOWN}</span>
        <span class="tl-card__date">${fmt.date(ev.date)}</span>
      </div>
      <div class="tl-card__title">${escapeHtml(ev.title)}</div>
      <div class="tl-card__price">${fmt.num(ev.price)} <small>$/barril</small></div>
      <span class="tl-card__cycle">${escapeHtml(ev.cycle)}</span>
      <p class="tl-card__text">${escapeHtml(ev.text)}</p>
      <span class="tl-card__more">Ver en el gráfico →</span>`;
    const pick = () => onSelect(ev.id);
    card.addEventListener('click', pick);
    card.addEventListener('keydown', (e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); pick(); } });
    tl.appendChild(card);
  }
  $('#tl-prev').addEventListener('click', () => tl.scrollBy({ left: -320, behavior: 'smooth' }));
  $('#tl-next').addEventListener('click', () => tl.scrollBy({ left: 320, behavior: 'smooth' }));
}

export function markSelectedEvent(id, scrollIntoView = false) {
  document.querySelectorAll('.tl-card').forEach((el) => {
    const on = el.dataset.id === id;
    el.classList.toggle('is-selected', on);
    if (on && scrollIntoView) el.scrollIntoView({ behavior: prefersReducedMotion() ? 'auto' : 'smooth', block: 'nearest', inline: 'center' });
  });
}

export function renderRecent(forecast) {
  $('#recent-movements').innerHTML = mdToHtml(forecast.recent_movements);
}

// ------------------------------------------------------------------ comparativa
export function renderModelCards(models) {
  const wrap = $('#model-cards');
  wrap.innerHTML = '';
  for (const m of models.models) {
    const card = document.createElement('article');
    card.className = `mcard${m.selected ? ' mcard--selected' : ''}`;
    const imp = (hz) => {
      const v = m.mejora[hz], p = m.p_vs_ingenuo[hz];
      return `<div><div class="mcard__k">Mejora ${hz}</div><div class="mcard__v ${dirClass(v)}">${fmt.pct(v)} <span class="stars ${stars(p) ? '' : 'is-none'}">${stars(p) || '·'}</span></div></div>`;
    };
    card.innerHTML = `
      ${m.selected ? '<span class="mcard__tag">Modelo en producción</span>' : ''}
      <div class="mcard__head"><span class="mcard__name">${m.red}<small>${m.hidden} neuronas × ${m.layers} ${m.layers > 1 ? 'capas' : 'capa'}</small></span></div>
      <p class="mcard__desc">${ARCH_DESC[m.red] || ''}</p>
      <div class="mcard__grid">
        <div><div class="mcard__k">Parámetros</div><div class="mcard__v">${fmt.int(m.parametros)}</div></div>
        <div><div class="mcard__k">Épocas · tiempo</div><div class="mcard__v">${m.epocas} · ${fmt.num(m.tiempo_entrenamiento_s, 0)} s</div></div>
        <div><div class="mcard__k">Error relativo en CV</div><div class="mcard__v">${fmt.num(m.score_cv, 3)} <span class="muted small">±${fmt.num(m.std_cv_semillas, 3)}</span></div></div>
        <div><div class="mcard__k">Acierto dirección (medio)</div><div class="mcard__v">${fmt.pct(models.comparison.find((r) => r.red === m.red && r.horizonte === 'medio')['acierto_direccion_%'], 0, false)}</div></div>
        ${imp('corto')}${imp('medio')}${imp('largo')}
      </div>
      <div class="mcard__folds">${m.cv_folds.map((f) => `<div class="fold">${f.label}<b>${fmt.num(f.score, 3)}</b></div>`).join('')}</div>`;
    wrap.appendChild(card);
  }
}

export function renderWhy(models, forecast) {
  const by = Object.fromEntries(models.models.map((m) => [m.red, m]));
  const sel = by[models.selected];
  const others = models.models.filter((m) => !m.selected);
  const cvList = others.map((m) => `${m.red} ${fmt.num(m.score_cv, 3)}`).join(' y ');
  const onlyOnePct = models.models.filter((m) => m.p_vs_ingenuo.corto < 0.01).length === 1;
  const medioList = others.map((m) => `${m.red} ${fmt.pct(m.mejora.medio)}`).join(', ');
  const largoP = models.models.map((m) => `${m.red} ${pValue(m.p_vs_ingenuo.largo)}`).join(', ');
  const items = [
    `<strong>Menor error de validación cruzada</strong> en los tres regímenes de mercado (${sel.cv_folds.map((f) => f.label).join(', ')}): ${fmt.num(sel.score_cv, 3)} frente a ${cvList}. El test de 2023 en adelante lo confirma: es la mejor a corto y medio plazo.`,
    `<strong>${onlyOnePct ? 'La única red' : 'Una de las redes'} que bate al ingenuo con ${pValue(sel.p_vs_ingenuo.corto)} a corto plazo</strong> (test de Diebold-Mariano). La mejora es pequeña (${fmt.pct(sel.mejora.corto)}) pero demostrada: sirve como orientación de tendencia.`,
    `<strong>La mejor a medio plazo</strong>: ${fmt.pct(sel.mejora.medio)} de mejora del RMSE frente al ingenuo (${medioList}) y acierta la dirección el ${fmt.pct(forecast.horizons.medio.metrics['acierto_direccion_%'], 0, false)} de las veces. Es donde el modelo aporta valor.`,
    `<strong>A largo plazo ninguna red es fiable</strong>: ${largoP}. Con solo 3–4 años no solapados el test no puede distinguirlas, y la dispersión entre semillas es enorme (±${fmt.num(sel.std_mejora.largo, 0)} % en la RNN). Por eso a 12 meses la aplicación destaca el rango y atenúa la cifra.`,
    `<strong>Coste contenido</strong>: ${fmt.int(sel.parametros)} parámetros por red y unos ${fmt.num(sel.tiempo_entrenamiento_s, 0)} s de reentrenamiento del conjunto en CPU. En el día a día no hace falta: los datos nuevos pasan por el modelo ya entrenado y solo se reentrena cuando se decide.`,
  ];
  $('#why-list').innerHTML = items.map((t) => `<li>${t}</li>`).join('');
}

export function renderSeals(models) {
  const wrap = $('#seals-grid');
  const nets = ['RNN', 'LSTM', 'GRU', 'Combinación'];
  wrap.innerHTML = nets.map((net) => {
    const rows = ['corto', 'medio', 'largo'].map((hz) => {
      const s = models.significance.find((r) => r.modelo === net && r.frente_a === 'Ingenuo' && r.horizonte === hz);
      if (!s) return '';
      const st = stars(s.p_valor);
      return `<div class="sealrow"><span class="sealrow__h">${hz.charAt(0).toUpperCase() + hz.slice(1)}</span><span class="sealrow__v ${dirClass(s['mejora_RMSE_%'])}">${fmt.pct(s['mejora_RMSE_%'])}</span><span class="muted small">${pValue(s.p_valor)}</span><span class="stars ${st ? '' : 'is-none'}">${st || '—'}</span></div>`;
    }).join('');
    const ref = net === 'Combinación';
    return `<div class="sealcard${net === models.selected ? ' sealcard--selected' : ''}${ref ? ' sealcard--ref' : ''}"><div class="sealcard__name">${net}${ref ? '<span class="sealcard__ref">solo referencia</span>' : net === models.selected ? '<span class="sealcard__ref">en producción</span>' : ''}</div>${rows}</div>`;
  }).join('');
}

export function renderTables(models) {
  const head = ['Red', 'Horizonte', 'MAE', 'MSE', 'RMSE', 'R² precio', 'R² vs ingenuo', 'MAE ingenuo', 'RMSE ingenuo', 'Mejora RMSE', '± semillas', 'Acierto dir.', 'Cobertura 80 %', 'Parámetros', 'Épocas', 'Tiempo', 'Error CV'];
  const body = models.comparison.map((r) => {
    const cls = r.reference ? 'is-ref' : r.selected ? 'is-selected' : '';
    const pill = r.reference ? '<span class="pill">solo referencia</span>' : r.selected && r.horizonte === 'corto' ? '<span class="pill pill--accent">producción</span>' : '';
    return `<tr class="${cls}">
      <td>${r.red}${pill}</td><td>${r.horizonte} (${r.dias} d)</td>
      <td>${fmt.num(r.MAE)}</td><td>${fmt.num(r.MSE)}</td><td>${fmt.num(r.RMSE)}</td>
      <td>${fmt.num(r.R2_precio, 3)}</td><td class="${dirClass(r['R2_OOS_%'])}">${fmt.pct(r['R2_OOS_%'])}</td><td>${fmt.num(r.MAE_ingenuo)}</td><td>${fmt.num(r.RMSE_ingenuo)}</td>
      <td class="${dirClass(r['mejora_RMSE_%'])}">${fmt.pct(r['mejora_RMSE_%'])}</td>
      <td>${r['std_mejora_RMSE_%'] != null ? `±${fmt.num(r['std_mejora_RMSE_%'], 1)}` : '—'}</td>
      <td>${fmt.pct(r['acierto_direccion_%'], 1, false)}</td>
      <td>${r['cobertura_IC80_%'] != null ? fmt.pct(r['cobertura_IC80_%'], 1, false) : '—'}</td>
      <td>${fmt.int(r.parametros)}</td><td>${r.epocas ?? '—'}</td><td>${fmt.num(r.tiempo_entrenamiento_s, 1)} s</td>
      <td>${r.score_cv != null ? fmt.num(r.score_cv, 3) : '—'}</td></tr>`;
  }).join('');
  $('#comparison-table').innerHTML = `<thead><tr>${head.map((h) => `<th>${h}</th>`).join('')}</tr></thead><tbody>${body}</tbody>`;

  const head2 = ['Configuración', 'Red', 'Parámetros', 'Épocas', 'Error CV', 'Mejora corto', 'Mejora medio', 'Mejora largo', 'Tiempo'];
  const cell = (v, s) => `<td class="${dirClass(v)}">${fmt.pct(v)} <span class="muted">(±${fmt.num(s, 1)})</span></td>`;
  const body2 = models.equal_size.map((r) => `<tr class="${r.red === models.selected && r.parametros === models.models.find((m) => m.selected).parametros ? 'is-selected' : ''}">
      <td class="mono">${r.configuracion}</td><td>${r.red}</td><td>${fmt.int(r.parametros)}</td><td>${r.epocas}</td><td>${fmt.num(r.score_cv, 3)}</td>
      ${cell(r['mejora_corto_%'], r.std_corto)}${cell(r['mejora_medio_%'], r.std_medio)}${cell(r['mejora_largo_%'], r.std_largo)}
      <td>${fmt.num(r.tiempo_entrenamiento_s, 1)} s</td></tr>`).join('');
  $('#equal-table').innerHTML = `<thead><tr>${head2.map((h) => `<th>${h}</th>`).join('')}</tr></thead><tbody>${body2}</tbody>`;
  $('#test-title').textContent = `Periodo de test: ${fmt.date(models.test_period.start)} → ${fmt.date(models.test_period.end)}`;
}

// ------------------------------------------------------------------ validación fuera de muestra
const HZ = { corto: 'Corto plazo', medio: 'Medio plazo', largo: 'Largo plazo' };
const HZ_DAYS = { corto: 5, medio: 63, largo: 252 };

export function renderValidationStatic(models) {
  const v = models.validation;
  const get = (model, hz) => v.metrics.find((r) => r.modelo === model && r.horizonte === hz);
  const kv = (k, val, cls = '', sub = '') => `<div><div class="vbox__k">${k}</div><div class="vbox__v ${cls}">${val}${sub ? `<small>${sub}</small>` : ''}</div></div>`;
  $('#val-strip').innerHTML = ['corto', 'medio', 'largo'].map((hz) => {
    const r = get(models.selected, hz), n = get('Ingenuo', hz);
    return `<div class="vbox"><div class="vbox__h">${HZ[hz]}<small>${HZ_DAYS[hz]} días · ${fmt.int(r.n)} predicciones</small></div><div class="vbox__grid">
      ${kv('MAE', fmt.price(r.MAE), '', `ingenuo ${fmt.num(n.MAE)}`)}
      ${kv('RMSE', fmt.price(r.RMSE), '', `ingenuo ${fmt.num(n.RMSE)}`)}
      ${kv('MSE', fmt.num(r.MSE), '', `ingenuo ${fmt.num(n.MSE)}`)}
      ${kv('R² precio', fmt.num(r.R2_precio, 3), '', `ingenuo ${fmt.num(n.R2_precio, 3)}`)}
      ${kv('R² vs ingenuo', fmt.pct(r['R2_OOS_%']), dirClass(r['R2_OOS_%']))}
      ${kv('Acierto dirección', fmt.pct(r['acierto_direccion_%'], 0, false))}
    </div></div>`;
  }).join('');

  const hzs = ['corto', 'medio', 'largo'];
  const cell = (tipo, g, hz) => {
    const r = v.periods.find((x) => x.tipo === tipo && x.grupo === g && x.horizonte === hz);
    if (!r) return '<td class="muted">—</td>';
    const val = r['mejora_RMSE_%'];
    const heat = Math.min(40, 6 + Math.abs(val) * 1.2).toFixed(0);
    return `<td class="${dirClass(val)} heat--${val >= 0 ? 'up' : 'down'}" style="--heat:${heat}%" title="RMSE RNN ${fmt.price(r.RMSE)} · ingenuo ${fmt.price(r.RMSE_ingenuo)} · ${fmt.int(r.n)} predicciones">${fmt.pct(val)}</td>`;
  };
  const rows = [
    '<tr class="group-row"><td colspan="4">Por año de origen</td></tr>',
    ...v.years.map((y) => `<tr><td>${y}</td>${hzs.map((hz) => cell('año', y, hz)).join('')}</tr>`),
    '<tr class="group-row"><td colspan="4">Por volatilidad del mercado</td></tr>',
    ...v.regimes.map((g) => `<tr><td>${g}</td>${hzs.map((hz) => cell('régimen', g, hz)).join('')}</tr>`),
  ];
  $('#period-table').innerHTML = `<thead><tr><th>Grupo</th><th>Corto</th><th>Medio</th><th>Largo</th></tr></thead><tbody>${rows.join('')}</tbody>`;
  const [q1, q2] = v.regime_thresholds;
  $('#period-note').textContent = `Positivo = la RNN se equivoca menos que el ingenuo. Régimen según la volatilidad del Brent a 20 días el día de la predicción (anualizada): calma ≤ ${fmt.num(q1, 0)} %, normal ${fmt.num(q1, 0)}–${fmt.num(q2, 0)} %, turbulencia > ${fmt.num(q2, 0)} %. A largo plazo, las predicciones hechas en 2026 aún no tienen valor real.`;
}

const sgn = (x) => `${x > 0 ? '+' : ''}${fmt.num(x)}`;

export function renderValidationHorizon(models, horizon) {
  const v = models.validation;
  const roll = v.rolling[horizon];
  $('#val-roll-title').textContent = `${HZ[horizon]}: ¿cuándo bate al ingenuo?`;
  $('#val-roll-note').textContent = `Cociente entre el RMSE de la RNN y el del ingenuo en una ventana móvil de ${roll.window} días. Por debajo de 1 (verde) la RNN se equivoca menos; lo consigue el ${fmt.num(roll.beats_pct, 0)} % del periodo de test.`;
  $('#worst-title').textContent = `Los 5 mayores errores · ${HZ[horizon].toLowerCase()}`;
  const rows = v.worst.filter((r) => r.horizonte === horizon).map((r) => `<tr>
      <td>${fmt.date(r.fecha_origen)}</td><td>${fmt.num(r.real)}</td><td>${fmt.num(r.prediccion)}</td>
      <td class="is-down">${sgn(r.error)}</td><td>${sgn(r.error_ingenuo)}</td></tr>`).join('');
  $('#worst-table').innerHTML = `<thead><tr><th>Origen</th><th>Real</th><th>Pred.</th><th>Error RNN</th><th>Err. ingenuo</th></tr></thead><tbody>${rows}</tbody>`;
}

// ------------------------------------------------------------------ cómo funciona y pie
export function renderMeta(meta) {
  const set = (k, v) => { const el = document.querySelector(`[data-stat="${k}"]`); if (el) el.textContent = v; };
  set('years', fmt.int(meta.years_of_data));
  set('features', fmt.int(meta.n_features));
  set('params', fmt.compact(meta.n_params));
  set('seconds', `~${fmt.num(meta.train_seconds, 0)} s`);
  $('#pipe-features').textContent = fmt.int(meta.n_features);
  $('#pipe-arch').textContent = `${meta.arch_label} × ${meta.n_members} semillas`;
  $('#update-meta').innerHTML = [
    ['Datos hasta', fmt.date(meta.data_until)],
    ['Modelo entrenado con datos hasta', fmt.date(meta.model_data_until)],
    ['Último reentrenamiento', fmt.dateTime(meta.trained_at)],
    ['Web generada', meta.generated_at],
    ['Modelo', `${meta.arch_label} · ${meta.n_members} semillas · ${meta.n_epochs} épocas · error ${meta.loss}`],
  ].map(([k, v]) => `<dt>${k}</dt><dd>${v}</dd>`).join('');
  $('#footer-sources').innerHTML = meta.sources.map((s) => `<li><a href="${s.url}" target="_blank" rel="noopener">${s.name}</a> <span class="muted">· ${s.detail}</span></li>`).join('');
  $('#footer-generated').textContent = `Datos hasta ${fmt.date(meta.data_until)} · generado ${meta.generated_at}`;
}

export function renderLegend(layers, horizonLabel) {
  const items = [
    ['Brent spot', 'legend__swatch'],
    layers.ma50 ? ['Media 50 d', 'legend__swatch legend__swatch--dash', 'var(--ma50)'] : null,
    layers.ma200 ? ['Media 200 d', 'legend__swatch legend__swatch--dash', 'var(--ma200)'] : null,
    layers.wti ? ['WTI', 'legend__swatch', 'var(--wti)'] : null,
    [`Predicción ${horizonLabel.toLowerCase()}`, 'legend__swatch legend__swatch--dash', 'var(--accent-2)'],
    ['Rango 80 %', 'legend__swatch legend__swatch--area'],
    layers.events ? ['▲ pico', 'legend__swatch legend__swatch--dot', 'var(--down)'] : null,
    layers.events ? ['▼ caída', 'legend__swatch legend__swatch--dot', 'var(--up)'] : null,
  ].filter(Boolean);
  $('#main-legend').innerHTML = items.map(([name, cls, color]) => `<span class="legend__item"><i class="${cls}" ${color ? `style="background:${color}"` : ''}></i>${name}</span>`).join('');
}

export function showToast(msg) {
  const t = $('#toast');
  t.textContent = msg;
  t.hidden = false;
}
