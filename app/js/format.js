// Formato numérico y de fechas en español, y un render mínimo de markdown.

const NBSP = ' ';
const cache = new Map();

function nf(nd) {
  const key = `n${nd}`;
  if (!cache.has(key)) {
    cache.set(key, new Intl.NumberFormat('es-ES', { minimumFractionDigits: nd, maximumFractionDigits: nd }));
  }
  return cache.get(key);
}

export const fmt = {
  num(x, nd = 2) {
    if (x === null || x === undefined || Number.isNaN(x)) return '—';
    return nf(nd).format(x);
  },
  int(x) {
    if (x === null || x === undefined) return '—';
    return nf(0).format(Math.round(x));
  },
  price(x, nd = 2) {
    if (x === null || x === undefined) return '—';
    return `${nf(nd).format(x)}${NBSP}$`;
  },
  pct(x, nd = 1, signed = true) {
    if (x === null || x === undefined || Number.isNaN(x)) return '—';
    const sign = signed && x > 0 ? '+' : '';
    return `${sign}${nf(nd).format(x)}${NBSP}%`;
  },
  signed(x, nd = 2, unit = '$') {
    if (x === null || x === undefined) return '—';
    const sign = x > 0 ? '+' : '';
    return `${sign}${nf(nd).format(x)}${NBSP}${unit}`;
  },
  compact(x) {
    if (x >= 1000) return `${nf(1).format(x / 1000)}${NBSP}k`;
    return nf(0).format(x);
  },
  parseDate(iso) {
    const [y, m, d] = iso.split('-').map(Number);
    return new Date(Date.UTC(y, m - 1, d));
  },
  date(iso) {
    if (!iso) return '—';
    const d = fmt.parseDate(iso);
    return d.toLocaleDateString('es-ES', { day: 'numeric', month: 'short', year: 'numeric', timeZone: 'UTC' }).replace('.', '');
  },
  dateLong(iso) {
    if (!iso) return '—';
    return fmt.parseDate(iso).toLocaleDateString('es-ES', { day: 'numeric', month: 'long', year: 'numeric', timeZone: 'UTC' });
  },
  monthYear(iso) {
    const s = fmt.parseDate(iso).toLocaleDateString('es-ES', { month: 'short', year: 'numeric', timeZone: 'UTC' }).replace('.', '');
    return s.charAt(0).toUpperCase() + s.slice(1);
  },
  dateTime(s) {
    // "2026-11-15 17:43" -> "15 nov 2026, 17:43"
    if (!s) return '—';
    const [d, t] = s.split(' ');
    return t ? `${fmt.date(d)}, ${t}` : fmt.date(d);
  },
};

export function stars(p) {
  if (p === null || p === undefined) return '';
  if (p < 0.01) return '★★★';
  if (p < 0.05) return '★★';
  if (p < 0.10) return '★';
  return '';
}

export function pValue(p) {
  if (p === null || p === undefined) return '—';
  return p < 0.001 ? 'p < 0,001' : `p = ${fmt.num(p, 2)}`;
}

export function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
}

// Markdown mínimo: párrafos separados por línea en blanco, **negrita** y _cursiva_.
export function mdToHtml(md) {
  if (!md) return '';
  return md.split(/\n\s*\n/).map((para) => {
    let html = escapeHtml(para.trim());
    html = html.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
    html = html.replace(/(^|\s)_(.+?)_(?=\s|$|[.,;:])/g, '$1<em>$2</em>');
    return `<p>${html}</p>`;
  }).join('');
}

export function addYears(date, years) {
  const d = new Date(date.getTime());
  d.setUTCFullYear(d.getUTCFullYear() + years);
  return d;
}

export function addMonths(date, months) {
  const d = new Date(date.getTime());
  d.setUTCMonth(d.getUTCMonth() + months);
  return d;
}

// Milisegundos de una fecha ISO en hora LOCAL (ECharts interpreta "YYYY-MM-DD" como local).
export function localMs(iso) {
  const [y, m, d] = iso.split('-').map(Number);
  return new Date(y, m - 1, d).getTime();
}

export function isoLocal(ms) {
  const d = new Date(ms);
  const pad = (n) => String(n).padStart(2, '0');
  return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())}`;
}

export function shiftMonths(ms, months) {
  const d = new Date(ms);
  d.setMonth(d.getMonth() + months);
  return d.getTime();
}

export const prefersReducedMotion = () =>
  window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;
