// Tema claro/oscuro: variables CSS en :root[data-theme]. Se recuerda en localStorage.

const KEY = 'bf-theme';

export function currentTheme() {
  return document.documentElement.getAttribute('data-theme') === 'light' ? 'light' : 'dark';
}

export function applyTheme(theme) {
  document.documentElement.setAttribute('data-theme', theme);
  const meta = document.querySelector('meta[name="theme-color"]');
  if (meta) meta.setAttribute('content', theme === 'light' ? '#f5f6f8' : '#0b0f17');
  try { localStorage.setItem(KEY, theme); } catch (e) { /* almacenamiento no disponible */ }
}

export function initTheme(button, onChange) {
  button.addEventListener('click', () => {
    const next = currentTheme() === 'light' ? 'dark' : 'light';
    applyTheme(next);
    onChange(next);
  });
}

// Lee los colores de las variables CSS para que los gráficos sigan al tema.
export function palette() {
  const cs = getComputedStyle(document.documentElement);
  const v = (name) => cs.getPropertyValue(name).trim();
  return {
    theme: currentTheme(),
    bg: v('--bg'), surface: v('--surface'), border: v('--border'), borderStrong: v('--border-strong'),
    text: v('--text'), text2: v('--text-2'), muted: v('--muted'),
    accent: v('--accent'), accent2: v('--accent-2'), accentSoft: v('--accent-soft'), accentGlow: v('--accent-glow'),
    up: v('--up'), down: v('--down'), info: v('--info'),
    lstm: v('--lstm'), gru: v('--gru'), combo: v('--combo'), wti: v('--wti'), ma50: v('--ma50'), ma200: v('--ma200'),
    grid: v('--grid-line'), tooltipBg: v('--tooltip-bg'),
    fontUi: v('--font-ui'), fontMono: v('--font-mono'),
  };
}
