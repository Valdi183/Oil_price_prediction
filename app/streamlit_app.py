"""
Aplicación de predicción del precio del Brent (corto, medio y largo plazo).

    poetry run streamlit run app/streamlit_app.py
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from oilcast.config import DATA_PATH, FIGURES_DIR, HORIZON_LABELS, HORIZONS, REPORTS_DIR
from oilcast.events import locate_events, recent_movements
from oilcast.explain import explain
from oilcast.pipeline import PROD_META, Dataset, forecast, load_production

st.set_page_config(page_title="Brent Forecast", page_icon="🛢️", layout="wide")

COLORS = {"corto": "#2ca02c", "medio": "#ff7f0e", "largo": "#d62728"}


# --------------------------------------------------------------------------- caché
def _mtime(p: Path) -> float:
    return p.stat().st_mtime if p.exists() else 0.0


@st.cache_data(show_spinner="Cargando datos...")
def get_dataset(_mtime: float):
    return Dataset.load()


@st.cache_resource(show_spinner="Cargando modelo...")
def get_model(_mtime: float):
    return load_production()


def clear_caches():
    get_dataset.clear()
    get_model.clear()


# --------------------------------------------------------------------------- sidebar
st.sidebar.title("🛢️ Brent Forecast")
horizon = st.sidebar.radio("Horizonte de predicción", list(HORIZONS),
                           format_func=lambda h: HORIZON_LABELS[h], index=0)
years = st.sidebar.slider("Años de histórico en el gráfico", 1, 26, 3)
show_ma = st.sidebar.checkbox("Mostrar tendencias (medias móviles 50 y 200 días)", value=True)

st.sidebar.divider()
st.sidebar.subheader("Actualización")
if st.sidebar.button("⬇️ Descargar datos recientes", width="stretch"):
    with st.spinner("Descargando de FRED y Yahoo Finance..."):
        r = subprocess.run([sys.executable, str(ROOT / "scripts" / "download_data.py")],
                           cwd=ROOT, capture_output=True, text=True)
    if r.returncode == 0:
        st.sidebar.success("Datos actualizados")
        clear_caches()
    else:
        st.sidebar.error("Error en la descarga")
        st.sidebar.code(r.stderr[-1500:])

if st.sidebar.button("🔁 Reentrenar modelo", width="stretch"):
    from scripts.retrain import retrain
    with st.spinner("Reentrenando con todos los datos disponibles (1-3 min)..."):
        m = retrain()
    st.sidebar.success(f"{m['arch']} reentrenado con datos hasta {m['data_until']}")
    clear_caches()

# --------------------------------------------------------------------------- carga
if not DATA_PATH.exists():
    st.error("No hay datos. Ejecuta `poetry run python scripts/download_data.py` o pulsa "
             "«Descargar datos recientes».")
    st.stop()
if not PROD_META.exists():
    st.error("No hay modelo entrenado. Ejecuta `poetry run python scripts/train_compare.py`.")
    st.stop()

ds = get_dataset(_mtime(DATA_PATH))
bundle = get_model(_mtime(PROD_META))
meta = bundle[2]
fc_all = forecast(ds, bundle)
brent = ds.df["brent"]
p0, last_date = fc_all["last_price"], fc_all["last_date"]
fc = fc_all["horizons"][horizon]
metrics = {m["horizonte"]: m for m in meta["test_metrics"]}

st.sidebar.divider()
st.sidebar.caption(
    f"**Modelo:** {meta['arch']} (hidden={meta['params']['hidden']}, capas={meta['params']['layers']}, "
    f"{meta['n_params']:,} parámetros)  \n**Entrenado:** {meta['trained_at']}  \n"
    f"**Datos hasta:** {meta['data_until']}  \n**Último dato cargado:** {last_date:%Y-%m-%d}")
if pd.Timestamp(meta["data_until"]) < last_date:
    st.sidebar.warning("Hay datos más recientes que el modelo: conviene reentrenar.")

# --------------------------------------------------------------------------- cabecera
st.title("Predicción del precio del petróleo Brent")
st.caption("Red neuronal recurrente entrenada con precios diarios desde 2000 y variables de mercado "
           "mundiales (WTI, gas, divisas, oro, cobre, VIX, tipo BCE). Fuente: FRED (EIA) y Yahoo Finance.")

tab_pred, tab_cycle, tab_models = st.tabs(["📈 Predicción", "🌍 Ciclo económico", "🧠 Comparativa de modelos"])

# =========================================================================== PREDICCIÓN
with tab_pred:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric(f"Último precio ({last_date:%d/%m/%Y})", f"{p0:.2f} $")
    c2.metric(f"Previsto {fc['target_date']:%d/%m/%Y}", f"{fc['price']:.2f} $", f"{fc['change_pct']:+.1f} %")
    c3.metric("Intervalo 80 %", f"{fc['low']:.1f} – {fc['high']:.1f} $")
    imp = metrics.get(horizon, {}).get("mejora_RMSE_%")
    c4.metric("Mejora vs. «precio sin cambios»", f"{imp:+.1f} %" if imp is not None else "—",
              help="Reducción del error (RMSE) frente al modelo ingenuo en el periodo de test (2023 →).")

    hist = brent.loc[last_date - pd.DateOffset(years=years):]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hist.index, y=hist, name="Brent (histórico)", line=dict(color="#1f3b57", width=1.6)))
    if show_ma:
        for w, col in [(50, "#7fb3d5"), (200, "#a569bd")]:
            ma = brent.rolling(w).mean().loc[hist.index]
            fig.add_trace(go.Scatter(x=ma.index, y=ma, name=f"Media {w} días",
                                     line=dict(color=col, width=1, dash="dot")))
    for h, f in fc_all["horizons"].items():
        sel = h == horizon
        if sel:
            fig.add_trace(go.Scatter(
                x=[last_date, f["target_date"], f["target_date"], last_date],
                y=[p0, f["high"], f["low"], p0], fill="toself", fillcolor="rgba(214,39,40,0.15)",
                line=dict(width=0), name="Intervalo 80 %", hoverinfo="skip"))
        fig.add_trace(go.Scatter(
            x=[last_date, f["target_date"]], y=[p0, f["price"]], mode="lines+markers",
            name=f"Predicción {h} ({f['h']} d)",
            line=dict(color=COLORS[h], width=3 if sel else 1.2, dash=None if sel else "dash"),
            marker=dict(size=[0, 11 if sel else 7]), opacity=1 if sel else 0.5,
            hovertemplate="%{x|%d/%m/%Y}: %{y:.2f} $<extra></extra>"))
    fig.update_layout(height=480, yaxis_title="$/barril", hovermode="x unified",
                      legend=dict(orientation="h", y=-0.15), margin=dict(t=20))
    st.plotly_chart(fig, width="stretch")

    st.subheader(f"Explicación — {HORIZON_LABELS[horizon]}")
    vix = float(ds.df["vix"].iloc[-1])
    st.markdown(explain(horizon, fc, p0, metrics.get(horizon), vix))

    st.subheader("Resumen de los tres horizontes")
    st.dataframe(pd.DataFrame([{
        "Horizonte": HORIZON_LABELS[h], "Fecha objetivo": f["target_date"].strftime("%d/%m/%Y"),
        "Precio previsto ($)": round(f["price"], 2), "Variación (%)": round(f["change_pct"], 1),
        "Intervalo 80 % ($)": f"{f['low']:.1f} – {f['high']:.1f}",
        "Mejora vs ingenuo en test (%)": round(metrics[h]["mejora_RMSE_%"], 1) if h in metrics else None,
    } for h, f in fc_all["horizons"].items()]), hide_index=True, width="stretch")

# =========================================================================== CICLO ECONÓMICO
with tab_cycle:
    ev = locate_events(brent)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=brent.index, y=brent, name="Brent", line=dict(color="#1f3b57", width=1.2)))
    for i, e in ev.reset_index(drop=True).iterrows():
        up = e["kind"] == "max"
        fig.add_trace(go.Scatter(
            x=[e["date"]], y=[e["price"]], mode="markers+text", text=[str(i + 1)],
            textposition="top center" if up else "bottom center",
            marker=dict(size=11, color="#d62728" if up else "#2ca02c", symbol="triangle-up" if up else "triangle-down"),
            name=f"{i + 1}. {e['title']}",
            hovertemplate=f"<b>{e['title']}</b><br>%{{x|%d/%m/%Y}}: %{{y:.2f}} $<extra></extra>"))
    for x0, x1, lab in [("2008-01-01", "2009-06-30", "Gran Recesión"), ("2020-02-01", "2020-04-30", "COVID-19")]:
        fig.add_vrect(x0=x0, x1=x1, fillcolor="grey", opacity=0.12, line_width=0,
                      annotation_text=lab, annotation_position="top left")
    fig.update_layout(height=520, yaxis_title="$/barril", legend=dict(font=dict(size=10)), margin=dict(t=20))
    st.plotly_chart(fig, width="stretch")
    st.caption("▲ rojo: picos · ▼ verde: caídas · zonas grises: recesiones mundiales")

    st.subheader("Movimientos más recientes")
    st.markdown(recent_movements(brent))

    st.subheader("Qué explica cada pico y caída")
    for i, e in ev.reset_index(drop=True).iterrows():
        with st.expander(f"{i + 1}. {e['title']} — {e['date']:%m/%Y}: {e['price']:.2f} $  ·  {e['cycle']}"):
            st.write(e["text"])

# =========================================================================== COMPARATIVA
with tab_models:
    comp_path = REPORTS_DIR / "comparison.csv"
    if not comp_path.exists():
        st.info("Ejecuta `scripts/train_compare.py` para generar la comparativa.")
    else:
        comp = pd.read_csv(comp_path)
        st.markdown(
            "Las tres redes se entrenaron con **los mismos datos, la misma ventana de 60 días, la misma "
            "partición** (validación walk-forward 2017-2022, test 2023 → hoy) y **la misma rejilla de "
            "hiperparámetros**. La «mejora» compara el error con el modelo ingenuo "
            "(«el precio futuro = el precio de hoy»); valores negativos significan que la red lo hace peor.")
        show = comp[["red", "horizonte", "MAE", "RMSE", "mejora_RMSE_%", "std_mejora_RMSE_%", "mejora_MAE_%",
                     "acierto_direccion_%", "cobertura_IC80_%", "parametros",
                     "tiempo_entrenamiento_s", "hiperparametros"]]
        st.dataframe(show.style.format(precision=2).background_gradient(
            subset=["mejora_RMSE_%"], cmap="RdYlGn", vmin=-30, vmax=30),
            hide_index=True, width="stretch")
        st.success(f"Modelo integrado en la aplicación: **{meta['arch']}**")
        for img in ["improvement_vs_naive.png", "test_predictions.png"]:
            if (FIGURES_DIR / img).exists():
                st.image(str(FIGURES_DIR / img))
