"""
Exporta a app/data/*.json todo lo que muestra la web estática (GitHub Pages).

Todo el cálculo (descarga de datos, predicción del conjunto de redes, intervalos del 80 %,
explicaciones en lenguaje sencillo, localización de los eventos del ciclo económico y la
comparativa de modelos) se hace aquí, en Python. La web solo lee estos ficheros y los dibuja.

Uso:
    poetry run python scripts/07_export_app_data.py          # requiere data/ y models/ (06_train_production_model.py)

Ficheros generados (cifras redondeadas a 2 decimales):
    forecast.json  último precio, variaciones, predicción por horizonte, explicación y fiabilidad
    history.json   serie diaria del Brent desde 2000, medias móviles de 50/200 días y WTI
    events.json    picos y caídas del ciclo económico situados sobre la serie real
    models.json    comparativa RNN/LSTM/GRU, tabla a igual tamaño, Diebold-Mariano y predicciones de test
    tools.json     simulador a 1-252 días, distribución del error, escenarios históricos y backtest
    meta.json      fechas de datos y reentrenamiento, arquitectura y enlaces (repo, Action, web)
"""
from __future__ import annotations

import json
import math
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import (APP_DATA_DIR, CV_FOLDS, HORIZONS, INTERVAL_HIGH_Q, INTERVAL_LOW_Q,  # noqa: E402
                        TABLES_DIR, TEST_START, WINDOW)
from src.data_cleaning import clean, load_raw  # noqa: E402
from src.decision_tools import (backtest_trades, crisis_scenarios, historical_moves,  # noqa: E402
                                load_forecast_errors, simulator_grid)
from src.evaluation import (REGIMES, by_group, full_metrics, load_test_predictions, regimes,  # noqa: E402
                                rolling_ratio, worst_errors)
from src.economic_events import locate_events, recent_movements  # noqa: E402
from src.explanations import explain, reliability  # noqa: E402
from src.prediction import forecast, load_production  # noqa: E402
from src.training import Dataset  # noqa: E402

SITE_DATA = APP_DATA_DIR
WORKFLOW_FILE = "daily_forecast.yml"
DEFAULT_REPO = "Valdi183/Oil_price_prediction"

SHORT_LABELS = {"corto": "Corto plazo", "medio": "Medio plazo", "largo": "Largo plazo"}
SUB_LABELS = {"corto": "5 días hábiles · ~1 semana", "medio": "63 días hábiles · ~3 meses",
              "largo": "252 días hábiles · ~12 meses"}
RELIABILITY_LABELS = {"media-alta": "Alta", "media": "Media", "baja": "Baja", "desconocida": "—"}
FOLD_LABELS = ["2017–2018", "2019–2020 (COVID)", "2021–2022 (Ucrania)"]
COMBO = "Combinación"
ARCHS = ["RNN", "LSTM", "GRU"]


# ----------------------------------------------------------------------------- utilidades
# Campos que necesitan más de 2 decimales (p-valores, scores de CV, learning rate)
PRECISION = {"R2_precio": 3, "lr": 4, "p_valor": 6, "p_vs_ingenuo": 6, "score": 3, "score_cv": 3, "std_cv_folds": 3, "std_cv_semillas": 3}


def rnd(x, nd: int = 2):
    """Redondea recursivamente floats (NaN -> None) para que el JSON sea compacto."""
    if isinstance(x, dict):
        return {k: rnd(v, PRECISION.get(k, nd)) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [rnd(v, nd) for v in x]
    if isinstance(x, (np.bool_, bool)):
        return bool(x)
    if isinstance(x, (np.integer, int)):
        return int(x)
    if isinstance(x, (np.floating, float)):
        return None if math.isnan(x) else round(float(x), nd)
    if isinstance(x, (pd.Timestamp, datetime)):
        return x.strftime("%Y-%m-%d")
    return x


def dump(name: str, obj) -> None:
    SITE_DATA.mkdir(parents=True, exist_ok=True)
    path = SITE_DATA / name
    path.write_text(json.dumps(obj, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    print(f"  {name:15s} {path.stat().st_size / 1024:7.1f} kB")


def repo_slug() -> str:
    """owner/repo: de GITHUB_REPOSITORY (en Actions) o de `git remote -v`."""
    if os.environ.get("GITHUB_REPOSITORY"):
        return os.environ["GITHUB_REPOSITORY"]
    try:
        url = subprocess.run(["git", "remote", "get-url", "origin"], capture_output=True, text=True,
                             cwd=ROOT, check=True).stdout.strip()
        m = re.search(r"github\.com[:/]([^/]+)/([^/\s]+?)(?:\.git)?$", url)
        if m:
            return f"{m.group(1)}/{m.group(2)}"
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass
    return DEFAULT_REPO


def pct(a: float, b: float) -> float:
    return 100 * (a / b - 1)


def es_numbers(text: str) -> str:
    """Coma decimal en los textos generados en Python ("92.14 $" -> "92,14 $")."""
    return re.sub(r"(?<=\d)\.(?=\d)", ",", text)


# ----------------------------------------------------------------------------- forecast.json
def export_forecast(ds: Dataset, meta: dict, sig: pd.DataFrame) -> dict:
    fc = forecast(ds, load_production())
    brent = ds.df["brent"]
    p0, last_date = fc["last_price"], fc["last_date"]
    metrics = {m["horizonte"]: dict(m) for m in meta["test_metrics"]}
    full = full_metrics(load_test_predictions()).set_index(["modelo", "horizonte"])
    for name, m in metrics.items():          # añade MSE y R² del modelo de producción
        if (meta["arch"], name) in full.index:
            f = full.loc[(meta["arch"], name)]
            m.update({"R2_precio": float(f["R2_precio"]), "R2_OOS_%": float(f["R2_OOS_%"])})
    vix = float(ds.df["vix"].iloc[-1])

    horizons = {}
    for name, f in fc["horizons"].items():
        row = sig[(sig["modelo"] == meta["arch"]) & (sig["frente_a"] == "Ingenuo") & (sig["horizonte"] == name)]
        p_value = float(row["p_valor"].iloc[0]) if len(row) else None
        level, rel_text = reliability(metrics.get(name), p_value)
        horizons[name] = {
            "key": name, "h": f["h"], "label": SHORT_LABELS[name], "sublabel": SUB_LABELS[name],
            "target_date": f["target_date"], "price": f["price"], "change_pct": f["change_pct"],
            "change_abs": f["price"] - p0, "low": f["low"], "high": f["high"],
            "width_pct": 100 * (f["high"] - f["low"]) / p0,
            "reliability": {"level": level, "label": RELIABILITY_LABELS[level], "text": es_numbers(rel_text)},
            "explanation": es_numbers(explain(name, f, p0, metrics.get(name), vix, p_value)),
            "metrics": metrics.get(name),
            "significance": None if not len(row) else {
                "mejora_RMSE_%": float(row["mejora_RMSE_%"].iloc[0]), "p_valor": round(p_value, 6),
                "significativo_5%": bool(row["significativo_5%"].iloc[0]),
                "significativo_10%": p_value < 0.10},
        }

    def back(n: int):
        return float(brent.iloc[-1 - n]) if len(brent) > n else None

    out = {
        "last_date": last_date, "last_price": p0,
        "changes": {
            "day_abs": p0 - back(1), "day_pct": pct(p0, back(1)),
            "month_pct": pct(p0, back(21)), "quarter_pct": pct(p0, back(63)), "year_pct": pct(p0, back(252)),
            "year_high": float(brent.iloc[-252:].max()), "year_low": float(brent.iloc[-252:].min()),
        },
        "vix": vix,
        "interval_level": int(round(100 * (INTERVAL_HIGH_Q - INTERVAL_LOW_Q))),
        "recent_movements": es_numbers(recent_movements(brent)),
        "horizons": horizons,
    }
    dump("forecast.json", rnd(out))
    return out


# ----------------------------------------------------------------------------- history.json
def export_history(last_date) -> pd.Series:
    raw = clean(load_raw()).loc[:last_date]
    brent = raw["brent"]
    out = {
        "dates": [d.strftime("%Y-%m-%d") for d in raw.index],
        "brent": rnd(brent.tolist()),
        "wti": rnd(raw["wti"].tolist()),
        "ma50": rnd(brent.rolling(50).mean().tolist()),
        "ma200": rnd(brent.rolling(200).mean().tolist()),
    }
    dump("history.json", out)
    return brent


# ----------------------------------------------------------------------------- events.json
def export_events(brent: pd.Series) -> None:
    ev = locate_events(brent)
    rows = [{"id": r["id"], "kind": r["kind"], "date": r["date"], "price": r["price"], "title": r["title"],
             "cycle": r["cycle"], "text": r["text"], "window": list(r["window"])}
            for _, r in ev.iterrows()]
    dump("events.json", rnd(rows))


# ----------------------------------------------------------------------------- models.json
def _direction_acc(p0: np.ndarray, pred: np.ndarray, real: np.ndarray) -> float:
    moved = real != p0
    return 100 * float(np.mean(np.sign(pred[moved] - p0[moved]) == np.sign(real[moved] - p0[moved])))


def export_models(meta: dict, sig: pd.DataFrame, ds_brent: pd.Series) -> None:
    comp = pd.read_csv(TABLES_DIR / "comparison.csv")
    equal = pd.read_csv(TABLES_DIR / "comparison_equal_size.csv")
    cv = pd.read_csv(TABLES_DIR / "cv_results.csv")
    tp = pd.read_csv(TABLES_DIR / "test_predictions.csv", parse_dates=["fecha_origen"])

    comp_cols = ["red", "horizonte", "dias", "MAE", "RMSE", "MAE_ingenuo", "RMSE_ingenuo", "mejora_RMSE_%",
                 "mejora_media_miembros_%", "std_mejora_RMSE_%", "mejora_MAE_%", "acierto_direccion_%",
                 "cobertura_IC80_%", "parametros", "epocas", "tiempo_entrenamiento_s", "hiperparametros",
                 "score_cv", "std_cv_folds", "std_cv_semillas"]
    rows = comp[comp_cols].to_dict("records")
    for r in rows:
        r["reference"] = False
        r["selected"] = r["red"] == meta["arch"]

    # Fila de la combinación (media de las tres redes): solo referencia, se recalcula de las
    # predicciones de test guardadas por scripts/04_significance_test.py.
    nets = comp.drop_duplicates("red").set_index("red")
    p0 = tp["brent_origen"].values
    for name, h in HORIZONS.items():
        real, pred = tp[f"real_{name}"].values, tp[f"{COMBO}_{name}"].values
        m = ~np.isnan(real)
        err, err_n = pred[m] - real[m], p0[m] - real[m]
        rmse, rmse_n = float(np.sqrt(np.mean(err ** 2))), float(np.sqrt(np.mean(err_n ** 2)))
        mae, mae_n = float(np.mean(np.abs(err))), float(np.mean(np.abs(err_n)))
        rows.append({
            "red": COMBO, "horizonte": name, "dias": h, "MAE": mae, "RMSE": rmse,
            "MAE_ingenuo": mae_n, "RMSE_ingenuo": rmse_n,
            "mejora_RMSE_%": 100 * (1 - rmse / rmse_n), "mejora_MAE_%": 100 * (1 - mae / mae_n),
            "acierto_direccion_%": _direction_acc(p0[m], pred[m], real[m]),
            "parametros": int(nets["parametros"].sum()),
            "tiempo_entrenamiento_s": float(nets["tiempo_entrenamiento_s"].sum()),
            "hiperparametros": "media de las tres redes", "reference": True, "selected": False,
        })

    # MSE y R² (sobre el precio y fuera de muestra frente al ingenuo) de src.evaluation
    full = full_metrics(tp).set_index(["modelo", "horizonte"])
    for r in rows:
        f = full.loc[(r["red"], r["horizonte"])]
        r.update({"MSE": float(f["MSE"]), "R2_precio": float(f["R2_precio"]), "R2_OOS_%": float(f["R2_OOS_%"])})

    models = []
    for arch in ARCHS:
        r = nets.loc[arch]
        hp = dict(re.findall(r"([a-z]+)([\d.e-]+)", r["hiperparametros"]))
        cv_row = cv[(cv["red"] == arch) & (cv["hidden"] == int(hp["h"])) & (cv["layers"] == int(hp["l"]))]
        folds = [float(cv_row[f"score_fold{i}"].iloc[0]) for i in (1, 2, 3)] if len(cv_row) else []
        models.append({
            "red": arch, "selected": arch == meta["arch"], "hiperparametros": r["hiperparametros"],
            "hidden": int(hp["h"]), "layers": int(hp["l"]), "lr": float(hp["lr"]),
            "parametros": int(r["parametros"]), "epocas": int(r["epocas"]),
            "tiempo_entrenamiento_s": float(r["tiempo_entrenamiento_s"]),
            "score_cv": float(r["score_cv"]), "std_cv_folds": float(r["std_cv_folds"]),
            "std_cv_semillas": float(r["std_cv_semillas"]),
            "cv_folds": [{"label": FOLD_LABELS[i], "score": sc} for i, sc in enumerate(folds)],
            "mejora": {h: float(comp[(comp["red"] == arch) & (comp["horizonte"] == h)]["mejora_RMSE_%"].iloc[0])
                       for h in HORIZONS},
            "std_mejora": {h: float(comp[(comp["red"] == arch) & (comp["horizonte"] == h)]["std_mejora_RMSE_%"].iloc[0])
                           for h in HORIZONS},
            "p_vs_ingenuo": {h: float(sig[(sig["modelo"] == arch) & (sig["frente_a"] == "Ingenuo")
                                         & (sig["horizonte"] == h)]["p_valor"].iloc[0]) for h in HORIZONS},
        })

    sig_rows = sig.to_dict("records")
    for r in sig_rows:
        r["significativo_10%"] = bool(r["p_valor"] < 0.10)
        r["significativo_1%"] = bool(r["p_valor"] < 0.01)

    preds = {"dates": [d.strftime("%Y-%m-%d") for d in tp["fecha_origen"]],
             "brent_origen": rnd(tp["brent_origen"].tolist())}
    for name in HORIZONS:
        preds[name] = {"real": rnd(tp[f"real_{name}"].tolist())}
        for net in ARCHS + [COMBO]:
            preds[name][net] = rnd(tp[f"{net}_{name}"].tolist())

    out = {
        "selected": meta["arch"],
        "test_period": {"start": preds["dates"][0], "end": meta["data_until"]},
        "cv_folds": [{"label": FOLD_LABELS[i], "start": a, "end": b} for i, (a, b) in enumerate(CV_FOLDS)],
        "models": rnd(models),
        "comparison": rnd(rows),
        "equal_size": rnd(equal.to_dict("records")),
        "significance": rnd(sig_rows),
        "test_predictions": preds,
        "validation": validation_block(tp, full, ds_brent),
    }
    dump("models.json", out)


def validation_block(tp: pd.DataFrame, full: pd.DataFrame, brent: pd.Series) -> dict:
    """Validación fuera de muestra de la RNN: métricas completas (incluido el ingenuo), errores
    por año y por régimen de volatilidad, cociente de RMSE móvil frente al ingenuo y peores fallos."""
    years = tp["fecha_origen"].dt.year.astype(str)
    reg, (q1, q2), _ = regimes(tp, brent)
    periods = pd.concat([by_group(tp, years, "año"), by_group(tp, reg, "régimen")], ignore_index=True)
    roll = rolling_ratio(tp)
    worst = worst_errors(tp)
    worst["fecha_origen"] = worst["fecha_origen"].astype(str)
    worst["fecha_objetivo"] = worst["fecha_objetivo"].astype(str)
    beats = {hz: 100 * float(np.mean(r["ratio"][~np.isnan(r["ratio"])] < 1)) for hz, r in roll.items()}
    return {
        "metrics": rnd(full.reset_index().to_dict("records"), 3),
        "years": sorted(periods[periods.tipo == "año"].grupo.unique().tolist()),
        "regimes": REGIMES, "regime_thresholds": rnd([q1, q2], 0),
        "periods": rnd(periods.to_dict("records")),
        "rolling": {hz: {"window": r["window"], "ratio": rnd(r["ratio"].tolist(), 3), "beats_pct": beats[hz]}
                    for hz, r in roll.items()},
        "worst": rnd(worst.to_dict("records")),
    }


# ----------------------------------------------------------------------------- tools.json
def export_tools(ds: Dataset, fc: dict) -> None:
    """Simulador a 1-252 días, distribución del error, escenarios históricos y backtest
    (src/decision_tools.py). La web solo los combina con lo que escribe el usuario."""
    brent = ds.df["brent"]
    tp = pd.read_csv(TABLES_DIR / "test_predictions.csv", parse_dates=["fecha_origen"])
    out = {
        "last_date": fc["last_date"], "last_price": fc["last_price"], "contract_barrels": 1000,
        "simulator": simulator_grid(fc, load_forecast_errors()),
        "historical": historical_moves(brent),
        "scenarios": crisis_scenarios(brent),
        "backtest": backtest_trades(tp, brent),
    }
    dump("tools.json", rnd(out, 5))


# ----------------------------------------------------------------------------- meta.json
def export_meta(ds: Dataset, meta: dict, brent: pd.Series) -> None:
    slug = repo_slug()
    owner, repo = slug.split("/")
    out = {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "data_until": ds.index[-1], "model_data_until": meta["data_until"], "data_from": brent.index[0], "n_days": int(len(brent)),
        "years_of_data": int(brent.index[-1].year - brent.index[0].year),
        "trained_at": meta["trained_at"], "train_seconds": meta["train_seconds"],
        "arch": meta["arch"], "arch_label": f"{meta['arch']} {meta['params']['hidden']}×{meta['params']['layers']}",
        "params": meta["params"], "n_params": meta["n_params"], "n_members": meta["n_members"],
        "seeds": meta["seeds"], "n_epochs": meta["n_epochs"], "window": meta.get("window", WINDOW),
        "features": meta["features"], "n_features": len(meta["features"]), "loss": "Huber",
        "horizons": meta["horizons"], "test_start": TEST_START,
        "selected_by": meta.get("selected_by"), "score_cv": meta.get("score_cv"),
        "repo": slug, "repo_url": f"https://github.com/{slug}",
        "action_url": f"https://github.com/{slug}/actions/workflows/{WORKFLOW_FILE}",
        "pages_url": f"https://{owner.lower()}.github.io/{repo}/",
        "sources": [
            {"name": "FRED (EIA, BCE, Fed, Cboe)", "url": "https://fred.stlouisfed.org",
             "detail": "Brent y WTI spot, gas natural, tipo BCE, dólar/euro, yuan/dólar, VIX"},
            {"name": "Yahoo Finance", "url": "https://finance.yahoo.com",
             "detail": "futuros del Brent y del WTI, cobre, oro e índice del dólar"},
        ],
    }
    dump("meta.json", rnd(out))


def main() -> None:
    print("Exportando la web a", SITE_DATA.relative_to(ROOT))
    ds = Dataset.load()
    _, _, meta = load_production()
    sig = pd.read_csv(TABLES_DIR / "significance.csv")
    fc = export_forecast(ds, meta, sig)
    brent = export_history(fc["last_date"])
    export_events(brent)
    export_models(meta, sig, ds.df["brent"])
    export_tools(ds, fc)
    export_meta(ds, meta, brent)
    print(f"Datos hasta {fc['last_date']:%Y-%m-%d} · modelo {meta['arch']} entrenado el {meta['trained_at']}")


if __name__ == "__main__":
    main()
