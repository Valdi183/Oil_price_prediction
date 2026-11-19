"""Explicación en lenguaje sencillo de una predicción, para un usuario no técnico."""
from __future__ import annotations

HORIZON_TEXT = {
    "corto": "dentro de una semana (5 días hábiles)",
    "medio": "dentro de unos tres meses (63 días hábiles)",
    "largo": "dentro de un año (252 días hábiles)",
}

FACTORS = {
    "corto": [
        "noticias geopolíticas repentinas (ataques a infraestructuras, cierres de rutas marítimas como Ormuz)",
        "anuncios inesperados de la OPEP+ sobre producción",
        "datos semanales de inventarios de EE. UU. muy distintos de lo esperado",
    ],
    "medio": [
        "cambios en la política de producción de la OPEP+",
        "la evolución de los conflictos en Oriente Medio y de las sanciones a Rusia e Irán",
        "un cambio de rumbo de la economía mundial (China, EE. UU., Europa) o de los tipos de interés",
    ],
    "largo": [
        "una recesión mundial o, al contrario, una fuerte aceleración económica",
        "decisiones estratégicas de la OPEP+ y la inversión en producción (p. ej. shale en EE. UU.)",
        "la transición energética y cambios estructurales en la demanda",
        "acontecimientos imposibles de anticipar (pandemias, guerras)",
    ],
}


def reliability(metrics: dict | None, p_value: float | None = None) -> tuple[str, str]:
    """Devuelve (etiqueta, frase) según la comparativa con el modelo ingenuo en test.
    p_value: p-valor del test de Diebold-Mariano frente al ingenuo (reports/tables/significance.csv).
    Si se conoce, una mejora que no es estadísticamente significativa no cuenta como fiable."""
    if not metrics:
        return "desconocida", "No hay métricas de evaluación disponibles para este horizonte."
    imp, acc = metrics["mejora_RMSE_%"], metrics["acierto_direccion_%"]
    base = (f"En la evaluación con datos de 2023 en adelante, el modelo se equivocó de media en "
            f"{metrics['MAE']:.1f} $/barril en este horizonte, frente a {metrics['MAE_ingenuo']:.1f} $ "
            f"si se supone que el precio no cambia, y acertó la dirección (sube/baja) el {acc:.0f} % de las veces. ")
    significant = p_value is None or p_value < 0.10
    if imp > 5 and significant:
        return "media-alta", base + ("Aquí el modelo **aporta valor**: mejora claramente a la referencia "
                                    "de no cambio.")
    if imp > 5 and not significant:
        return "baja", base + (f"La mejora media (+{imp:.1f} %) **no es estadísticamente significativa** "
                              f"(p = {p_value:.2f}) y varía mucho de una red a otra, así que en este horizonte "
                              "la predicción **no es fiable**: fíjate en el rango, no en la cifra.")
    if imp > -5:
        extra = ""
        if p_value is not None and p_value < 0.05 and imp > 0:
            extra = (f" La mejora es pequeña pero **demostrada estadísticamente** (p = {p_value:.2f}), "
                     "lo bastante para orientar la tendencia.")
        return "media", base + ("Su precisión es **similar** a suponer que el precio no cambia: "
                               "úsalo como una orientación de tendencia, no como una cifra exacta." + extra)
    return "baja", base + ("En este horizonte el modelo **no supera** a la referencia de no cambio: "
                          "la predicción debe tomarse con mucha cautela y apoyarse en otros análisis.")


def explain(horizon: str, fc: dict, last_price: float, metrics: dict | None, vix: float | None = None,
            p_value: float | None = None) -> str:
    chg = fc["change_pct"]
    if abs(chg) < 1.5:
        direction = "se mantenga prácticamente estable"
    elif chg > 0:
        direction = f"**suba un {chg:.1f} %**"
    else:
        direction = f"**baje un {abs(chg):.1f} %**"
    width = 100 * (fc["high"] - fc["low"]) / last_price
    level, rel_text = reliability(metrics, p_value)

    parts = [
        f"**Qué dice la predicción.** El modelo estima que el Brent {direction} {HORIZON_TEXT[horizon]}, "
        f"desde {last_price:.2f} $ hasta unos **{fc['price']:.2f} $/barril** "
        f"(fecha objetivo aproximada: {fc['target_date']:%d/%m/%Y}).",
        f"**Rango de incertidumbre.** En 8 de cada 10 casos parecidos del pasado el precio real "
        f"quedó entre **{fc['low']:.2f} $ y {fc['high']:.2f} $**. Es un margen de ±{width / 2:.0f} % "
        f"aproximadamente: cuanto más lejano el horizonte, más ancho es el rango.",
        f"**Fiabilidad ({level}).** {rel_text}",
    ]
    ctx = ""
    if vix is not None and vix > 25:
        ctx = (f" Además, la volatilidad de los mercados es alta ahora mismo (VIX = {vix:.0f}), "
               "lo que aumenta la probabilidad de movimientos bruscos.")
    factors = "; ".join(FACTORS[horizon])
    parts.append("**Qué podría invalidarla.** " + factors[0].upper() + factors[1:] + "." + ctx)
    parts.append("_Herramienta de apoyo a la decisión: no es una recomendación de inversión._")
    return "\n\n".join(parts)
