"""Principales picos y caídas del Brent y su relación con el ciclo económico mundial.
Las fechas y precios exactos se calculan sobre la propia serie (máximo/mínimo en la ventana)."""
from __future__ import annotations

import pandas as pd

EVENTS = [
    {
        "id": "2008_max", "kind": "max", "window": ("2008-05-01", "2008-08-31"),
        "title": "Máximo histórico de 2008",
        "cycle": "Final de una expansión mundial",
        "text": "El Brent alcanzó su máximo histórico tras años de fuerte crecimiento de la demanda "
                "de las economías emergentes (sobre todo China), un dólar débil, poca capacidad de "
                "producción sobrante y flujos financieros hacia las materias primas.",
    },
    {
        "id": "2008_crash", "kind": "min", "window": ("2008-11-01", "2009-03-31"),
        "title": "Desplome por la crisis financiera",
        "cycle": "Recesión mundial",
        "text": "Tras la quiebra de Lehman Brothers (septiembre de 2008) la economía mundial entró en "
                "recesión, la demanda de crudo se hundió y el precio perdió más de dos tercios de su "
                "valor en apenas cinco meses. Los recortes de la OPEP y los estímulos lo recuperaron en 2009.",
    },
    {
        "id": "2011_plateau", "kind": "max", "window": ("2011-01-01", "2012-12-31"),
        "title": "Primavera Árabe y meseta de 110 $",
        "cycle": "Recuperación post-crisis",
        "text": "La guerra de Libia y la inestabilidad en Oriente Medio, unidas a la recuperación "
                "de la demanda, mantuvieron el Brent en torno a 100-120 $ entre 2011 y mediados de 2014.",
    },
    {
        "id": "2016_min", "kind": "min", "window": ("2015-11-01", "2016-03-31"),
        "title": "Caída 2014-2016 por exceso de oferta",
        "cycle": "Desaceleración de emergentes y exceso de oferta",
        "text": "El auge del petróleo de esquisto (shale) en EE. UU., la decisión de la OPEP en "
                "noviembre de 2014 de no recortar producción para defender cuota, la desaceleración "
                "de China y un dólar fuerte hundieron el precio desde más de 110 $ hasta menos de 30 $.",
    },
    {
        "id": "2020_min", "kind": "min", "window": ("2020-03-01", "2020-05-31"),
        "title": "Desplome por la pandemia",
        "cycle": "Recesión por la COVID-19",
        "text": "Los confinamientos paralizaron el transporte y la demanda cayó cerca de un 20 %, "
                "coincidiendo con la guerra de precios entre Arabia Saudí y Rusia. El WTI llegó a "
                "cotizar en negativo (20/04/2020) por falta de capacidad de almacenamiento. El "
                "acuerdo de recortes OPEP+ y la reapertura impulsaron la recuperación.",
    },
    {
        "id": "2022_max", "kind": "max", "window": ("2022-02-15", "2022-07-31"),
        "title": "Subida tras la invasión de Ucrania",
        "cycle": "Reapertura post-COVID + shock geopolítico",
        "text": "La invasión rusa de Ucrania (24/02/2022) y las sanciones a Rusia, uno de los mayores "
                "exportadores del mundo, se sumaron a una demanda ya recuperada tras la pandemia. "
                "Las subidas de tipos de los bancos centrales y el miedo a la recesión enfriaron "
                "el precio en la segunda mitad de 2022.",
    },
    {
        "id": "2025_min", "kind": "min", "window": ("2025-03-01", "2025-12-31"),
        "title": "Debilidad de 2025",
        "cycle": "Tensiones comerciales y más oferta OPEP+",
        "text": "Las tensiones arancelarias y la reversión gradual de los recortes de producción de "
                "la OPEP+ llevaron el Brent a la zona de 60-65 $, sus niveles más bajos desde 2021.",
    },
    {
        "id": "2026_max1", "kind": "max", "window": ("2026-03-01", "2026-05-31"),
        "title": "Guerra con Irán y crisis de Ormuz (2026)",
        "cycle": "Shock de oferta geopolítico",
        "text": "Tras los ataques de EE. UU. e Israel contra Irán (28/02/2026) se interrumpió el "
                "tráfico por el estrecho de Ormuz, por donde pasa en torno a una cuarta parte del "
                "comercio marítimo de crudo. El Banco Mundial lo calificó como el mayor shock de "
                "oferta de la historia; en marzo el Brent registró la mayor subida mensual desde 1988. "
                "El alto el fuego temporal de abril moderó el precio.",
    },
    {
        "id": "2026_max2", "kind": "max", "window": ("2026-09-01", "2026-11-30"),
        "title": "Nueva escalada en otoño de 2026",
        "cycle": "Riesgo geopolítico persistente",
        "text": "El temor a nuevos ataques sobre Irán y los ataques a infraestructuras energéticas "
                "en el Golfo devolvieron el Brent por encima de 110 $; octubre de 2026 fue el mes "
                "con el precio medio más alto desde 2022.",
    },
]


def locate_events(brent: pd.Series) -> pd.DataFrame:
    """Sitúa cada evento en el máximo/mínimo real de la serie dentro de su ventana."""
    rows = []
    for ev in EVENTS:
        s = brent.loc[ev["window"][0]:ev["window"][1]]
        if s.empty:
            continue
        date = s.idxmax() if ev["kind"] == "max" else s.idxmin()
        rows.append({**ev, "date": date, "price": float(s.loc[date])})
    return pd.DataFrame(rows)


def recent_movements(brent: pd.Series) -> str:
    """Texto automático sobre los movimientos más recientes de la serie."""
    last_date, last = brent.index[-1], float(brent.iloc[-1])
    ytd = brent.loc[str(last_date.year)]
    lines = []
    for label, days in [("1 mes", 21), ("3 meses", 63), ("12 meses", 252)]:
        if len(brent) > days:
            lines.append(f"{label}: {100 * (last / brent.iloc[-1 - days] - 1):+.1f} %")
    last12 = brent.iloc[-252:]
    hi_d, lo_d = last12.idxmax(), last12.idxmin()
    return (
        f"Último dato ({last_date:%d/%m/%Y}): **{last:.2f} $/barril**. Variación en "
        + ", ".join(lines) + ". "
        f"En los últimos 12 meses el máximo fue {last12.max():.2f} $ ({hi_d:%d/%m/%Y}) y el mínimo "
        f"{last12.min():.2f} $ ({lo_d:%d/%m/%Y}); el precio actual está un "
        f"{100 * (1 - last / last12.max()):.0f} % por debajo de ese máximo. "
        f"Media del año {last_date.year}: {ytd.mean():.2f} $."
    )
