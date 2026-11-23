"""Entrenamiento continuo · paso 1: ingesta de un lote nuevo de entregas con etiqueta.

En CAB las etiquetas (días reales de entrega) llegan cuando la orden se entrega. Este paso toma las órdenes de
producción posteriores a data-v1.0.0 (desde el 01/10/2026) que ya se entregaron en la fecha de corte y las
guarda como una versión nueva de datos, con su SHA-256 en datos/entregas/lotes/versiones.json.

No hay un SAP real: el lote sale del MISMO generador y semilla de la A6/Fase 2/A8 (la historia es idéntica) y,
en el escenario "drift_planta", desde el 09/11/2026 las plantas suben 15 puntos de carga (incidente INC-01 de la
Actividad 8). Se reutiliza la función produccion() de monitoreo/src/monitoreo_semanal.py.

Uso:  python pipeline/ct/ingestar_lote.py --hasta 2026-11-30 [--escenario drift_planta|normal]
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ / "pipeline"))
sys.path.insert(0, str(RAIZ / "monitoreo" / "src"))
from comun import info_git, sha256  # noqa: E402

CARPETA = RAIZ / "datos" / "entregas" / "lotes"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hasta", required=True, help="fecha de corte (AAAA-MM-DD): entregas registradas hasta ese día")
    ap.add_argument("--escenario", choices=["drift_planta", "normal"], default="drift_planta")
    a = ap.parse_args()
    import monitoreo_semanal  # A8 (importa MLflow; por eso se carga aquí y no al inicio)

    hasta = pd.Timestamp(a.hasta)
    prod = monitoreo_semanal.produccion(con_drift=a.escenario == "drift_planta")
    prod["fecha_entrega_real"] = pd.to_datetime(prod["fecha_entrega_real"])
    lote = prod[prod["fecha_entrega_real"] <= hasta].copy()
    for c in ("fecha_pedido", "fecha_entrega_real"):
        lote[c] = pd.to_datetime(lote[c]).dt.strftime("%Y-%m-%d")
    CARPETA.mkdir(parents=True, exist_ok=True)
    nombre = f"entregas_produccion_{a.escenario}_hasta_{hasta:%Y-%m-%d}.csv"
    ruta = CARPETA / nombre
    lote.to_csv(ruta, index=False)

    manifiesto_ruta = CARPETA / "versiones.json"
    manifiesto = json.loads(manifiesto_ruta.read_text(encoding="utf-8")) if manifiesto_ruta.exists() else \
        {"descripcion": "Lotes de producción con etiqueta (entrenamiento continuo, Fase 3)", "archivos": {}}
    previos = [v for k, v in manifiesto["archivos"].items() if k != nombre]
    version = f"data-v1.{len(previos) + 1}.0"
    manifiesto["archivos"][nombre] = {
        "version": version, "filas": int(len(lote)), "columnas": int(lote.shape[1]),
        "periodo_pedido": [lote["fecha_pedido"].min(), lote["fecha_pedido"].max()],
        "entregadas_hasta": str(hasta.date()), "escenario": a.escenario,
        "tasa_retraso": round(float(lote["retraso"].mean()), 4), "sha256": sha256(ruta),
        "generador": "monitoreo/src/monitoreo_semanal.py::produccion (semilla 42, continuación de data-v1.0.0)",
        "creado_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "git_commit": info_git()["commit"],
    }
    manifiesto["version"] = version
    manifiesto_ruta.write_text(json.dumps(manifiesto, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"{version}: {len(lote):,} órdenes entregadas hasta {hasta.date()} → {ruta.relative_to(RAIZ)}")
    print(f"LOTE={ruta.relative_to(RAIZ)}")


if __name__ == "__main__":
    main()
