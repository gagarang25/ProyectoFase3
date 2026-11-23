"""Pruebas del propio pipeline (Fase 3): que las compuertas detecten lo que deben detectar.

Una compuerta que nunca falla no protege nada. Aquí se corrompen copias de los datos a propósito y se
comprueba que la validación las rechaza, y se verifica que la evaluación del pipeline calcula el ETA
EXACTAMENTE igual que la API (sin training-serving skew, deuda técnica señalada en la A5).
"""

from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ / "pipeline"))
sys.path.insert(0, str(RAIZ / "backend"))

import validar_datos as vd  # noqa: E402
from comun import Reporte, cargar_yaml  # noqa: E402
from evaluar_modelos import predecir_eta  # noqa: E402

CONTRATO = cargar_yaml("contrato_entregas.yaml")


@pytest.fixture(scope="module")
def entregas() -> pd.DataFrame:
    return pd.read_csv(RAIZ / "datos" / "entregas" / "piloto_ab_cab.csv")


def validar(df: pd.DataFrame) -> Reporte:
    rep = Reporte("prueba")
    vd.verificar_entregas(rep, df, "copia", CONTRATO)
    return rep


def fallidas(rep: Reporte) -> list[str]:
    return [v.nombre for v in rep.verificaciones if not v.aprobada and v.nivel == "error"]


def test_datos_versionados_pasan(entregas):
    assert validar(entregas).aprobado


@pytest.mark.parametrize("corromper,esperada", [
    (lambda d: d.assign(transportista=d["transportista"].where(d.index != 3, "TR-Z")), "categorías"),
    (lambda d: d.assign(distancia_km=d["distancia_km"].where(d.index != 5, -10)), "rangos"),
    (lambda d: d.assign(retraso=1 - d["retraso"]), "retraso_coherente"),
    (lambda d: pd.concat([d, d.iloc[[0]]]), "llave única"),
    (lambda d: d.drop(columns=["holgura_dias"]), "columnas"),
    (lambda d: d.assign(fecha_pedido=d["fecha_pedido"].where(d.index != 1, "31/13/2026")), "fecha_pedido"),
    (lambda d: d.assign(cliente=d["cliente"].where(d.index != 2, None)), "nulos"),
])
def test_datos_corruptos_se_rechazan(entregas, corromper, esperada):
    rep = validar(corromper(entregas.copy()).reset_index(drop=True))
    assert not rep.aprobado
    assert any(esperada in n or esperada in v.detalle for n in fallidas(rep)
               for v in rep.verificaciones if v.nombre == n), fallidas(rep)


def test_manifiesto_detecta_archivo_modificado(tmp_path, monkeypatch):
    origen = RAIZ / "datos" / "entregas"
    (tmp_path / "datos" / "entregas").mkdir(parents=True)
    for n in ("piloto_ab_cab.csv", "versiones.json"):
        (tmp_path / "datos" / "entregas" / n).write_bytes((origen / n).read_bytes())
    m = json.loads((origen / "versiones.json").read_text(encoding="utf-8"))
    m["archivos"] = {"piloto_ab_cab.csv": m["archivos"]["piloto_ab_cab.csv"]}
    (tmp_path / "datos" / "entregas" / "versiones.json").write_text(json.dumps(m), encoding="utf-8")
    monkeypatch.setattr(vd, "RAIZ", tmp_path)
    rep = Reporte("manifiesto")
    vd.verificar_manifiestos(rep, ["datos/entregas/versiones.json"])
    assert rep.aprobado
    with open(tmp_path / "datos" / "entregas" / "piloto_ab_cab.csv", "a", encoding="utf-8") as f:
        f.write("\n")                      # un solo salto de línea cambia el SHA-256
    rep = Reporte("manifiesto")
    vd.verificar_manifiestos(rep, ["datos/entregas/versiones.json"])
    assert not rep.aprobado


def test_drift_detecta_planta_saturada(entregas):
    meta = json.loads((RAIZ / "backend" / "modelos" / "metadatos_modelos.json").read_text(encoding="utf-8"))
    cfg = cargar_yaml("pipeline.yaml")["datos"]
    historico = pd.read_csv(RAIZ / "datos" / "entregas" / "entregas_cab.csv")
    rep = Reporte("drift")
    estable = vd.verificar_drift(rep, entregas, historico, cfg, meta["rangos_numericos_entrenamiento"])
    assert estable["fuera_de_rango_utilizacion"] <= cfg["fuera_de_rango_max"]   # piloto: 1.9 %
    saturada = entregas.assign(utilizacion_planta=np.clip(entregas["utilizacion_planta"] + 0.15, 0, 1.2))
    rep = Reporte("drift")
    con_drift = vd.verificar_drift(rep, saturada, historico, cfg, meta["rangos_numericos_entrenamiento"])
    assert con_drift["fuera_de_rango_utilizacion"] > cfg["fuera_de_rango_max"]
    assert any(not v.aprobada and v.nivel == "aviso" for v in rep.verificaciones)


def test_sin_training_serving_skew(entregas):
    """El pipeline evalúa con la misma lógica de ETA y rango que sirve la API."""
    import joblib

    from app.modelos_entrega import ModelosEntrega

    carpeta = RAIZ / "backend" / "modelos"
    api = ModelosEntrega(carpeta)
    modelos = {"eta": joblib.load(carpeta / "eta_mediana.joblib"), "inf": joblib.load(carpeta / "eta_rango_inferior.joblib"),
               "sup": joblib.load(carpeta / "eta_rango_superior.joblib")}
    muestra = entregas.sample(150, random_state=42).reset_index(drop=True)
    eta, lo, hi = predecir_eta(modelos, muestra[api.entradas], api.desfase_rango)
    for i, fila in muestra.iterrows():
        est = api.estimar({c: (None if pd.isna(fila[c]) else fila[c]) for c in api.entradas}, "es")
        assert (est.eta_dias, est.eta_min_dias, est.eta_max_dias) == (eta[i], lo[i], hi[i]), fila["id_orden"]


def test_umbrales_de_la_configuracion_son_coherentes():
    cfg = cargar_yaml("pipeline.yaml")
    assert cfg["modelos"]["eta"]["mae_max"] == cfg["ct"]["disparadores"]["mae_reciente_max"]  # mismo SLO (A8)
    assert 0 < cfg["modelos"]["riesgo"]["recall_min"] <= 0.80
    assert cfg["servicio"]["p95_ms_max"] == 400
    assert date.fromisoformat(cfg["modelos"]["corte_prueba"]) < date.fromisoformat(cfg["ct"]["inicio_produccion"])
