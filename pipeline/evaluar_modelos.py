"""Pruebas automatizadas de MODELOS (compuerta 2 del pipeline): ETA y riesgo de retraso.

Qué se verifica con los modelos que la API va a servir (backend/modelos/):
  1. Integridad: el SHA-256 de cada .joblib coincide con metadatos_modelos.json (nadie cambió un modelo a mano).
  2. Reproducibilidad: las métricas recalculadas coinciden con las registradas al entrenar (misma versión de
     scikit-learn, mismos datos, mismo código).
  3. Desempeño mínimo (pipeline.yaml): MAE y cobertura del rango del ETA, mejora frente a la fecha de SAP,
     AUC, recall y precisión del riesgo de retraso.
  4. Equidad por segmento (A9, Tema 22): recall del riesgo y MAE del ETA por transportista.

Conjuntos de evaluación: por defecto la prueba temporal de la Fase 2 (pedidos desde el 01/05/2026). Si el
modelo se reentrenó con el pipeline de CT, metadatos_modelos.json declara además el lote de validación reciente
("evaluaciones"), que ningún modelo vio al entrenar, y las compuertas de ETA se aplican sobre ese lote.

Uso:  python pipeline/evaluar_modelos.py [--modelos backend/modelos]
Salida: pipeline/reportes/evaluacion_modelos.{json,md}; código 1 si una compuerta de nivel "error" falla.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from comun import RAIZ, Reporte, cargar_yaml, sha256  # noqa: E402

SALIDA = RAIZ / "pipeline" / "reportes"


def predecir_eta(modelos: dict, X: pd.DataFrame, desfase: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Misma lógica de redondeo y rango que la API (backend/app/modelos_entrega.py)."""
    med = modelos["eta"].predict(X)
    eta = np.maximum(1, np.rint(med))
    lo = np.maximum(1, np.rint(np.minimum(modelos["inf"].predict(X), med)) - desfase)
    hi = np.maximum(eta, np.rint(np.maximum(modelos["sup"].predict(X), med)) + desfase)
    return eta, lo, hi


def metricas_eta(modelos, df, entradas, desfase) -> dict:
    eta, lo, hi = predecir_eta(modelos, df[entradas], desfase)
    y, sap = df["dias_reales"].to_numpy(), df["dias_comprometidos"].to_numpy()
    mae = float(np.mean(np.abs(eta - y)))
    mae_sap = float(np.mean(np.abs(sap - y)))
    return {"ordenes": int(len(df)), "mae_dias": mae, "cobertura_rango": float(np.mean((y >= lo) & (y <= hi))),
            "dentro_1_dia": float(np.mean(np.abs(eta - y) <= 1)), "mae_linea_base_sap": mae_sap,
            "mejora_vs_sap": 1 - mae / mae_sap, "ancho_promedio_rango": float(np.mean(hi - lo))}


def metricas_riesgo(modelo, df, entradas, umbral) -> dict:
    p = modelo.predict_proba(df[entradas])[:, 1]
    alerta = (p >= umbral).astype(int)
    y = df["retraso"].to_numpy()
    return {"auc": float(roc_auc_score(y, p)), "recall": float(recall_score(y, alerta)),
            "precision": float(precision_score(y, alerta)), "falsos_negativos": int(((alerta == 0) & (y == 1)).sum()),
            "tasa_alertas": float(alerta.mean())}


def conjuntos(meta: dict, cfg: dict) -> dict[str, dict]:
    """Conjuntos declarados en los metadatos; por defecto, la prueba temporal de la Fase 2."""
    declarados = meta.get("evaluaciones") or {
        "prueba_historica": {"archivo": "datos/entregas/entregas_cab.csv", "columna": "fecha_pedido",
                             "desde": cfg["modelos"]["corte_prueba"], "principal_eta": True}}
    salida = {}
    for nombre, d in declarados.items():
        df = pd.read_csv(RAIZ / d["archivo"], parse_dates=["fecha_pedido", "fecha_entrega_real"])
        if d.get("desde"):
            df = df[df[d["columna"]] >= pd.Timestamp(d["desde"])]
        if d.get("hasta"):
            df = df[df[d["columna"]] <= pd.Timestamp(d["hasta"])]
        salida[nombre] = {**d, "df": df.reset_index(drop=True)}
    return salida


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--modelos", type=Path, default=RAIZ / "backend" / "modelos")
    ap.add_argument("--salida", type=Path, default=SALIDA)
    a = ap.parse_args()
    cfg = cargar_yaml("pipeline.yaml")
    g = cfg["modelos"]
    meta = json.loads((a.modelos / "metadatos_modelos.json").read_text(encoding="utf-8"))
    rep = Reporte(f"Pruebas de modelos ({meta['version_modelos']})")

    # 1. Integridad
    for nombre, esperado in meta["archivos"].items():
        ok = (a.modelos / nombre).exists() and sha256(a.modelos / nombre) == esperado
        rep.agregar(f"sha256 {nombre}", ok, "coincide con metadatos" if ok else "el modelo no es el registrado")
    if not rep.aprobado:
        rep.guardar(a.salida, "evaluacion_modelos")
        sys.exit(1)

    modelos = {"eta": joblib.load(a.modelos / "eta_mediana.joblib"),
               "inf": joblib.load(a.modelos / "eta_rango_inferior.joblib"),
               "sup": joblib.load(a.modelos / "eta_rango_superior.joblib")}
    riesgo = joblib.load(a.modelos / "riesgo_retraso.joblib")
    entradas, umbral = meta["entradas"], float(meta["riesgo_retraso"]["umbral"])
    desfase = int(meta["eta"].get("desfase_rango_dias", 0))
    tol = g["reproducibilidad_tolerancia"]
    resultados = {}

    for nombre, c in conjuntos(meta, cfg).items():
        df = c["df"]
        m_eta = metricas_eta(modelos, df, entradas, desfase)
        resultados[nombre] = {"eta": m_eta}
        esperado = c.get("metricas_eta") or (meta["eta"]["metricas_prueba"] if nombre == "prueba_historica" else None)
        if esperado:
            dif = abs(m_eta["mae_dias"] - esperado["mae_dias"]) + abs(m_eta["cobertura_rango"] - esperado["cobertura_rango"])
            rep.agregar(f"{nombre}: reproducibilidad ETA", dif <= tol,
                        f"MAE {m_eta['mae_dias']:.4f} vs registrado {esperado['mae_dias']:.4f}", valor=dif, umbral=tol)
        if c.get("principal_eta"):
            e, nivel = g["eta"], c.get("nivel", "error")
            if "mae_modelo_anterior" in c:      # modelo reentrenado (CT): debe mejorar al anterior en el mismo lote
                minimo = cfg["ct"]["promocion"]["mejora_mae_min"]
                mejora = c["mae_modelo_anterior"] - m_eta["mae_dias"]
                rep.agregar(f"{nombre}: mejora frente al modelo anterior", mejora >= minimo,
                            f"MAE {c['mae_modelo_anterior']:.3f} → {m_eta['mae_dias']:.3f} días", valor=mejora,
                            umbral=minimo)
            rep.agregar(f"{nombre}: MAE del ETA", m_eta["mae_dias"] <= e["mae_max"],
                        f"{m_eta['mae_dias']:.3f} días en {m_eta['ordenes']:,} órdenes", nivel=nivel,
                        valor=m_eta["mae_dias"], umbral=e["mae_max"])
            rep.agregar(f"{nombre}: cobertura del rango", m_eta["cobertura_rango"] >= e["cobertura_min"],
                        f"{m_eta['cobertura_rango']:.1%} de entregas dentro del rango (KPI R08)", nivel=nivel,
                        valor=m_eta["cobertura_rango"], umbral=e["cobertura_min"])
            rep.agregar(f"{nombre}: mejora frente a SAP", m_eta["mejora_vs_sap"] >= e["mejora_vs_sap_min"], nivel=nivel,
                        detalle=
                        f"{m_eta['mejora_vs_sap']:.1%} menos error que la fecha de SAP "
                        f"(MAE SAP {m_eta['mae_linea_base_sap']:.2f})", valor=m_eta["mejora_vs_sap"],
                        umbral=e["mejora_vs_sap_min"])
        if c.get("principal_riesgo", nombre == "prueba_historica"):
            m_r = metricas_riesgo(riesgo, df, entradas, umbral)
            resultados[nombre]["riesgo"] = m_r
            r = g["riesgo"]
            esperado_r = meta["riesgo_retraso"]["metricas_prueba"]
            if nombre == "prueba_historica":
                dif = abs(m_r["auc"] - esperado_r["auc"]) + abs(m_r["recall"] - esperado_r["recall"])
                rep.agregar("riesgo: reproducibilidad", dif <= tol, f"AUC {m_r['auc']:.4f} vs {esperado_r['auc']:.4f}",
                            valor=dif, umbral=tol)
            rep.agregar("riesgo: AUC", m_r["auc"] >= r["auc_min"], f"{m_r['auc']:.3f}", valor=m_r["auc"], umbral=r["auc_min"])
            rep.agregar("riesgo: recall (umbral 0.17)", m_r["recall"] >= r["recall_min"],
                        f"{m_r['recall']:.3f}; {m_r['falsos_negativos']} retrasos no anticipados",
                        valor=m_r["recall"], umbral=r["recall_min"])
            rep.agregar("riesgo: precisión", m_r["precision"] >= r["precision_min"], f"{m_r['precision']:.3f}",
                        valor=m_r["precision"], umbral=r["precision_min"])

            # 4. Equidad por transportista (A9)
            q = g["equidad"]
            p = riesgo.predict_proba(df[entradas])[:, 1]
            eta, _, _ = predecir_eta(modelos, df[entradas], desfase)
            seg = pd.DataFrame({"seg": df[q["variable"]], "y": df["retraso"], "alerta": (p >= umbral).astype(int),
                                "err": np.abs(eta - df["dias_reales"].to_numpy())})
            por_seg = seg.groupby("seg").apply(lambda s: pd.Series({
                "ordenes": len(s), "recall_riesgo": recall_score(s["y"], s["alerta"], zero_division=0),
                "mae_eta": s["err"].mean()}), include_groups=False)
            resultados[nombre]["equidad"] = por_seg.round(4).reset_index().to_dict(orient="records")
            peor = por_seg["recall_riesgo"].idxmin()
            rep.agregar(f"equidad: recall mínimo por {q['variable']}", por_seg["recall_riesgo"].min() >= q["recall_segmento_min"],
                        f"{peor} = {por_seg['recall_riesgo'].min():.3f} (A9: TR-A 0.698)",
                        valor=por_seg["recall_riesgo"].min(), umbral=q["recall_segmento_min"])
            brecha = float(por_seg["mae_eta"].max() - por_seg["mae_eta"].min())
            rep.agregar(f"equidad: brecha de MAE entre segmentos de {q['variable']}", brecha <= q["brecha_mae_segmento_max"],
                        f"{por_seg['mae_eta'].idxmax()} {por_seg['mae_eta'].max():.2f} vs "
                        f"{por_seg['mae_eta'].idxmin()} {por_seg['mae_eta'].min():.2f} días",
                        valor=brecha, umbral=q["brecha_mae_segmento_max"])

    rep.extra = {"version_modelos": meta["version_modelos"], "version_datos": meta["version_datos"],
                 "git_commit_entrenamiento": meta.get("git_commit"), "resultados": resultados}
    d = rep.guardar(a.salida, "evaluacion_modelos")
    sys.exit(0 if d["resultado"] == "aprobado" else 1)


if __name__ == "__main__":
    main()
