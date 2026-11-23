"""Entrenamiento continuo · pasos 2 a 5: disparadores, challenger, compuertas, registro y promoción.

  2. Disparadores (pipeline.yaml → ct.disparadores). Con las entregas de las últimas 2 semanas se mide el MAE del
     modelo en producción (champion) y el drift de las entradas. Si nada se dispara y no se pide --forzar, termina
     sin reentrenar (también es una decisión auditable).
  3. Challenger: misma receta de la Fase 2 y mismo procedimiento del runbook RB-02 de la A8
     (obs.respuestas.reentrenar_challenger): pesos ×10 a las órdenes recientes y rango calibrado hasta cubrir 87 %.
     La última semana entregada NO se usa para entrenar: es la validación.
  4. Compuertas de promoción (champion/challenger): el challenger debe bajar el MAE reciente al menos 0.05 días,
     cubrir ≥ 85 % con su rango y no romper el SLO en el periodo histórico. El riesgo de retraso no se reentrena.
  5. Registro: MLflow (experimento, métricas, modelo registrado con alias @challenger y, si aprueba, @candidato),
     respaldo portable de MLflow y, si aprueba, los archivos del modelo nuevo en backend/modelos/ con
     metadatos completos (versión, datos, commit, SHA-256). La promoción a producción NO es automática: el
     workflow sube una rama y el dueño del modelo la aprueba con un pull request (GitOps).

Uso:  python pipeline/ct/reentrenar.py --lote datos/entregas/lotes/<archivo>.csv --corte 2026-11-30
              [--version-nueva model-v1.1.0] [--forzar] [--sin-mlflow]
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from importlib.metadata import version as version_lib
from pathlib import Path

import joblib
import pandas as pd

RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ / "pipeline"))
sys.path.insert(0, str(RAIZ / "monitoreo" / "src"))
from comun import Reporte, cargar_yaml, info_git, sha256  # noqa: E402
from evaluar_modelos import metricas_eta, metricas_riesgo  # noqa: E402
from obs import drift  # noqa: E402  (A8)
from obs import modelos_eta as me  # noqa: E402  (A8: misma receta de la Fase 2)
from obs.respuestas import reentrenar_challenger  # noqa: E402  (A8: runbook RB-02)

DIR_MOD = RAIZ / "backend" / "modelos"
SALIDA = RAIZ / "pipeline" / "reportes"
REGISTRO_GIT = RAIZ / "pipeline" / "registro" / "modelos.json"


def a_api(modelos: dict) -> dict:
    """Formato de obs.modelos_eta → formato de evaluar_modelos (mismo que la API)."""
    return {"eta": modelos["mediana"], "inf": modelos["inferior"], "sup": modelos["superior"]}


def disparadores(campeon: dict, lote: pd.DataFrame, historico: pd.DataFrame, corte: pd.Timestamp, cfg: dict,
                 meta: dict) -> tuple[dict, list[str]]:
    d = cfg["ct"]["disparadores"]
    reciente = lote[lote["fecha_entrega_real"] > corte - pd.Timedelta(days=14)]
    pedidos = lote[lote["fecha_pedido"] > corte - pd.Timedelta(days=14)]
    m = metricas_eta(a_api(campeon), reciente, meta["entradas"], campeon.get("desfase", 0))
    referencia = historico[historico["fecha_pedido"] < cfg["modelos"]["corte_prueba"]]
    psi = {c: drift.psi(referencia[c], pedidos[c]) for c in ("distancia_km", "peso_kg", "cantidad_piezas",
                                                               "holgura_dias", "dias_comprometidos")}
    psi |= {c: drift.psi_categorico(referencia[c], pedidos[c]) for c in ("transportista", "region_destino", "planta",
                                                                          "familia_producto", "modalidad")}
    fuera = drift.fuera_de_rango(pedidos, meta["rangos_numericos_entrenamiento"], ["utilizacion_planta"])
    medido = {"mae_champion_2_semanas": m["mae_dias"], "cobertura_champion_2_semanas": m["cobertura_rango"],
              "entregas_2_semanas": m["ordenes"], "psi_max": max(psi.values()), "psi_variable": max(psi, key=psi.get),
              "fuera_de_rango_utilizacion": fuera}
    motivos = []
    if m["mae_dias"] > d["mae_reciente_max"]:
        motivos.append(f"MAE reciente {m['mae_dias']:.2f} > {d['mae_reciente_max']} (alerta A5 de la A8)")
    if medido["psi_max"] > d["psi_max"]:
        motivos.append(f"PSI {medido['psi_variable']} = {medido['psi_max']:.2f} > {d['psi_max']}")
    if fuera > d["fuera_de_rango_max"]:
        motivos.append(f"{fuera:.0%} de las órdenes con carga de planta fuera del rango de entrenamiento")
    return medido, motivos


def registrar_mlflow(campeon, reto, resultado, rep, etiquetas, metricas) -> dict:
    import mlflow

    from obs.registro import RegistroETA

    carpeta = RAIZ / "pipeline" / "mlflow"
    carpeta.mkdir(parents=True, exist_ok=True)
    os.chdir(carpeta)                                    # artefactos con rutas relativas (respaldo portable, A6)
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))
    mlflow.set_experiment("CAB_Fase3_entrenamiento_continuo")
    registro = RegistroETA("cab_eta_entrega")
    with mlflow.start_run(run_name=f"ct_{etiquetas['version_candidata']}_{etiquetas['corte']}") as run:
        mlflow.set_tags({"tipo_run": "entrenamiento_continuo", **etiquetas})
        mlflow.log_params({"peso_reciente": resultado["peso_reciente"], "cobertura_meta": resultado["cobertura_meta"],
                           "desfase_rango_dias": reto["desfase"], "ordenes_entrenamiento": resultado["ordenes_entrenamiento"],
                           "ordenes_validacion": resultado["ordenes_validacion"]})
        mlflow.log_metrics({k: float(v) for k, v in metricas.items()})
        mlflow.log_dict(rep.como_dict(), "compuertas_ct.json")
        if registro.alias("champion") is None:            # primera ejecución: se registra el modelo actual
            v_camp = registro.registrar(campeon, "Modelo en producción antes del reentrenamiento",
                                        {**etiquetas, "version_semantica": etiquetas["version_champion"]},
                                        nombre_run=f"registro_{etiquetas['version_champion']}")
            registro.asignar("champion", v_camp)
        v_reto = registro.registrar(reto, f"Challenger del entrenamiento continuo (corte {etiquetas['corte']})",
                                    {**etiquetas, "version_semantica": etiquetas["version_candidata"],
                                     "decision": "candidato" if rep.aprobado else "rechazado"},
                                    metricas=metricas, nombre_run=f"registro_{etiquetas['version_candidata']}")
        registro.asignar("challenger", v_reto)
        if rep.aprobado:
            registro.asignar("candidato", v_reto)        # @champion se mueve hasta que se aprueba el PR (GitOps)
        info = {"run_id": run.info.run_id, "experimento": "CAB_Fase3_entrenamiento_continuo",
                "modelo_registrado": "cab_eta_entrega", "version_registro": v_reto}
    os.chdir(RAIZ)
    respaldo = shutil.make_archive(str(SALIDA / "mlflow_respaldo_ct"), "zip", carpeta)
    info["respaldo"] = str(Path(respaldo).relative_to(RAIZ))
    return info


def guardar_modelo(reto: dict, meta: dict, version: str, lote_rel: str, lote_manifiesto: dict, corte, limite,
                   metricas_hist: dict, resultado: dict, motivos: list[str]) -> None:
    joblib.dump(reto["mediana"], DIR_MOD / "eta_mediana.joblib")
    joblib.dump(reto["inferior"], DIR_MOD / "eta_rango_inferior.joblib")
    joblib.dump(reto["superior"], DIR_MOD / "eta_rango_superior.joblib")
    anterior = meta["version_modelos"]
    nuevo = json.loads(json.dumps(meta))
    nuevo.update({
        "version_modelos": version,
        "version_datos": f"{meta['version_datos'].split(' + ')[0]} + {lote_manifiesto['version']} (lote de producción)",
        "sha256_datos_lote": lote_manifiesto["sha256"],
        "git_commit": info_git()["commit"],
        "fecha_entrenamiento_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "modelo_anterior": anterior,
    })
    nuevo["eta"]["algoritmo"] = meta["eta"]["algoritmo"] + " · reentrenado (CT, runbook RB-02)"
    nuevo["eta"]["desfase_rango_dias"] = int(reto["desfase"])
    nuevo["eta"]["metricas_prueba"] = metricas_hist
    nuevo["eta"]["reentrenamiento"] = {"corte": str(corte.date()), "peso_reciente": resultado["peso_reciente"],
                                       "ordenes_entrenamiento": resultado["ordenes_entrenamiento"],
                                       "ordenes_produccion_usadas": resultado["ordenes_produccion_usadas"],
                                       "motivos": motivos}
    validacion = {k: v for k, v in resultado["challenger"].items()}
    nuevo["evaluaciones"] = {
        "prueba_historica": {"archivo": "datos/entregas/entregas_cab.csv", "columna": "fecha_pedido",
                             "desde": "2026-05-01", "principal_riesgo": True,
                             "nota": "periodo incluido en el reentrenamiento: verifica que el modelo no 'olvide'"},
        "validacion_reciente": {"archivo": lote_rel, "columna": "fecha_entrega_real",
                                "desde": str((limite + pd.Timedelta(days=1)).date()), "hasta": str(corte.date()),
                                "principal_eta": True, "principal_riesgo": False, "nivel": "aviso",
                                "metricas_eta": {"mae_dias": validacion["mae"], "cobertura_rango": validacion["cobertura"]},
                                "mae_modelo_anterior": resultado["champion"]["mae"],
                                "nota": "última semana entregada: ningún modelo la vio al entrenar"},
    }
    for p in sorted(DIR_MOD.glob("*.joblib")):
        nuevo["archivos"][p.name] = sha256(p)
    for lib in ("scikit-learn", "pandas", "numpy", "joblib"):
        nuevo["versiones"][lib] = version_lib(lib)
    (DIR_MOD / "metadatos_modelos.json").write_text(json.dumps(nuevo, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lote", type=Path, required=True)
    ap.add_argument("--corte", required=True)
    ap.add_argument("--version-nueva", default=None, help="por defecto sube el número menor (model-v1.0.0 → v1.1.0)")
    ap.add_argument("--forzar", action="store_true", help="reentrena aunque no haya disparadores")
    ap.add_argument("--sin-mlflow", action="store_true")
    ap.add_argument("--solo-diagnostico", action="store_true",
                    help="monitoreo: solo mide los disparadores y escribe alerta=true/false (no entrena)")
    a = ap.parse_args()
    cfg = cargar_yaml("pipeline.yaml")
    ct = cfg["ct"]
    corte = pd.Timestamp(a.corte)
    meta = json.loads((DIR_MOD / "metadatos_modelos.json").read_text(encoding="utf-8"))
    mayor, menor, _ = meta["version_modelos"].removeprefix("model-v").split(".")
    version = a.version_nueva or f"model-v{mayor}.{int(menor) + 1}.0"
    lote_ruta = a.lote if a.lote.is_absolute() else RAIZ / a.lote
    lote_rel = str(lote_ruta.relative_to(RAIZ))
    manifiesto = json.loads((lote_ruta.parent / "versiones.json").read_text(encoding="utf-8"))["archivos"][lote_ruta.name]
    lote = pd.read_csv(lote_ruta, parse_dates=["fecha_pedido", "fecha_entrega_real"])
    historico = pd.read_csv(RAIZ / "datos" / "entregas" / "entregas_cab.csv", parse_dates=["fecha_pedido", "fecha_entrega_real"])
    campeon = me.cargar_v1()
    campeon["desfase"] = int(meta["eta"].get("desfase_rango_dias", 0))

    rep = Reporte(f"Entrenamiento continuo: {meta['version_modelos']} (champion) vs {version} (challenger)")
    medido, motivos = disparadores(campeon, lote, historico, corte, cfg, meta)
    rep.extra = {"corte": str(corte.date()), "lote": lote_rel, "version_lote": manifiesto["version"],
                 "disparadores": {"medido": medido, "motivos": motivos}}
    # Un disparador activo es un AVISO (se ve en amarillo): no detiene el pipeline, inicia el reentrenamiento.
    rep.agregar("disparadores", not motivos, "; ".join(motivos) if motivos else "ninguno (modelo estable)", nivel="aviso",
                valor=len(motivos), umbral=0)
    if a.solo_diagnostico:
        rep.paso = f"Monitoreo del modelo en producción ({meta['version_modelos']}, corte {corte.date()})"
        rep.extra["decision"] = "reentrenar" if motivos else "estable"
        rep.guardar(SALIDA, "monitoreo_modelo")
        _salida_gha(promovido=False, version=version, alerta=bool(motivos))
        return
    if not motivos and not a.forzar:
        rep.extra["decision"] = "sin_reentrenamiento"
        rep.guardar(SALIDA, "ct_decision")
        _salida_gha(promovido=False, version=version)
        return

    resultado = reentrenar_challenger(historico, lote, corte, campeon, ct["peso_reciente"], ct["cobertura_meta"])
    resultado |= {"peso_reciente": ct["peso_reciente"], "cobertura_meta": ct["cobertura_meta"]}
    reto = resultado["modelos"]
    limite = corte - pd.Timedelta(days=7)
    c_val, k_val = resultado["challenger"], resultado["champion"]
    p = ct["promocion"]
    mejora = k_val["mae"] - c_val["mae"]
    rep.agregar("validación reciente: mejora del MAE", mejora >= p["mejora_mae_min"],
                f"{k_val['mae']:.3f} → {c_val['mae']:.3f} días ({resultado['ordenes_validacion']} entregas de la última semana)",
                valor=mejora, umbral=p["mejora_mae_min"])
    rep.agregar("validación reciente: cobertura del rango", c_val["cobertura"] >= p["cobertura_reciente_min"],
                f"{c_val['cobertura']:.1%} (champion {k_val['cobertura']:.1%}); rango ensanchado {reto['desfase']} día(s)",
                valor=c_val["cobertura"], umbral=p["cobertura_reciente_min"])
    rep.agregar("validación reciente: SLO de MAE", c_val["mae"] <= cfg["modelos"]["eta"]["mae_max"],
                f"{c_val['mae']:.3f} días; si no se cumple, el reentrenamiento mitiga pero no resuelve: "
                "se escala al dueño del producto (A8, INC-01)", nivel="aviso", valor=c_val["mae"],
                umbral=cfg["modelos"]["eta"]["mae_max"])
    hist = historico[historico["fecha_pedido"] >= pd.Timestamp(cfg["modelos"]["corte_prueba"])]
    m_hist = metricas_eta(a_api(reto), hist, meta["entradas"], reto["desfase"])
    rep.agregar("periodo histórico: sin olvido", m_hist["mae_dias"] <= p["mae_prueba_historica_max"],
                f"MAE {m_hist['mae_dias']:.3f} (model-v1.0.0: {meta['eta']['metricas_prueba']['mae_dias']:.3f})",
                valor=m_hist["mae_dias"], umbral=p["mae_prueba_historica_max"])
    valida = lote[lote["fecha_entrega_real"] > limite]
    riesgo = joblib.load(DIR_MOD / "riesgo_retraso.joblib")
    m_r = metricas_riesgo(riesgo, valida, meta["entradas"], float(meta["riesgo_retraso"]["umbral"]))
    rep.agregar("riesgo de retraso en la semana reciente (no se reentrena)", m_r["recall"] >= cfg["modelos"]["riesgo"]["recall_min"],
                f"recall {m_r['recall']:.3f}, AUC {m_r['auc']:.3f}", nivel="aviso", valor=m_r["recall"],
                umbral=cfg["modelos"]["riesgo"]["recall_min"])

    metricas = {"mae_challenger_reciente": c_val["mae"], "mae_champion_reciente": k_val["mae"],
                "cobertura_challenger_reciente": c_val["cobertura"], "cobertura_champion_reciente": k_val["cobertura"],
                "mae_challenger_historico": m_hist["mae_dias"], "mejora_mae": mejora,
                "recall_riesgo_reciente": m_r["recall"], **{k: v for k, v in medido.items() if isinstance(v, (int, float))}}
    etiquetas = {"version_champion": meta["version_modelos"], "version_candidata": version, "corte": str(corte.date()),
                 "version_datos": manifiesto["version"], "sha256_lote": manifiesto["sha256"],
                 "git_commit": info_git()["commit"], "ejecucion": info_git()["ejecucion"]}
    rep.extra |= {"decision": "promover (requiere PR)" if rep.aprobado else "rechazado", "metricas": metricas,
                  "champion_reciente": k_val, "challenger_reciente": c_val, "challenger_historico": m_hist}
    if not a.sin_mlflow:
        rep.extra["mlflow"] = registrar_mlflow(campeon, reto, resultado, rep, etiquetas, metricas)
    if rep.aprobado:
        guardar_modelo(reto, meta, version, lote_rel, manifiesto, corte, limite, m_hist, resultado, motivos)
        REGISTRO_GIT.parent.mkdir(parents=True, exist_ok=True)
        historial = json.loads(REGISTRO_GIT.read_text(encoding="utf-8")) if REGISTRO_GIT.exists() else []
        historial.append({"version": version, "reemplaza_a": meta["version_modelos"], "estado": "candidato (PR pendiente)",
                          "corte": str(corte.date()), "datos": manifiesto["version"], "motivos": motivos,
                          "mae_reciente": round(c_val["mae"], 4), "mae_anterior": round(k_val["mae"], 4),
                          "desfase_rango_dias": int(reto["desfase"]), **etiquetas,
                          "fecha_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")})
        REGISTRO_GIT.write_text(json.dumps(historial, ensure_ascii=False, indent=2), encoding="utf-8")
    rep.guardar(SALIDA, "ct_decision")
    _salida_gha(promovido=rep.aprobado, version=version)


def _salida_gha(promovido: bool, version: str, alerta: bool | None = None) -> None:
    """Variables de salida para el workflow (GITHUB_OUTPUT)."""
    print(f"PROMOVIDO={str(promovido).lower()} VERSION={version} ALERTA={alerta}")
    if os.getenv("GITHUB_OUTPUT"):
        with open(os.environ["GITHUB_OUTPUT"], "a", encoding="utf-8") as f:
            f.write(f"promovido={str(promovido).lower()}\nversion={version}\n")
            if alerta is not None:
                f.write(f"alerta={str(alerta).lower()}\n")


if __name__ == "__main__":
    main()
