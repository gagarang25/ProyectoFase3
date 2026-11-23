"""Registro de auditoría y tarjeta del modelo (último paso del pipeline de CI).

Junta en un solo registro inmutable, por ejecución, QUÉ se probó, CON QUÉ versiones, QUIÉN lo disparó y QUÉ se
decidió: commit, actor, enlace a la ejecución, versiones de datos y modelos con SHA-256, resultado de cada
compuerta, imagen publicada (etiqueta, digest y tamaño), inventario de componentes (SBOM) y vulnerabilidades.
Responde a las preguntas de una auditoría (Tema 22, A9): ¿qué modelo atendió a los clientes tal día?, ¿con qué
datos se entrenó?, ¿quién aprobó el cambio?, ¿pasó todas las pruebas?

También genera la tarjeta del modelo (model card; Mitchell et al., 2019) a partir de los metadatos, para que la
documentación nunca quede desactualizada respecto al modelo desplegado.

Uso:  python pipeline/auditoria.py --reportes <carpeta> [<carpeta> ...] [--imagen ghcr.io/...:tag --digest sha256:...]
Salida: pipeline/reportes/auditoria/registro_auditoria.json, MODEL_CARD.md y resumen en GitHub Actions.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from comun import RAIZ, cargar_yaml, info_git, sha256  # noqa: E402

SALIDA = RAIZ / "pipeline" / "reportes" / "auditoria"


def buscar(carpetas: list[Path], patron: str) -> list[Path]:
    encontrados = []
    for c in carpetas:
        if c.exists():
            encontrados += sorted(c.rglob(patron))
    return encontrados


def vulnerabilidades(carpetas: list[Path]) -> dict | None:
    """Resumen del escaneo de Grype (anchore/scan-action) si existe."""
    for ruta in buscar(carpetas, "*.json"):
        try:
            d = json.loads(ruta.read_text(encoding="utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            continue
        if isinstance(d, dict) and "matches" in d and "descriptor" in d:
            conteo: dict[str, int] = {}
            for m in d["matches"]:
                sev = m.get("vulnerability", {}).get("severity", "Unknown")
                conteo[sev] = conteo.get(sev, 0) + 1
            return {"herramienta": "grype", "total": len(d["matches"]), "por_severidad": conteo,
                    "con_parche": sum(bool(m.get("vulnerability", {}).get("fix", {}).get("versions")) for m in d["matches"])}
    return None


def tarjeta_modelo(meta: dict, cfg: dict, compuertas: dict, salud: dict | None) -> str:
    eta, riesgo = meta["eta"], meta["riesgo_retraso"]
    m_eta, m_r = eta["metricas_prueba"], riesgo["metricas_prueba"]
    router = (salud or {}).get("router", {})
    reciente = (meta.get("evaluaciones") or {}).get("validacion_reciente")
    lineas = [
        f"# Tarjeta del modelo · chatbot de CAB ({cfg['versiones']['app']})",
        "",
        f"_Generada automáticamente por el pipeline el {datetime.now(timezone.utc):%d/%m/%Y %H:%M} UTC "
        f"(commit {info_git()['commit'][:7]}). No editar a mano._",
        "",
        "## Uso previsto",
        "Responder a clientes B2B de Componentes Automotrices del Bajío (empresa ficticia) el estatus y la fecha "
        "estimada de entrega de sus órdenes de compra. Los datos son simulados. No toma decisiones sobre personas; "
        "toda queja, solicitud de un ejecutivo o mensaje con baja confianza se escala a una persona.",
        "",
        "## Componentes",
        "| Componente | Versión | Algoritmo | Datos |",
        "|---|---|---|---|",
        f"| Router de intención | {router.get('version', cfg['versiones']['router'])} | mDeBERTa-v3 ajustado; servido con "
        f"ONNX Runtime + embeddings INT8 (equivalencia con PyTorch {router.get('equivalencia_pytorch', '560/560')}) | "
        f"{router.get('version_datos', 'data-v3.0.0')} |",
        f"| ETA (fecha estimada) | {meta['version_modelos']} | {eta['algoritmo']} | {meta['version_datos']} |",
        f"| Riesgo de retraso | {meta['version_modelos']} | {riesgo['algoritmo']}, umbral {riesgo['umbral']} | "
        f"{meta['version_datos']} |",
        "",
        "## Desempeño (conjunto de prueba temporal, pedidos desde el 01/05/2026)",
        f"- ETA: MAE {m_eta['mae_dias']:.2f} días; {m_eta['cobertura_rango']:.1%} de entregas dentro del rango; "
        f"rango ensanchado {eta.get('desfase_rango_dias', 0)} día(s).",
        f"- Riesgo: AUC {m_r['auc']:.3f}, recall {m_r['recall']:.3f}, precisión {m_r['precision']:.3f}.",
    ]
    if reciente:
        lineas.append(f"- Validación reciente ({reciente['desde']} a {reciente['hasta']}): MAE "
                      f"{reciente['metricas_eta']['mae_dias']:.2f} días (modelo anterior "
                      f"{reciente['mae_modelo_anterior']:.2f}); cobertura {reciente['metricas_eta']['cobertura_rango']:.1%}.")
    lineas += [
        "",
        "## Compuertas de esta versión",
        "| Paso | Resultado | Errores | Avisos |",
        "|---|---|---|---|",
        *[f"| {k} | {v['resultado']} | {v['errores']} | {v['avisos']} |" for k, v in compuertas.items()],
        "",
        "## Limitaciones conocidas",
        "- Datos simulados (no hay SAP real); los resultados no se trasladan sin validar con datos de CAB.",
        "- El recall del riesgo es menor para TR-A y la modalidad Dedicado (A9); el router es menos preciso en "
        "español que en inglés en respuestas automáticas (0.884 vs 0.941).",
        "- 2 de los 160 mensajes de la prueba externa también están en el entrenamiento (detectado por el pipeline).",
        "- Ante drift fuerte (planta saturada) el reentrenamiento mitiga pero no recupera el SLO de MAE: se escala "
        "al dueño del producto.",
        "",
        "## Trazabilidad",
        f"- Commit de entrenamiento: `{meta.get('git_commit')}`",
        f"- SHA-256 de los archivos del modelo: {', '.join(f'`{k}` {v[:12]}…' for k, v in meta['archivos'].items())}",
    ]
    return "\n".join(lineas) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reportes", type=Path, nargs="+", default=[RAIZ / "pipeline" / "reportes"])
    ap.add_argument("--imagen", default=os.getenv("IMAGEN_REF", ""))
    ap.add_argument("--digest", default=os.getenv("IMAGEN_DIGEST", ""))
    ap.add_argument("--imagen-mb", type=float, default=float(os.getenv("IMAGEN_MB", "0") or 0))
    ap.add_argument("--salida", type=Path, default=SALIDA)
    a = ap.parse_args()
    cfg = cargar_yaml("pipeline.yaml")
    meta = json.loads((RAIZ / "backend" / "modelos" / "metadatos_modelos.json").read_text(encoding="utf-8"))

    compuertas, salud, aceptacion = {}, None, None
    for ruta in buscar(a.reportes, "*.json"):
        try:
            d = json.loads(ruta.read_text(encoding="utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            continue
        if isinstance(d, dict) and "paso" in d and "resultado" in d:
            compuertas[d["paso"]] = {"resultado": d["resultado"], "errores": d["errores"], "avisos": d["avisos"],
                                     "archivo": ruta.name}
        elif ruta.name == "salud.json":
            salud = d
        elif ruta.name == "resumen.json" and "casos_pasan" in d:
            aceptacion = d
    if aceptacion:
        ok = aceptacion["casos_pasan"] == aceptacion["casos_total"]
        compuertas["Pruebas de aceptación (contenedor)"] = {
            "resultado": "aprobado" if ok else "rechazado", "errores": aceptacion["casos_total"] - aceptacion["casos_pasan"],
            "avisos": 0, "detalle": f"{aceptacion['casos_pasan']}/{aceptacion['casos_total']} casos; p95 "
                                    f"{aceptacion['latencia']['p95_ms']:.0f} ms; router "
                                    f"{aceptacion['coincidencia_colab'][0]}/{aceptacion['coincidencia_colab'][1]}"}

    registro = {
        "tipo": "registro_auditoria_pipeline",
        "fecha_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git": info_git(),
        "evento": os.getenv("GITHUB_EVENT_NAME", "local"),
        "configuracion": {"version": cfg["version_config"], "sha256": sha256(RAIZ / "pipeline" / "config" / "pipeline.yaml")},
        "versiones": {
            "app": cfg["versiones"]["app"], "router": (salud or {}).get("router", {}).get("version", cfg["versiones"]["router"]),
            "router_runtime": (salud or {}).get("router", {}).get("runtime"),
            "router_sha256_onnx": (salud or {}).get("router", {}).get("sha256_onnx"),
            "modelos_tabulares": meta["version_modelos"], "datos": meta["version_datos"],
            "sha256_modelos": meta["archivos"], "commit_entrenamiento": meta.get("git_commit"),
        },
        "imagen": {"referencia": a.imagen, "digest": a.digest, "mb_descomprimida": a.imagen_mb or None,
                   "limite_mb": cfg["servicio"]["imagen_mb_max"]},
        "sbom": [str(p.name) for p in buscar(a.reportes, "*sbom*")],
        "vulnerabilidades": vulnerabilidades(a.reportes),
        "compuertas": compuertas,
        "decision": "desplegable" if compuertas and all(v["resultado"] == "aprobado" for v in compuertas.values())
                    else "bloqueado",
    }
    a.salida.mkdir(parents=True, exist_ok=True)
    (a.salida / "registro_auditoria.json").write_text(json.dumps(registro, ensure_ascii=False, indent=2), encoding="utf-8")
    (a.salida / "MODEL_CARD.md").write_text(tarjeta_modelo(meta, cfg, compuertas, salud), encoding="utf-8")

    md = [f"### 🧾 Registro de auditoría: **{registro['decision']}**", "",
          f"- Commit `{registro['git']['commit'][:7]}` · actor `{registro['git']['actor']}` · evento `{registro['evento']}`",
          f"- Imagen `{a.imagen or '—'}` {('· ' + a.digest[:19] + '…') if a.digest else ''} "
          f"{f'· {a.imagen_mb:,.0f} MB' if a.imagen_mb else ''}",
          f"- Modelos {meta['version_modelos']} · datos {meta['version_datos']} · router "
          f"{registro['versiones']['router']} ({registro['versiones']['router_runtime'] or 'n/d'})",
          f"- Vulnerabilidades: {registro['vulnerabilidades'] or 'sin escaneo'}", "",
          "| Compuerta | Resultado | Errores | Avisos |", "|---|---|---|---|",
          *[f"| {k} | {v['resultado']} | {v['errores']} | {v['avisos']} |" for k, v in compuertas.items()]]
    texto = "\n".join(md) + "\n"
    print(texto)
    if os.getenv("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a", encoding="utf-8") as f:
            f.write(texto)
    sys.exit(0 if registro["decision"] == "desplegable" else 1)


if __name__ == "__main__":
    main()
