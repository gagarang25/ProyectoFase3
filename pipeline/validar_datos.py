"""Pruebas automatizadas de DATOS (compuerta 1 del pipeline).

Qué se verifica
  1. Integridad y versionado: el SHA-256 de cada archivo coincide con su manifiesto (versiones.json).
     Un cambio en un CSV sin actualizar el manifiesto (y su versión) detiene el pipeline.
  2. Contrato de entregas: columnas, tipos, nulos, categorías permitidas, rangos, llave única y reglas de
     coherencia entre columnas (por ejemplo, retraso = 1 solo si dias_retraso > 0).
  3. Contrato de mensajes del router: clases e idiomas válidos, largo, balance, español representado y
     fuga de información entre entrenamiento y prueba (mensajes repetidos).
  4. Drift (solo con --lote, en el entrenamiento continuo): PSI de las variables de entrada contra los datos
     de entrenamiento y proporción de órdenes fuera del rango visto al entrenar (reutiliza obs.drift de la A8).

Uso:
    python pipeline/validar_datos.py                                   # CI: datos versionados del repositorio
    python pipeline/validar_datos.py --lote datos/entregas/lotes/x.csv # CT: lote nuevo + drift
Salida:
    pipeline/reportes/validacion_datos.{json,md}  (y resumen en la página de GitHub Actions)
    Código de salida 1 si alguna verificación de nivel "error" falla.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from comun import RAIZ, Reporte, cargar_yaml, sha256  # noqa: E402

sys.path.insert(0, str(RAIZ / "monitoreo" / "src"))
from obs import drift  # noqa: E402  (Actividad 8)

SALIDA = RAIZ / "pipeline" / "reportes"
NUM_DRIFT = ["distancia_km", "peso_kg", "cantidad_piezas", "holgura_dias", "dias_comprometidos"]   # A8
CAT_DRIFT = ["transportista", "region_destino", "planta", "familia_producto", "modalidad"]          # A8


# ----------------------------------------------------------------------------- 1. integridad
def verificar_manifiestos(rep: Reporte, manifiestos: list[str]) -> None:
    for rel in manifiestos:
        ruta = RAIZ / rel
        m = json.loads(ruta.read_text(encoding="utf-8"))
        for nombre, info in m["archivos"].items():
            archivo = ruta.parent / nombre
            if not archivo.exists():
                rep.agregar(f"sha256 {nombre}", False, f"falta el archivo declarado en {rel}")
                continue
            ok = sha256(archivo) == info["sha256"]
            rep.agregar(f"sha256 {nombre}", ok,
                        f"{m['version']}: coincide con el manifiesto" if ok else
                        f"{m['version']}: el archivo cambió sin actualizar {rel} (versión de datos nueva)")
            if "filas" in info:
                n = sum(1 for _ in open(archivo, encoding="utf-8")) - 1
                rep.agregar(f"filas {nombre}", n == info["filas"], f"{n:,} filas", valor=n, umbral=info["filas"])


# ----------------------------------------------------------------------------- 2. entregas
def verificar_entregas(rep: Reporte, df: pd.DataFrame, nombre: str, contrato: dict) -> None:
    cols = contrato["columnas"]
    faltan = [c for c in cols if c not in df.columns]
    rep.agregar(f"{nombre}: columnas", not faltan, "todas presentes" if not faltan else f"faltan {faltan}")
    if faltan:
        return
    llave = contrato["llave"]
    dup = int(df[llave].duplicated().sum())
    rep.agregar(f"{nombre}: llave única", dup == 0, f"{dup} {llave} repetidos", valor=dup, umbral=0)
    malas = int((~df[llave].astype(str).str.match(contrato["patron_llave"])).sum())
    rep.agregar(f"{nombre}: formato de {llave}", malas == 0, f"{malas} con formato inválido", valor=malas, umbral=0)
    for f in contrato["fechas"]:
        invalidas = int(pd.to_datetime(df[f], format="%Y-%m-%d", errors="coerce").isna().sum())  # ISO 8601
        rep.agregar(f"{nombre}: fecha {f}", invalidas == 0, f"{invalidas} fechas inválidas", valor=invalidas, umbral=0)

    problemas = []
    for c, regla in cols.items():
        nulos = float(df[c].isna().mean())
        if nulos > regla.get("nulos_max", 0):
            problemas.append(f"{c}: {nulos:.1%} nulos")
        v = df[c].dropna()
        if regla["tipo"] == "categoria":
            fuera = sorted(set(v.astype(str)) - set(regla["valores"]))
            if fuera:
                problemas.append(f"{c}: categorías no permitidas {fuera[:5]}")
        elif regla["tipo"] in ("entero", "numero"):
            num = pd.to_numeric(v, errors="coerce")
            if num.isna().any():
                problemas.append(f"{c}: {int(num.isna().sum())} valores no numéricos")
            if regla["tipo"] == "entero" and not np.all(np.equal(np.mod(num.dropna(), 1), 0)):
                problemas.append(f"{c}: valores no enteros")
            if (num < regla["min"]).any() or (num > regla["max"]).any():
                problemas.append(f"{c}: fuera de [{regla['min']}, {regla['max']}] "
                                 f"(mín {num.min():g}, máx {num.max():g})")
        elif "patron" in regla:
            malos = int((~v.astype(str).str.match(regla["patron"])).sum())
            if malos:
                problemas.append(f"{c}: {malos} valores con formato inválido")
    rep.agregar(f"{nombre}: tipos, nulos, rangos y categorías", not problemas,
                f"{len(cols)} columnas conformes" if not problemas else "; ".join(problemas[:6]),
                valor=len(problemas), umbral=0)

    fp = pd.to_datetime(df["fecha_pedido"], format="%Y-%m-%d", errors="coerce")
    fe = pd.to_datetime(df["fecha_entrega_real"], format="%Y-%m-%d", errors="coerce")
    reglas = {
        "retraso_coherente": (df["retraso"] == (df["dias_retraso"] > 0).astype(int)).mean(),
        "dias_retraso_coherente": (df["dias_retraso"] == (df["dias_reales"] - df["dias_comprometidos"]).clip(lower=0)).mean(),
        "entrega_despues_del_pedido": (fe > fp).mean(),
    }
    for r, prop in reglas.items():
        rep.agregar(f"{nombre}: regla {r}", prop == 1.0, f"{prop:.2%} de las filas la cumplen", valor=prop, umbral=1.0)
    tasa = float(df["retraso"].mean())
    rep.agregar(f"{nombre}: tasa de retraso plausible", 0.10 <= tasa <= 0.40, f"{tasa:.1%} (A6: 22.8 %)",
                valor=tasa, umbral="0.10–0.40")


# ----------------------------------------------------------------------------- 3. mensajes
def verificar_mensajes(rep: Reporte, contrato: dict) -> None:
    carpeta = RAIZ / "datos" / "mensajes_cab"
    def normalizar(s):
        return s.astype(str).str.lower().str.strip()

    conjuntos = {n: pd.read_csv(carpeta / n) for n in contrato["divisiones"]}
    for n, df in conjuntos.items():
        clases_malas = sorted(set(df["intencion"]) - set(contrato["clases"]))
        idiomas_malos = sorted(set(df["idioma"]) - set(contrato["idiomas"]))
        largo = df["mensaje"].astype(str).str.len()
        vacios = int((largo < contrato["largo_min"]).sum() + (largo > contrato["largo_max"]).sum())
        ok = not clases_malas and not idiomas_malos and vacios == 0 and df["mensaje"].notna().all()
        rep.agregar(f"mensajes {n}: contrato", ok,
                    f"{len(df):,} mensajes, {df['intencion'].nunique()} clases" if ok else
                    f"clases {clases_malas}, idiomas {idiomas_malos}, {vacios} fuera de largo")
    entrena = conjuntos["entrenamiento.csv"]
    props = entrena["intencion"].value_counts(normalize=True)
    minima = float(props.min())
    rep.agregar("mensajes: balance de clases", minima >= contrato["proporcion_minima_clase"],
                f"clase menos representada {props.idxmin()} = {minima:.1%}", valor=minima,
                umbral=contrato["proporcion_minima_clase"])
    es = float((entrena["idioma"] == "es").mean())
    rep.agregar("mensajes: español representado", es >= contrato["proporcion_minima_espanol"],
                f"{es:.0%} en español (retroalimentación A4/A5)", valor=es, umbral=contrato["proporcion_minima_espanol"])
    vistos = set(normalizar(entrena["mensaje"]))
    for n in ("validacion.csv", "prueba.csv"):
        fuga = int(normalizar(conjuntos[n]["mensaje"]).isin(vistos).sum())
        rep.agregar(f"mensajes: fuga entrenamiento → {n}", fuga <= contrato["fuga_permitida"],
                    f"{fuga} mensajes idénticos", valor=fuga, umbral=contrato["fuga_permitida"])
    externa = pd.read_csv(carpeta / contrato["externa"])
    repetidos = externa[normalizar(externa["mensaje"]).isin(vistos)]
    rep.agregar("mensajes: fuga entrenamiento → prueba externa", len(repetidos) == 0,
                f"{len(repetidos)} de {len(externa)} mensajes escritos a mano también están en entrenamiento "
                f"(ids {repetidos['id'].tolist()}); se reporta la accuracy con y sin ellos", nivel="aviso",
                valor=len(repetidos), umbral=0)


# ----------------------------------------------------------------------------- 4. drift
def verificar_drift(rep: Reporte, lote: pd.DataFrame, referencia: pd.DataFrame, cfg: dict, rangos: dict) -> dict:
    filas = []
    for c in NUM_DRIFT:
        filas.append({"variable": c, "tipo": "numérica", "psi": drift.psi(referencia[c], lote[c])})
    for c in CAT_DRIFT:
        filas.append({"variable": c, "tipo": "categórica", "psi": drift.psi_categorico(referencia[c], lote[c])})
    tabla = pd.DataFrame(filas)
    tabla["nivel"] = tabla["psi"].map(drift.nivel_psi)
    peor = tabla.sort_values("psi").iloc[-1]
    rep.agregar("drift: PSI máximo de las entradas", peor["psi"] <= cfg["psi_alerta"],
                f"{peor['variable']} = {peor['psi']:.3f} ({peor['nivel']})", nivel="aviso",
                valor=peor["psi"], umbral=cfg["psi_alerta"])
    fuera = drift.fuera_de_rango(lote, rangos, ["utilizacion_planta"])
    rep.agregar("drift: órdenes con carga de planta fuera del rango de entrenamiento", fuera <= cfg["fuera_de_rango_max"],
                f"{fuera:.1%} de las órdenes (máximo visto al entrenar {rangos['utilizacion_planta'][1]:.2f})",
                nivel="aviso", valor=fuera, umbral=cfg["fuera_de_rango_max"])
    return {"psi": tabla.round(4).to_dict(orient="records"), "fuera_de_rango_utilizacion": round(fuera, 4)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lote", type=Path, help="lote nuevo de entregas (entrenamiento continuo)")
    ap.add_argument("--salida", type=Path, default=SALIDA)
    a = ap.parse_args()

    cfg = cargar_yaml("pipeline.yaml")
    contrato_e, contrato_m = cargar_yaml("contrato_entregas.yaml"), cargar_yaml("contrato_mensajes.yaml")
    rep = Reporte("Pruebas de datos")
    if a.lote is None:
        manifiestos = list(cfg["datos"]["manifiestos"])
        archivos = list(contrato_e["archivos"])
        lotes = RAIZ / "datos" / "entregas" / "lotes" / "versiones.json"
        if lotes.exists():                      # lotes que agregó el entrenamiento continuo (CT)
            manifiestos.append(str(lotes.relative_to(RAIZ)))
            archivos += [f"datos/entregas/lotes/{n}" for n in json.loads(lotes.read_text(encoding="utf-8"))["archivos"]]
        verificar_manifiestos(rep, manifiestos)
        for rel in archivos:
            verificar_entregas(rep, pd.read_csv(RAIZ / rel), Path(rel).name, contrato_e)
        verificar_mensajes(rep, contrato_m)
        nombre = "validacion_datos"
    else:
        lote = pd.read_csv(a.lote)
        verificar_entregas(rep, lote, a.lote.name, contrato_e)
        historico = pd.read_csv(RAIZ / "datos" / "entregas" / "entregas_cab.csv")
        traslape = int(lote["id_orden"].isin(historico["id_orden"]).sum())
        rep.agregar("lote: órdenes nuevas (sin traslape con el histórico)", traslape == 0, f"{traslape} repetidas",
                    valor=traslape, umbral=0)
        referencia = historico[historico["fecha_pedido"] < cfg["modelos"]["corte_prueba"]]
        meta = json.loads((RAIZ / "backend" / "modelos" / "metadatos_modelos.json").read_text(encoding="utf-8"))
        rep.extra["drift"] = verificar_drift(rep, lote, referencia, cfg["datos"],
                                             meta["rangos_numericos_entrenamiento"])
        nombre = "validacion_lote"
    d = rep.guardar(a.salida, nombre)
    sys.exit(0 if d["resultado"] == "aprobado" else 1)


if __name__ == "__main__":
    main()
