"""Exporta el router de intención (model-v1.0.0) a ONNX y lo cuantiza, DENTRO de la construcción de la imagen.

Es la optimización elegida en la Actividad 9 (variante onnx_emb8): mismo modelo, ejecutado con ONNX Runtime y
con la tabla de embeddings en INT8. Se hace en la etapa "exportador" del Dockerfile, así la imagen final no
necesita PyTorch y el archivo ONNX siempre se genera desde la versión etiquetada en Hugging Face Hub.

Compuerta de equivalencia (falla la construcción si no se cumple): las acciones del router ONNX deben ser
idénticas a las del modelo original de PyTorch en los 560 mensajes de prueba (400 de plantilla + 160 escritos
a mano). La Actividad 9 midió 560/560; si una nueva versión de una librería cambia el resultado, la imagen
no se publica.

Uso:  python exportar_router_onnx.py --modelo /modelo_hf --salida /modelo_onnx --datos /datos_prueba
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

CLASES = ["estatus_orden", "fecha_entrega", "queja", "contactar_ejecutivo", "otro"]


def sha256(ruta: Path) -> str:
    h = hashlib.sha256()
    with open(ruta, "rb") as f:
        for bloque in iter(lambda: f.read(1 << 20), b""):
            h.update(bloque)
    return h.hexdigest()


def exportar(modelo: Path, destino: Path, max_len: int) -> None:
    """Paso 1 (proceso aparte): PyTorch → ONNX fp32."""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(modelo)
    red = AutoModelForSequenceClassification.from_pretrained(modelo, dtype=torch.float32).eval()
    ejemplo = tok(["¿cuándo llega mi orden 4500123456?", "where is my order"], padding=True, truncation=True,
                  max_length=max_len, return_tensors="pt")
    entradas = ("input_ids", "attention_mask")
    torch.onnx.export(red, tuple(ejemplo[k] for k in entradas), destino, input_names=list(entradas),
                      output_names=["logits"], opset_version=17, dynamo=False,
                      dynamic_axes={k: {0: "lote", 1: "tokens"} for k in entradas} | {"logits": {0: "lote"}})


def cuantizar(origen: Path, destino: Path) -> None:
    """Paso 2 (proceso aparte, sin PyTorch en memoria): solo la tabla de embeddings (Gather) en INT8."""
    from onnxruntime.quantization import QuantType, quantize_dynamic

    quantize_dynamic(origen, destino, weight_type=QuantType.QInt8, op_types_to_quantize=["Gather"])


def logits_pytorch(modelo: Path, textos: list[str], max_len: int):
    import numpy as np
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    torch.set_num_threads(2)
    tok = AutoTokenizer.from_pretrained(modelo)
    red = AutoModelForSequenceClassification.from_pretrained(modelo, dtype=torch.float32).eval()
    salida = []
    with torch.no_grad():
        for t in textos:
            e = tok(t, truncation=True, max_length=max_len, return_tensors="pt")
            salida.append(red(**e).logits[0].float().numpy())
    return np.array(salida)


def logits_onnx(ruta: Path, tokenizer: Path, textos: list[str], max_len: int):
    import numpy as np
    import onnxruntime as ort
    from tokenizers import Tokenizer

    op = ort.SessionOptions()
    op.intra_op_num_threads, op.inter_op_num_threads = 2, 1
    sesion = ort.InferenceSession(str(ruta), op, providers=["CPUExecutionProvider"])
    tok = Tokenizer.from_file(str(tokenizer))
    tok.enable_truncation(max_length=max_len)
    salida = []
    for t in textos:
        e = tok.encode(t)
        salida.append(sesion.run(["logits"], {"input_ids": np.array([e.ids], dtype=np.int64),
                                              "attention_mask": np.array([e.attention_mask], dtype=np.int64)})[0][0])
    return np.array(salida)


def acciones(logits, temperatura: float, umbral: float) -> list[tuple[str, str]]:
    """Misma política que backend/app/politica.py (copiada para no depender del código de la API aquí)."""
    import numpy as np

    salida = []
    for lg in logits:
        z = lg / temperatura
        p = np.exp(z - z.max())
        p /= p.sum()
        orden = np.argsort(p)[::-1]
        clase, conf, conf2 = CLASES[orden[0]], float(p[orden[0]]), float(p[orden[1]])
        if conf < umbral:
            dos = {CLASES[orden[0]], CLASES[orden[1]]}
            accion = "aclarar" if dos == {"estatus_orden", "fecha_entrega"} and conf + conf2 >= umbral else "escalar"
        elif clase in ("estatus_orden", "fecha_entrega"):
            accion = "automatica"
        elif clase in ("queja", "contactar_ejecutivo"):
            accion = "escalar"
        else:
            accion = "canalizar"
        salida.append((clase, accion))
    return salida


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--modelo", type=Path, required=True, help="carpeta del modelo de Hugging Face (safetensors)")
    ap.add_argument("--salida", type=Path, required=True, help="carpeta del router ONNX para la imagen final")
    ap.add_argument("--datos", type=Path, required=True, help="carpeta con prueba.csv y prueba_externa_v3.csv")
    ap.add_argument("--paso", choices=["exportar", "cuantizar"], help=argparse.SUPPRESS)
    a = ap.parse_args()

    cfg = json.loads((a.modelo / "config_cab.json").read_text(encoding="utf-8"))
    max_len = int(cfg.get("max_len", 64))
    trabajo = a.salida / "_trabajo"
    fp32, emb8 = trabajo / "router_fp32.onnx", a.salida / "router.onnx"

    if a.paso == "exportar":
        exportar(a.modelo, fp32, max_len)
        return
    if a.paso == "cuantizar":
        cuantizar(fp32, emb8)
        return

    a.salida.mkdir(parents=True, exist_ok=True)
    trabajo.mkdir(exist_ok=True)
    tiempos = {}
    for paso in ("exportar", "cuantizar"):  # cada paso pesado en su propio proceso (memoria, lección de la A9)
        inicio = time.perf_counter()
        subprocess.run([sys.executable, __file__, "--modelo", str(a.modelo), "--salida", str(a.salida),
                        "--datos", str(a.datos), "--paso", paso], check=True)
        tiempos[paso] = round(time.perf_counter() - inicio, 1)
        print(f"{paso}: {tiempos[paso]} s", flush=True)

    # Compuerta de equivalencia: mismas acciones que PyTorch en los 560 mensajes de prueba
    import pandas as pd

    textos = pd.concat([pd.read_csv(a.datos / "prueba.csv")["mensaje"],
                        pd.read_csv(a.datos / "prueba_externa_v3.csv")["mensaje"]]).astype(str).tolist()
    t, u = float(cfg["temperatura"]), float(cfg["umbral_confianza"])
    ref = acciones(logits_pytorch(a.modelo, textos, max_len), t, u)
    nue = acciones(logits_onnx(emb8, a.modelo / "tokenizer.json", textos, max_len), t, u)
    iguales_accion = sum(r[1] == n[1] for r, n in zip(ref, nue))
    iguales_clase = sum(r[0] == n[0] for r, n in zip(ref, nue))
    verificacion = {
        "variante": "onnx_emb8 (Actividad 9): ONNX Runtime + embeddings INT8",
        "modelo_origen": cfg.get("version_modelo"),
        "sha256_safetensors": sha256(a.modelo / "model.safetensors"),
        "sha256_onnx": sha256(emb8),
        "mb_onnx": round(emb8.stat().st_size / 1e6, 1),
        "mensajes_verificados": len(textos),
        "acciones_iguales": iguales_accion,
        "intenciones_iguales": iguales_clase,
        "segundos": tiempos,
        "compuerta": "acciones_iguales == mensajes_verificados",
    }
    print(json.dumps(verificacion, indent=2, ensure_ascii=False))
    if iguales_accion != len(textos):
        sys.exit(f"Compuerta de equivalencia ONNX NO superada: {iguales_accion}/{len(textos)} acciones iguales")

    for nombre in ("config.json", "config_cab.json", "tokenizer.json", "tokenizer_config.json"):
        if (a.modelo / nombre).exists():
            shutil.copy2(a.modelo / nombre, a.salida / nombre)
    (a.salida / "verificacion_onnx.json").write_text(json.dumps(verificacion, indent=2, ensure_ascii=False),
                                                    encoding="utf-8")
    shutil.rmtree(trabajo)


if __name__ == "__main__":
    main()
