"""Router de intención con ONNX Runtime (optimización de la Actividad 9, integrada en la Fase 3).

Mismo modelo model-v1.0.0, exportado a ONNX con la tabla de embeddings en INT8 (variante onnx_emb8).
El archivo se genera al construir la imagen (docker/exportar_router_onnx.py) y solo se acepta si sus
acciones son idénticas a las del modelo de PyTorch en los 560 mensajes de prueba. No importa PyTorch:
la imagen final es más ligera, arranca más rápido y usa menos memoria.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import numpy as np

from .politica import CLASES, UMBRAL_POR_DEFECTO, Decision, decidir, softmax

log = logging.getLogger("cab.router")


class RouterIntencionOnnx:
    cargado = True
    runtime = "onnxruntime"

    def __init__(self, carpeta: Path, umbral: float | None = None, hilos: int = 2):
        import onnxruntime as ort
        from tokenizers import Tokenizer

        inicio = time.perf_counter()
        self.cfg = json.loads((carpeta / "config_cab.json").read_text(encoding="utf-8"))
        if list(self.cfg.get("clases", CLASES)) != CLASES:
            raise ValueError(f"Las clases del modelo {self.cfg.get('clases')} no coinciden con la política {CLASES}")
        ruta_ver = carpeta / "verificacion_onnx.json"
        self.verificacion = json.loads(ruta_ver.read_text(encoding="utf-8")) if ruta_ver.exists() else {}
        opciones = ort.SessionOptions()
        opciones.intra_op_num_threads, opciones.inter_op_num_threads = max(1, hilos), 1
        opciones.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.sesion = ort.InferenceSession(str(carpeta / "router.onnx"), opciones, providers=["CPUExecutionProvider"])
        self.max_len = int(self.cfg.get("max_len", 64))
        self.tokenizer = Tokenizer.from_file(str(carpeta / "tokenizer.json"))
        self.tokenizer.enable_truncation(max_length=self.max_len)
        self.temperatura = float(self.cfg.get("temperatura", 1.0))
        self.umbral = float(umbral if umbral is not None else self.cfg.get("umbral_confianza", UMBRAL_POR_DEFECTO))
        self.segundos_carga = round(time.perf_counter() - inicio, 2)
        log.info("Router ONNX cargado en %.1f s (umbral %.2f, temperatura %.3f)", self.segundos_carga, self.umbral,
                 self.temperatura)

    def info(self) -> dict:
        return {
            "cargado": True,
            "runtime": self.runtime,
            "variante": "onnx_emb8 (embeddings INT8)",
            "modelo_base": self.cfg.get("modelo_base"),
            "version": self.cfg.get("version_modelo"),
            "version_datos": self.cfg.get("version_datos"),
            "git_commit_entrenamiento": self.cfg.get("git_commit"),
            "umbral_confianza": self.umbral,
            "temperatura": self.temperatura,
            "segundos_carga": self.segundos_carga,
            "sha256_onnx": self.verificacion.get("sha256_onnx"),
            "equivalencia_pytorch": (f"{self.verificacion['acciones_iguales']}/{self.verificacion['mensajes_verificados']}"
                                     if self.verificacion else None),
        }

    def logits(self, texto: str) -> np.ndarray:
        e = self.tokenizer.encode(texto)
        return self.sesion.run(["logits"], {"input_ids": np.array([e.ids], dtype=np.int64),
                                            "attention_mask": np.array([e.attention_mask], dtype=np.int64)})[0][0]

    def predecir(self, texto: str) -> Decision:
        return decidir(softmax(self.logits(texto), self.temperatura), self.umbral)
