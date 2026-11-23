"""Utilidades comunes del pipeline (Fase 3): configuración, hashes, Git y reportes de compuertas.

Cada paso del pipeline produce un reporte con la misma forma:
    {"paso", "fecha_utc", "git_commit", "resultado": "aprobado" | "rechazado", "verificaciones": [...]}
Cada verificación tiene nivel "error" (detiene el pipeline) o "aviso" (se reporta, no detiene).
El reporte se guarda en JSON (auditoría) y en Markdown (resumen visible en GitHub Actions).
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import yaml

RAIZ = Path(__file__).resolve().parents[1]
CONFIG = RAIZ / "pipeline" / "config"


def cargar_yaml(nombre: str) -> dict:
    return yaml.safe_load((CONFIG / nombre).read_text(encoding="utf-8"))


def sha256(ruta: Path) -> str:
    h = hashlib.sha256()
    with open(ruta, "rb") as f:
        for bloque in iter(lambda: f.read(1 << 20), b""):
            h.update(bloque)
    return h.hexdigest()


def git(*args: str) -> str:
    try:
        return subprocess.run(["git", *args], cwd=RAIZ, capture_output=True, text=True, check=True).stdout.strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        return ""


def info_git() -> dict:
    """Commit evaluado. En GitHub Actions se usan las variables del runner (fuente confiable)."""
    return {
        "commit": os.getenv("GITHUB_SHA") or git("rev-parse", "HEAD") or "sin_git",
        "rama": os.getenv("GITHUB_REF_NAME") or git("rev-parse", "--abbrev-ref", "HEAD"),
        "arbol_limpio": git("status", "--porcelain") == "",
        "ejecucion": (f"{os.getenv('GITHUB_SERVER_URL')}/{os.getenv('GITHUB_REPOSITORY')}/actions/runs/"
                      f"{os.getenv('GITHUB_RUN_ID')}" if os.getenv("GITHUB_RUN_ID") else "local"),
        "actor": os.getenv("GITHUB_ACTOR", "local"),
    }


@dataclass
class Verificacion:
    nombre: str
    aprobada: bool
    detalle: str
    nivel: str = "error"          # "error" detiene el pipeline; "aviso" solo se reporta
    valor: float | str | None = None
    umbral: float | str | None = None


@dataclass
class Reporte:
    paso: str
    verificaciones: list[Verificacion] = field(default_factory=list)
    extra: dict = field(default_factory=dict)

    def agregar(self, nombre: str, aprobada: bool, detalle: str, nivel: str = "error", valor=None, umbral=None):
        self.verificaciones.append(Verificacion(nombre, bool(aprobada), detalle, nivel,
                                                _num(valor), _num(umbral)))
        return aprobada

    @property
    def aprobado(self) -> bool:
        return all(v.aprobada for v in self.verificaciones if v.nivel == "error")

    def como_dict(self) -> dict:
        return {"paso": self.paso, "fecha_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "git": info_git(), "resultado": "aprobado" if self.aprobado else "rechazado",
                "errores": sum(not v.aprobada and v.nivel == "error" for v in self.verificaciones),
                "avisos": sum(not v.aprobada and v.nivel == "aviso" for v in self.verificaciones),
                "verificaciones": [asdict(v) for v in self.verificaciones], **self.extra}

    def markdown(self) -> str:
        d = self.como_dict()
        icono = "✅" if self.aprobado else "❌"
        lineas = [f"### {icono} {self.paso}: {d['resultado']} ({d['errores']} errores, {d['avisos']} avisos)", "",
                  "| Verificación | Resultado | Valor | Umbral | Detalle |", "|---|---|---|---|---|"]
        for v in self.verificaciones:
            r = "✅" if v.aprobada else ("⚠️ aviso" if v.nivel == "aviso" else "❌ error")
            lineas.append(f"| {v.nombre} | {r} | {_fmt(v.valor)} | {_fmt(v.umbral)} | {v.detalle} |")
        return "\n".join(lineas) + "\n"

    def guardar(self, carpeta: Path, nombre: str) -> dict:
        carpeta.mkdir(parents=True, exist_ok=True)
        d = self.como_dict()
        (carpeta / f"{nombre}.json").write_text(json.dumps(d, ensure_ascii=False, indent=2), encoding="utf-8")
        (carpeta / f"{nombre}.md").write_text(self.markdown(), encoding="utf-8")
        resumen = os.getenv("GITHUB_STEP_SUMMARY")
        if resumen:                                  # se ve en la página de la ejecución de GitHub Actions
            with open(resumen, "a", encoding="utf-8") as f:
                f.write(self.markdown() + "\n")
        print(self.markdown())
        return d


def _num(v):
    if v is None or isinstance(v, str):
        return v
    try:
        return round(float(v), 6)
    except (TypeError, ValueError):
        return str(v)


def _fmt(v) -> str:
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.4g}"
    return str(v)
