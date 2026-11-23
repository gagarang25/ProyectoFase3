"""Resume el escaneo de Grype (JSON) para el resumen de GitHub Actions: severidad, tipo de paquete, cuántas
tienen parche y los paquetes con más hallazgos. Sirve para decidir con datos qué atender primero."""

from __future__ import annotations

import json
import sys
from collections import Counter

ORDEN = ["Critical", "High", "Medium", "Low", "Negligible", "Unknown"]


def main(ruta: str) -> None:
    d = json.load(open(ruta, encoding="utf-8"))
    m = d.get("matches", [])
    sev = Counter(x["vulnerability"].get("severity", "Unknown") for x in m)
    tipo = Counter(x["artifact"].get("type", "?") for x in m)
    parche = sum(bool(x["vulnerability"].get("fix", {}).get("versions")) for x in m)
    paquetes = Counter(f"{x['artifact']['name']} ({x['artifact'].get('type', '?')})" for x in m
                       if x["vulnerability"].get("severity") in ("Critical", "High"))
    print(f"### 🛡️ Vulnerabilidades conocidas en la imagen: {len(m)} ({parche} con parche disponible)\n")
    print("| Severidad | " + " | ".join(ORDEN) + " |")
    print("|---|" + "---|" * len(ORDEN))
    print("| Hallazgos | " + " | ".join(str(sev.get(s, 0)) for s in ORDEN) + " |\n")
    print("Por tipo de paquete: " + ", ".join(f"{k} {v}" for k, v in tipo.most_common()) + "\n")
    if paquetes:
        print("| Paquete con más hallazgos críticos o altos | Hallazgos |\n|---|---|")
        for p, n in paquetes.most_common(8):
            print(f"| {p} | {n} |")
    print()


if __name__ == "__main__":
    main(sys.argv[1])
