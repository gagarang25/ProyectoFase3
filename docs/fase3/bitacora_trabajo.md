# Bitácora de trabajo · Proyecto Fase 3

Registro cronológico de decisiones, problemas y resultados. Las ejecuciones en GitHub Actions se enlazan desde la
pestaña *Actions* del repositorio; los números de esta bitácora salen de esas ejecuciones o de las pruebas locales
que se indican.

| Fecha | Actividad | Decisión o resultado | Evidencia |
|---|---|---|---|
| 23/11/2026 | Análisis de la rúbrica y de los temas 23 y 24 | Se integran la A8 (monitoreo, runbooks) y la A9 (ONNX, autoescalado, equidad) en lugar de rehacerlas | — |
| 23/11/2026 | Repositorio | Repositorio nuevo `ProyectoFase3` con el historial y las etiquetas de la Fase 2 (trazabilidad de `code-v1.0.0` a `opt-v1.0.0`) | `git log --oneline --decorate` |
| 23/11/2026 | Optimización integrada (A9) | El router pasa a ONNX Runtime + embeddings INT8 dentro del `docker build`. Prueba local: exportación 22 s, cuantización 13 s, **560/560 acciones iguales** a PyTorch y el mismo SHA-256 del ONNX que en la A9 (`1d9fd839…`): la exportación es reproducible | `docker/exportar_router_onnx.py`, `verificacion_onnx.json` |
| 23/11/2026 | API app-v2.0.0 | 72 pruebas pytest con el router ONNX real (61 + 11 del modelo); aceptación local 36/36, router 160/160, p95 91 ms, 1,215 MB de memoria (Fase 2: p95 184 ms, 1.7–1.9 GB) | `pruebas/pruebas_funcionales.py --estricto` |
| 23/11/2026 | Pruebas de datos | **Hallazgo:** 2 de los 160 mensajes de la prueba externa están idénticos en el entrenamiento (ids 31 y 43). Se registra como aviso (impacto ≤ 1.25 %) y como limitación en la tarjeta del modelo | `pipeline/validar_datos.py` |
| 23/11/2026 | Pruebas del pipeline | **Defecto encontrado por las propias pruebas:** una fecha con formato inválido hacía fallar el validador con una excepción en lugar de reportarla. Se corrigió (fechas ISO 8601 con `errors="coerce"`) | `pipeline/tests/test_pipeline.py` |
| 23/11/2026 | Training-serving skew (deuda de la A5) | Prueba que compara el ETA del pipeline con el de la API en 150 órdenes: idénticos con model-v1.0.0 y con el modelo reentrenado | `test_sin_training_serving_skew` |
| 23/11/2026 | Entrenamiento continuo (prueba local) | Corte 23/11: hay drift (PSI 0.43 en días comprometidos, 100 % de órdenes con carga fuera de rango) pero el challenger **no mejora** (1.378 → 1.378): no se promueve. Corte 30/11: mejora 1.521 → 1.374 días y cobertura 88.7 % → se propone `model-v1.1.0`; el SLO de 1.30 sigue sin cumplirse (aviso, escalamiento al dueño del producto como en INC-01) | `pipeline/ct/reentrenar.py` |
| 23/11/2026 | Herramientas | Se valida la sintaxis de los 6 workflows con actionlint 1.7.12. kustomize y kubeconform se ejecutan en GitHub Actions (job `manifiestos-gitops`) | — |
| 23/11/2026 | Primer push a GitHub | El push inicial a un repositorio vacío no dispara el CI (no hay contra qué comparar los archivos de `paths-ignore`); se lanzó manualmente (`workflow_dispatch`) | CI #1 |
| 23/11/2026 | CI #1 (commit `1f32a60`) | **Todo en verde a la primera, 16 min 35 s**: 82 pruebas, datos (0 errores, 1 aviso), modelos (0/0), imagen de **896 MB** (Fase 2: 4,370), compuerta ONNX 560/560, aceptación 36/36 con p95 de 61 ms, auditoría *desplegable*, bot de GitOps (`2a15467`) y Argo CD sincronizó staging en 10 s; self-heal en 6 s | Actions → CI #1 |
| 23/11/2026 | Escaneo de seguridad (Grype) | 254 vulnerabilidades conocidas en la imagen (0 críticas, 84 altas) y 49 con parche → acción de la iteración 2 | Auditoría CI #1 |
| 23/11/2026 | Promoción a producción | `promover.yml` abrió el PR #1 (`sha-1f32a60`); el CI del PR pasó. Al ser un PR del bot, GitHub pide aprobar la ejecución (control de seguridad) | PR #1 |
| 23/11/2026 | Monitoreo + CT (corte 23/11) | Alerta por drift (PSI 0.43, 100 % de órdenes con carga fuera de rango) → CT automático → challenger **rechazado** (1.378 → 1.378). Idéntico a la prueba local: el pipeline es reproducible | Monitoreo #1, CT #1 |
| 23/11/2026 | Monitoreo + CT (corte 30/11, semana simulada) | Alerta (MAE 1.46, PSI 0.38, 86 %) → challenger aprobado: MAE 1.521 → 1.374, cobertura 88.7 %; PR #2 `model-v1.1.0` para aprobación humana | Monitoreo #2, CT #2, PR #2 |
| 23/11/2026 | Iteración 2 | (1) el disparador activo ahora se muestra como aviso (antes salía en verde aunque hubiera alerta); (2) parches del sistema operativo y pip en la imagen; (3) resumen de vulnerabilidades por severidad, tipo y paquete en el CI | commit de la iteración 2 |
