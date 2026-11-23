# Pipeline MLOps/GitOps · Proyecto Fase 3

Código de las compuertas automáticas que ejecutan los workflows de `.github/workflows/`. Todos los umbrales están
en `config/pipeline.yaml`; cada uno cita de qué entrega anterior sale (Fase 1, Fase 2, A6, A8 o A9).

| Archivo | Paso del pipeline | Qué decide |
|---|---|---|
| `validar_datos.py` | CI · pruebas de datos; CT · lote nuevo | ¿Los datos son los versionados (SHA-256)? ¿Cumplen el contrato? ¿Hay fuga entre entrenamiento y prueba? ¿Hay drift? |
| `evaluar_modelos.py` | CI · pruebas de modelos | ¿Los `.joblib` son los registrados? ¿Las métricas se reproducen? ¿Cumplen MAE, cobertura, AUC, recall y equidad? |
| `auditoria.py` | CI · último paso; release | Registro de auditoría por ejecución y tarjeta del modelo; decisión *desplegable* o *bloqueado* |
| `ct/ingestar_lote.py` | CT · paso 1 | Lote de entregas con etiqueta hasta la fecha de corte (versión `data-v1.x.0`) |
| `ct/reentrenar.py` | CT · pasos 2–5; monitoreo | Disparadores, challenger (runbook RB-02 de la A8), compuertas de promoción, MLflow |
| `config/contrato_*.yaml` | contratos de datos | Columnas, tipos, rangos, categorías y reglas de coherencia |
| `registro/modelos.json` | registro de modelos en Git | Historial de versiones candidatas y promovidas |
| `tests/test_pipeline.py` | CI · pytest | Las compuertas rechazan datos corruptos; el pipeline calcula el ETA igual que la API |

Niveles de cada verificación: **error** detiene el pipeline; **aviso** se reporta en el resumen de GitHub Actions y en
la auditoría, pero no detiene (por ejemplo, drift de entradas o un SLO que el reentrenamiento mitiga sin cumplir).

## Ejecutar localmente

```bash
pip install -r backend/requirements.txt -r monitoreo/requirements.txt
python pipeline/validar_datos.py
python pipeline/evaluar_modelos.py
python -m pytest pipeline/tests

# Entrenamiento continuo con el escenario de drift de la A8 (planta +15 puntos desde el 09/11/2026)
python pipeline/ct/ingestar_lote.py --hasta 2026-11-30
python pipeline/ct/reentrenar.py --lote datos/entregas/lotes/entregas_produccion_drift_planta_hasta_2026-11-30.csv --corte 2026-11-30
```

Los reportes quedan en `pipeline/reportes/` (JSON para la auditoría y Markdown para el resumen de la ejecución);
en GitHub Actions se publican como artefactos.
