"""Métricas de operación en formato Prometheus (Fase 3).

La Actividad 8 definió los SLI y las alertas con una telemetría simulada; aquí la API los expone de verdad en
GET /metrics para que un recolector (Prometheus, Grafana Agent, Azure Monitor, CloudWatch) los lea cada 15 s.
Dos planos, igual que en la A8:

  Servicio  cab_solicitudes_total, cab_latencia_segundos        → disponibilidad y latencia (SLO 99.5 %, p95 < 400 ms)
  Modelo    cab_decisiones_router_total, cab_confianza_router,  → cobertura automática, escalamiento, drift de
            cab_eta_dias, cab_prob_retraso                         las salidas (PSI de la A8 sobre estas distribuciones)

No se registra el texto de los mensajes ni el número de orden (LFPDPPP): solo conteos y distribuciones.
"""

from __future__ import annotations

from prometheus_client import CONTENT_TYPE_LATEST, CollectorRegistry, Counter, Gauge, Histogram, generate_latest

REGISTRO = CollectorRegistry()

SOLICITUDES = Counter("cab_solicitudes_total", "Solicitudes HTTP atendidas", ["ruta", "metodo", "codigo"],
                      registry=REGISTRO)
LATENCIA = Histogram("cab_latencia_segundos", "Latencia de la API por ruta", ["ruta"], registry=REGISTRO,
                     buckets=(0.025, 0.05, 0.1, 0.2, 0.4, 0.8, 1.6, 3.2))
DECISIONES = Counter("cab_decisiones_router_total", "Decisiones del router de intención",
                     ["intencion", "accion", "idioma"], registry=REGISTRO)
CONFIANZA = Histogram("cab_confianza_router", "Confianza de la intención elegida", registry=REGISTRO,
                      buckets=(0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0))
ETA_DIAS = Histogram("cab_eta_dias", "Días estimados de entrega (salida del modelo de ETA)", registry=REGISTRO,
                     buckets=(2, 4, 6, 8, 10, 12, 15, 20, 30))
PROB_RETRASO = Histogram("cab_prob_retraso", "Probabilidad de retraso (salida del clasificador de riesgo)",
                         registry=REGISTRO, buckets=(0.05, 0.1, 0.17, 0.25, 0.4, 0.6, 0.8, 1.0))
INFO = Gauge("cab_info", "Versiones desplegadas (valor fijo 1)",
             ["version_app", "git_commit", "router_version", "router_runtime", "modelos_entrega"], registry=REGISTRO)


def publicar_info(version_app: str, git_commit: str, router: dict, entrega: dict) -> None:
    INFO.clear()
    INFO.labels(version_app, git_commit[:12], str(router.get("version")), str(router.get("runtime", "ninguno")),
                str(entrega.get("version_modelos", entrega.get("version")))).set(1)


def observar_solicitud(ruta: str, metodo: str, codigo: int, segundos: float) -> None:
    if ruta == "/metrics":
        return
    SOLICITUDES.labels(ruta, metodo, str(codigo)).inc()
    LATENCIA.labels(ruta).observe(segundos)


def observar_decision(intencion: str, accion: str, idioma: str, confianza: float) -> None:
    DECISIONES.labels(intencion, accion, idioma).inc()
    CONFIANZA.observe(confianza)


def observar_estimacion(eta_dias: float, prob_retraso: float) -> None:
    ETA_DIAS.observe(eta_dias)
    PROB_RETRASO.observe(prob_retraso)


def exportar() -> tuple[bytes, str]:
    return generate_latest(REGISTRO), CONTENT_TYPE_LATEST
