# syntax=docker/dockerfile:1
# =============================================================================
# Imagen de la API del chatbot de CAB · app-v2.0.0 (Proyecto Fase 3)
#
# Construcción en dos etapas (multi-stage):
#   1. exportador  Descarga el router model-v1.0.0 de Hugging Face Hub, lo exporta a ONNX, cuantiza la tabla
#                  de embeddings a INT8 (optimización elegida en la Actividad 9) y verifica que sus acciones
#                  sean idénticas a las de PyTorch en 560 mensajes. Si la compuerta falla, la imagen no se crea.
#   2. final       Solo lo necesario para servir: ONNX Runtime, scikit-learn y la API. Sin PyTorch.
#
# La imagen con PyTorch de la Fase 2 se conserva en Dockerfile.pytorch.
#
# Construir:  docker build -t cab-chatbot-api:2.0.0 --build-arg GIT_COMMIT=$(git rev-parse HEAD) .
# Ejecutar:   docker run --rm -p 8000:8000 cab-chatbot-api:2.0.0      (chat en http://localhost:8000)
# =============================================================================

# ---------------------------------------------------------------- 1. exportador
FROM python:3.12-slim AS exportador
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1 HF_HUB_DISABLE_TELEMETRY=1 TOKENIZERS_PARALLELISM=false
WORKDIR /exportar
RUN pip install torch==2.15.0 --index-url https://download.pytorch.org/whl/cpu
COPY backend/requirements.txt backend/requirements-pytorch.txt ./
RUN pip install -r requirements.txt -r requirements-pytorch.txt

# Versión del router: cambiarla no requiere tocar código, solo estos argumentos (los fija el pipeline).
ARG MODELO_ROUTER=gagarang/router-intencion-cab-mdeberta
ARG MODELO_REVISION=model-v1.0.0
RUN python -c "from huggingface_hub import snapshot_download; \
snapshot_download(repo_id='${MODELO_ROUTER}', revision='${MODELO_REVISION}', local_dir='/modelo_hf', \
allow_patterns=['*.json', '*.safetensors', '*.model', '*.txt'])"

COPY docker/exportar_router_onnx.py ./
COPY datos/mensajes_cab/prueba.csv datos/mensajes_cab/prueba_externa_v3.csv ./datos/
RUN python exportar_router_onnx.py --modelo /modelo_hf --salida /modelo_onnx --datos ./datos

# ---------------------------------------------------------------- 2. final
FROM python:3.12-slim AS final

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    TOKENIZERS_PARALLELISM=false

WORKDIR /app
COPY backend/requirements.txt .
RUN pip install -r requirements.txt

# Router ONNX ya verificado en la etapa anterior (incluye verificacion_onnx.json para la auditoría).
COPY --from=exportador /modelo_onnx ./modelo_router
COPY backend/app ./app
COPY backend/modelos ./modelos
COPY backend/tests ./tests
COPY datos/entregas/piloto_ab_cab.csv ./datos/piloto_ab_cab.csv
COPY frontend/public ./frontend

# Trazabilidad: commit, versión y fecha de construcción quedan dentro de la imagen y se ven en /api/salud y /metrics.
ARG GIT_COMMIT=desconocido
ARG VERSION_APP=app-v2.0.0
ENV GIT_COMMIT=${GIT_COMMIT} \
    VERSION_APP=${VERSION_APP} \
    ROUTER_RUNTIME=onnx \
    MODELO_ROUTER_RUTA=/app/modelo_router \
    MODELOS_ENTREGA_RUTA=/app/modelos \
    DATOS_SAP_RUTA=/app/datos/piloto_ab_cab.csv \
    FRONTEND_RUTA=/app/frontend \
    PORT=8000
LABEL org.opencontainers.image.title="cab-chatbot-api" \
      org.opencontainers.image.source="https://github.com/gagarang25/ProyectoFase3" \
      org.opencontainers.image.description="Chatbot de estatus de órdenes y ETA de CAB (prototipo académico, datos simulados)" \
      org.opencontainers.image.revision="${GIT_COMMIT}" \
      org.opencontainers.image.version="${VERSION_APP}"

# Usuario sin privilegios (uid 1000): requisito de Kubernetes (runAsNonRoot) y de Hugging Face Spaces.
RUN useradd --create-home --uid 1000 cab && chown -R cab:cab /app
USER 1000

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
  CMD python -c "import os, urllib.request; urllib.request.urlopen(f'http://localhost:{os.environ.get(\"PORT\", 8000)}/api/salud', timeout=4)"

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT}"]
