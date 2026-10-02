FROM python:3.12-slim@sha256:57cd7c3a7a273101a6485ba99423ee568157882804b1124b4dd04266317710de

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    VIRTUAL_ENV=/app/.venv \
    PATH="/app/.venv/bin:$PATH"

WORKDIR /app

COPY pyproject.toml uv.lock README.md ./
COPY src ./src
COPY web/__init__.py ./web/__init__.py
COPY web/backend ./web/backend

RUN python -m pip install --no-cache-dir "uv==0.12.21" && \
    uv sync --frozen --no-dev --extra web --extra distilbert-runtime \
      --no-editable --compile-bytecode && \
    useradd --create-home --shell /usr/sbin/nologin threatlens && \
    mkdir -p /var/cache/threatlens && \
    chown -R threatlens:threatlens /var/cache/threatlens

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/v1/live', timeout=3).read()"

USER threatlens

CMD ["python", "-m", "uvicorn", "web.backend.main:app", "--host", "0.0.0.0", "--port", "8000", "--proxy-headers", "--forwarded-allow-ips=*"]
