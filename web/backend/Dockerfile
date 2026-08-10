FROM python:3.12-slim@sha256:57cd7c3a7a273101a6485ba99423ee568157882804b1124b4dd04266317710de

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    EMAIL_THREAT_MODEL_PATH=artifacts/tfidf_logreg.joblib

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src
COPY artifacts/tfidf_logreg.joblib ./artifacts/tfidf_logreg.joblib
COPY web/__init__.py ./web/__init__.py
COPY web/backend ./web/backend

RUN python -m pip install --upgrade pip && \
    python -m pip install ".[web,distilbert-runtime]" && \
    useradd --create-home --shell /usr/sbin/nologin threatlens

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/v1/live', timeout=3).read()"

USER threatlens

CMD ["python", "-m", "uvicorn", "web.backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
