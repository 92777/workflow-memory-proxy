FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV MCPROXY_STORE_ENABLED=1
ENV MCPROXY_STORE_DB_PATH=/data/memory_proxy.db
ENV MCPROXY_STORE_MAX_REQUESTS=100

WORKDIR /app

COPY pyproject.toml README_CN.md /app/
COPY src /app/src
COPY storage /app/storage
COPY docs /app/docs

RUN pip install --no-cache-dir .

VOLUME ["/data"]

EXPOSE 8000

CMD ["memory-proxy"]
