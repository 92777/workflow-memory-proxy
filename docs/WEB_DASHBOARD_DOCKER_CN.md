# Web Dashboard + Docker

## What this adds

This project now includes a built-in web dashboard at `/dashboard`.

The dashboard supports:

- test chat calls against the proxy
- recent sessions and request audit records
- original vs forwarded payload comparison
- prompt memory and memory DSL inspection
- estimated input token savings
- upstream usage display when available
- English / Chinese UI toggle

## Storage retention

To avoid unbounded SQLite growth, the proxy now keeps only the most recent request audits.

Default:

- latest `100` request audits
- related raw messages, extracted events, and snapshots for older requests are pruned together

Configure it with:

```bash
export MCPROXY_STORE_MAX_REQUESTS=100
```

## Important requirement

The dashboard depends on SQLite audit storage.

Enable it with:

```bash
export MCPROXY_STORE_ENABLED=1
export MCPROXY_STORE_DB_PATH=/absolute/path/to/memory_proxy.db
```

## Run locally

```bash
cd /workspace/workflow-memory-proxy
export MCPROXY_UPSTREAM_BASE_URL=http://127.0.0.1:8317/v1/
export MCPROXY_UPSTREAM_API_KEY=your_api_key
export MCPROXY_STORE_ENABLED=1
export MCPROXY_STORE_DB_PATH=/workspace/workflow-memory-proxy/memory_proxy.db
export MCPROXY_STORE_MAX_REQUESTS=100
export PYTHONPATH=/workspace/workflow-memory-proxy/src
.venv/bin/python -m uvicorn memory_proxy.server:create_app --factory --host 0.0.0.0 --port 8000
```

Then open:

- `http://127.0.0.1:8000/dashboard`

## Docker build

```bash
cd /workspace/workflow-memory-proxy
docker build -t memory-compression-proxy:latest .
```

## Docker run

When the upstream model server is running on the host machine, Docker containers usually need `host.docker.internal` instead of `127.0.0.1`.

```bash
docker run --rm \
  -p 8000:8000 \
  -e MCPROXY_UPSTREAM_BASE_URL=http://host.docker.internal:8317/v1/ \
  -e MCPROXY_UPSTREAM_API_KEY=your_api_key \
  -e MCPROXY_STORE_ENABLED=1 \
  -e MCPROXY_STORE_DB_PATH=/data/memory_proxy.db \
  -e MCPROXY_STORE_MAX_REQUESTS=100 \
  -v "$(pwd)/data:/data" \
  --add-host=host.docker.internal:host-gateway \
  memory-compression-proxy:latest
```

Open:

- `http://127.0.0.1:8000/dashboard`

## Docker Compose

The repository includes [docker-compose.yml](/workspace/workflow-memory-proxy/docker-compose.yml).

Example:

```bash
cd /workspace/workflow-memory-proxy
export MCPROXY_UPSTREAM_BASE_URL=http://host.docker.internal:8317/v1/
export MCPROXY_UPSTREAM_API_KEY=your_api_key
export MCPROXY_STORE_MAX_REQUESTS=100
docker compose up --build
```

## Notes

- `127.0.0.1` inside a container points to the container itself, not your host.
- The dashboard shows request audits only after at least one request has gone through the proxy.
- Streaming requests are proxied, but the dashboard currently stores richer usage details mainly for non-stream JSON responses.
