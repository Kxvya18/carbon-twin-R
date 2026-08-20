FROM python:3.12-slim

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
COPY api ./api
COPY simulator ./simulator
COPY dashboard ./dashboard
COPY configs ./configs
COPY docs ./docs
RUN pip install --no-cache-dir -e .

CMD ["uvicorn","api.main:app","--host","0.0.0.0","--port","8000"]
