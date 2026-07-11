FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src
COPY examples ./examples
COPY doc ./doc

RUN pip install --no-cache-dir ".[api,yaml]" \
    && mkdir -p runs/api_sessions

EXPOSE 8000

ENTRYPOINT ["agenteval"]
CMD ["serve", "--host", "0.0.0.0", "--port", "8000", "--run-root", "/app/runs/api_sessions"]
