FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src
COPY examples ./examples
COPY doc ./doc

RUN pip install --no-cache-dir -e . "uvicorn[standard]>=0.23" \
    && mkdir -p runs/api_sessions

EXPOSE 8000

CMD ["uvicorn", "agenteval.api:app", "--host", "0.0.0.0", "--port", "8000"]
