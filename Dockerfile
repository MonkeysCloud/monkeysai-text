FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/app
COPY pyproject.toml ./
RUN pip install --no-cache-dir poetry==1.8.2 && poetry config virtualenvs.create false \
    && poetry install --no-interaction --no-ansi

COPY app ./app

ENV PORT=8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]