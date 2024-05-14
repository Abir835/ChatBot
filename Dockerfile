ARG PYTHON_VERSION=3.11.9
FROM python:${PYTHON_VERSION}-slim as base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /apps


ARG UID=10001
RUN adduser \
    --disabled-password \
    --gecos "" \
    --home "/nonexistent" \
    --shell "/sbin/nologin" \
    --no-create-home \
    --uid "${UID}" \
    appuser

RUN --mount=type=cache,target=/root/.cache/pip \
    --mount=type=bind,source=requirements.txt,target=requirements.txt \
    pip install --upgrade pip && \
    pip install -r requirements.txt

USER appuser

COPY .env apps/app/.env

COPY apps/app.py .

COPY . .



EXPOSE 8000


CMD ["gunicorn", "app:app", "--bind=0.0.0.0:8000"]
