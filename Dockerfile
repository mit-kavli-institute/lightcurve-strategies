ARG PYTHON_VERSION=3.12
FROM python:${PYTHON_VERSION}-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

RUN pip install --no-cache-dir nox

COPY pyproject.toml ./
COPY src/ src/
COPY tests/ tests/
COPY noxfile.py ./

CMD ["nox", "-s", "tests"]
