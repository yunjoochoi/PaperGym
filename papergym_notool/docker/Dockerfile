FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        git ca-certificates curl \
        libgl1 libglib2.0-0 libxcb1 libxext6 libxrender1 libsm6 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /papergym
COPY pyproject.toml .
COPY src ./src
COPY scripts ./scripts

RUN pip install -e .

# Pre warm docling
RUN python -c "from papergym.env.preparer import _get_docling_converter; _get_docling_converter()"

RUN mkdir -p /work /library /events
WORKDIR /work

ENTRYPOINT ["python", "/papergym/scripts/accumulate_one.py"]
