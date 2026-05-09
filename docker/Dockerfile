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

# Pre-warm docling: the runtime constructs the converter with table-structure
# enabled (preparer.py), so we instantiate it the same way at build time so
# both the layout and TableFormer weights are baked into the image instead of
# 4 workers racing to download them from HF Hub on first paper.
RUN python -c "from papergym.env.preparer import _get_docling_converter; _get_docling_converter()"

RUN mkdir -p /work /library /events
WORKDIR /work

ENTRYPOINT ["python", "/papergym/scripts/accumulate_one.py"]
