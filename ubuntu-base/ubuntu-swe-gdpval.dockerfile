# syntax=docker/dockerfile:1
# ubuntu-swe-gdpval — GDPval agentic file-producing sandbox variant.
#
# Thin layer over the built ubuntu-swe image (conv #3: light, fast-changing deps
# go in an image layer via build-time uv, never a Ray runtime venv). Adds the
# document-production toolchain the GDPval agentic environment expects: python
# libraries to author xlsx/docx/pptx/pdf and pandoc for cross-format conversion.
# The in-container deliverable extractor (environments/_gdpval_extract.py) is
# shipped by the environment at grading time and uses these same libraries to read
# the produced files back to text — so a build that imports them cleanly gates
# both production and grading. Select via the data.*.image / env.gdpval_agentic.image
# config refs.
ARG BASE_IMAGE=ubuntu-swe:latest
FROM ${BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# pandoc for document conversion (e.g. Markdown → docx/pdf). LibreOffice is NOT
# installed (large); the python libraries below cover native authoring of each
# target format, which is what the extractor reads back.
RUN apt-get update && apt-get install -y --no-install-recommends \
    pandoc \
    && rm -rf /var/lib/apt/lists/*

# Document authoring + readback libraries, installed into the system Python
# (no virtualenv — conv #3).
RUN --mount=type=cache,target=/root/.cache/uv uv pip install --system \
    openpyxl \
    python-docx \
    python-pptx \
    reportlab \
    pypdf \
    pandas \
    matplotlib \
    markitdown \
    Pillow

# Gate dependency breakage at build time: every library the extractor dispatches
# on must import cleanly.
RUN python3 -c "import openpyxl, docx, pptx, reportlab, pypdf, pandas, matplotlib; print('gdpval doc toolchain OK')"
